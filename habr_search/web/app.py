"""запускаем локальную страницу и api семантического поиска."""
from __future__ import annotations

import argparse
import math
import logging
import os
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import ExitStack, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Protocol

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, ConfigDict, Field
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from habr_search.ingestion.chunking import ChunkingConfig, TextTokenizer, load_tokenizer
from habr_search.embeddings import DEFAULT_URL, EmbeddingAPIError, EmbeddingConfig, EmbeddingsClient, read_api_key
from habr_search.ingestion.index import EmbeddingProvider, collection_name
from habr_search.retrieval.search import (
    build_query, search_articles, validate_collection, validate_min_score,
)
from habr_search.retrieval.spelling import DEFAULT_DICTIONARY, MAX_QUERY_CHARS, SpellingDictionary, load_dictionary
from habr_search.vector_store import QdrantConfig, connect_store, resolve_qdrant_config
from qdrant_client import QdrantClient
from habr_search.index_version import index_revision
from habr_search.paths import DATA_DIR
from habr_search.retrieval.hybrid import HybridConfig, HybridRetriever
from habr_search.retrieval.tantivy_index import literal_query
from habr_search.model_profiles import BGE_M3, GEMMA, ModelProfile, deployment_name, get_profile

STATIC_DIR: Path = Path(__file__).resolve().parent / 'static'


# строгие схемы исключают случайные поля и преобразование строк в числа или флаги
class QueryRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)


# поисковый запрос дополнительно задаёт размер выдачи и применение исправлений
class SearchRequest(QueryRequest):
    top_k: int = Field(default=5, ge=1, le=20)
    correct_typos: bool = False


# отдаём предложения опечаточника без обращения к модели
class SuggestionResponse(BaseModel):
    original_query: str
    corrected_query: str
    available: bool
    alternatives: dict[str, list[str]]


# одна карточка выдачи: статья, подходящий отрывок и оценка выбранного поиска
class ArticleResponse(BaseModel):
    article_id: int
    title: str
    url: str
    snippet: str
    score: float


# возвращаем выдачу исходной фразы и отдельное предложение, которое пользователь может выбрать
class SearchResponse(BaseModel):
    original_query: str
    used_query: str
    corrected: bool
    results: list[ArticleResponse]
    elapsed_ms: float
    suggested_query: str | None = None
    alternatives: dict[str, list[str]] = Field(default_factory=dict)
    deferred_suggestion: bool = False  # локальная подсказка загружается после выдачи


# сведения об индексе и доступности словаря на момент запуска сервиса
class HealthResponse(BaseModel):
    collection: str
    fragments: int
    articles: int | None
    spelling_available: bool


class SearchBusyError(RuntimeError):
    """один запрос уже выполняется; новый не отправляется в платный api."""


# общий интерфейс настоящего сервиса и его подстановки в http-тестах
class SearchBackend(Protocol):
    # читаем сведения о загруженном сервисе
    def health(self) -> HealthResponse: ...
    # получаем локальные предложения для запроса
    def suggest(self, query: str) -> SuggestionResponse: ...
    # выполняем поиск с заданными параметрами
    def search(self, request: SearchRequest) -> SearchResponse: ...
    # освобождаем ресурсы сервиса при остановке
    def close(self) -> None: ...


class SearchService:
    """переиспользуем подключения и словарь, выполняем один запрос поиска за раз."""
    # проверяем совместимость индекса и словаря до обработки пользовательских запросов
    def __init__(self, store: QdrantClient, name: str, tokenizer: TextTokenizer,
                 embeddings: EmbeddingProvider, dictionary: SpellingDictionary | None,
                 raw_query: bool = False, min_score: float | None = None,
                 profile: ModelProfile = GEMMA) -> None:
        validate_min_score(min_score)
        validate_collection(store, name, profile.vector_size)
        points = store.count(name, exact=True).count
        if dictionary is not None:
            try:
                dictionary.validate_context(name, points, index_revision(store, name))
            except ValueError:
                # старый словарь отключает только подсказки, исходный поиск остаётся доступен
                logging.warning('словарь устарел; подсказки отключены до пересборки')
                dictionary = None
        self.profile: ModelProfile = profile
        self.store: QdrantClient = store
        self.name: str = name
        self.tokenizer: TextTokenizer = tokenizer
        self.embeddings: EmbeddingProvider = embeddings
        self.dictionary: SpellingDictionary | None = dictionary
        self.raw_query: bool = raw_query
        self.min_score: float | None = min_score
        self.lock: Lock = Lock()
        self.status: HealthResponse = HealthResponse(
            collection=name, fragments=points, articles=dictionary.info.articles if dictionary else None,
            spelling_available=dictionary is not None,
        )
        self.resources: ExitStack = ExitStack()

    # возвращаем подготовленные сведения без сетевых проверок
    def health(self) -> HealthResponse:
        # это сведения на момент запуска, а не платная проверка модели
        return self.status

    # проверяем фразу и возвращаем варианты из уже загруженного словаря
    def suggest(self, query: str) -> SuggestionResponse:
        build_query(query, profile=self.profile)
        if self.dictionary is not None:
            try:
                self.dictionary.validate_context(self.name, self.store.count(self.name, exact=True).count,
                                                 index_revision(self.store, self.name))
            except ValueError:
                return SuggestionResponse(original_query=query, corrected_query=query, available=False, alternatives={})
        if self.dictionary is None:
            return SuggestionResponse(original_query=query, corrected_query=query, available=False, alternatives={})
        result = self.dictionary.suggest(query)
        return SuggestionResponse(original_query=query, corrected_query=result.corrected, available=True,
                                  alternatives={item.original: [candidate.word for candidate in item.candidates]
                                                for item in result.suggestions})

    def _prepare_spelling(self, query: str, correct_typos: bool) -> tuple[str, dict[str, list[str]], str | None, bool]:
        """подготавливаем подсказки, сохраняя прежнее поведение семантических режимов."""
        alternatives: dict[str, list[str]] = {}
        suggested_query: str | None = None
        dictionary_available = self.dictionary is not None
        if self.dictionary is not None:
            try:
                self.dictionary.validate_context(self.name, self.store.count(self.name, exact=True).count,
                                                 index_revision(self.store, self.name))
            except ValueError:
                # устаревший словарь не мешает искать исходную фразу, но не даёт подсказок
                if correct_typos:
                    raise
                dictionary_available = False
        if correct_typos and not dictionary_available:
            raise ValueError('словарь опечаток не загружен')
        if self.dictionary is not None and dictionary_available:
            spelling = self.dictionary.suggest(query)
            if correct_typos:
                query = spelling.corrected
            elif spelling.corrected != query:
                suggested_query = spelling.corrected
            # неприменённые варианты показываем отдельно, не меняя ими запрос
            alternatives = {item.original: [candidate.word for candidate in item.candidates]
                            for item in spelling.suggestions if item.replacement is None}
        return query, alternatives, suggested_query, False

    def _retrieve(self, query: str, vector: list[float], top_k: int) -> list[ArticleResponse]:
        """обычный режим использует только семантическую выдачу."""
        found = search_articles(self.store, self.name, vector, top_k, self.profile.vector_size, min_score=self.min_score)
        results: list[ArticleResponse] = []
        for item in found:
            text = ' '.join(item.text.split())
            snippet = text[:500] + ('…' if len(text) > 500 else '')
            results.append(ArticleResponse(article_id=item.article_id, title=item.title, url=item.url,
                                           snippet=snippet, score=item.score))
        return results

    # ищем исходную фразу и отдельно предлагаем исправление; замена требует явного флага api
    def search(self, request: SearchRequest) -> SearchResponse:
        if not self.lock.acquire(blocking=False):
            raise SearchBusyError('поиск уже выполняется; дождитесь результата')
        try:
            start = perf_counter()
            validate_collection(self.store, self.name, self.profile.vector_size)
            query, alternatives, suggested_query, deferred = self._prepare_spelling(request.query.strip(), request.correct_typos)
            prepared = build_query(query, self.raw_query, self.profile)
            if self.tokenizer.count(prepared) > self.profile.query_max_tokens:
                raise ValueError(f'запрос вместе с инструкцией превышает {self.profile.query_max_tokens} токенов')
            # только этот вызов обращается к модели; повторов и фоновых поисков нет
            vector = self.embeddings.embed([prepared])[0]
            results = self._retrieve(query, vector, request.top_k)
            return SearchResponse(original_query=request.query, used_query=query, corrected=query != request.query.strip(),
                                  results=results, elapsed_ms=round((perf_counter() - start) * 1000, 1),
                                  alternatives=alternatives, suggested_query=suggested_query, deferred_suggestion=deferred)
        finally:
            self.lock.release()

    # закрываем подключения, переданные сервису при его создании
    def close(self) -> None:
        self.resources.close()


class TantivyService:
    """ищем локально; вычисление подсказки не задерживает основную выдачу."""
    def __init__(self, path: Path, spelling: bool = True) -> None:
        from habr_search.retrieval.tantivy_index import TantivySearch
        self.backend: TantivySearch = TantivySearch(path)
        self.dictionary: SpellingDictionary | None = load_dictionary(path / 'spelling.sqlite3') if spelling else None
        if self.dictionary is not None:
            self.dictionary.validate_context('tantivy', self.backend.info.articles, self.backend.info.revision)

    def health(self) -> HealthResponse:
        return HealthResponse(collection='tantivy', fragments=self.backend.info.articles,
                              articles=self.backend.info.articles, spelling_available=self.dictionary is not None)

    def suggest(self, query: str) -> SuggestionResponse:
        from habr_search.retrieval.tantivy_index import literal_query
        literal_query(query)
        if self.dictionary is None:
            return SuggestionResponse(original_query=query, corrected_query=query, available=False, alternatives={})
        spelling = self.dictionary.suggest(query)
        return SuggestionResponse(original_query=query, corrected_query=spelling.corrected, available=True,
                                  alternatives={item.original: [candidate.word for candidate in item.candidates]
                                                for item in spelling.suggestions})

    def search(self, request: SearchRequest) -> SearchResponse:
        start = perf_counter()
        query = request.query.strip()
        if request.correct_typos:
            suggestion = self.suggest(query)
            if not suggestion.available:
                raise ValueError('словарь опечаток не загружен')
            query = suggestion.corrected_query
        found = self.backend.search(query, request.top_k)
        return SearchResponse(original_query=request.query, used_query=query,
                              corrected=query != request.query.strip(),
                              results=[ArticleResponse(article_id=item.article_id, title=item.title,
                                                       url=item.url, snippet=item.snippet, score=item.score)
                                       for item in found],
                              elapsed_ms=round((perf_counter() - start) * 1000, 1),
                              deferred_suggestion=self.dictionary is not None and not request.correct_typos)

    def close(self) -> None:
        # снимок индекса не содержит открытого писателя или сетевых подключений
        pass


class HybridService(SearchService):
    """сохраняем общий путь с одним вызовом модели и отдельной локальной подсказкой."""
    def __init__(self, store: QdrantClient, name: str, tokenizer: TextTokenizer,
                 embeddings: EmbeddingProvider, local: TantivyService,
                 config: HybridConfig = HybridConfig()) -> None:
        super().__init__(store, name, tokenizer, embeddings, None, profile=BGE_M3)
        self.local: TantivyService = local
        self.hybrid: HybridRetriever = HybridRetriever(store, name, local.backend, config)
        self.status.articles = local.health().articles
        self.status.spelling_available = local.health().spelling_available

    def search(self, request: SearchRequest) -> SearchResponse:
        if request.top_k > self.hybrid.config.candidate_limit:
            raise ValueError('top-k не должен превышать candidate-limit')
        return super().search(request)

    def suggest(self, query: str) -> SuggestionResponse:
        return self.local.suggest(query)

    def _prepare_spelling(self, query: str, correct_typos: bool) -> tuple[str, dict[str, list[str]], str | None, bool]:
        # проверяем источники и длину до платного вызова; исходный текст не подменяем
        self.hybrid.check_context()
        literal_query(query)
        if correct_typos:
            suggestion = self.suggest(query)
            if not suggestion.available:
                raise ValueError('словарь опечаток не загружен')
            return suggestion.corrected_query, suggestion.alternatives, None, False
        return query, {}, None, self.status.spelling_available

    def _retrieve(self, query: str, vector: list[float], top_k: int) -> list[ArticleResponse]:
        run = self.hybrid.search(query, vector, top_k)
        return [ArticleResponse(article_id=item.article_id, title=item.title, url=item.url,
                                snippet=item.snippet, score=item.score) for item in run.results]


# настройки подключений, локальных файлов и подготовки поискового запроса
@dataclass(frozen=True)
class WebConfig:
    store: QdrantConfig
    embeddings: EmbeddingConfig
    tokenizer_file: Path | None = None
    dictionary_file: Path | None = DEFAULT_DICTIONARY
    chunking: ChunkingConfig = ChunkingConfig()
    raw_query: bool = False
    min_score: float | None = None  # выбранный порог задаёт сервер, а не браузер
    profile: ModelProfile = GEMMA


def connect_service(config: WebConfig) -> SearchService:
    """проверяем локальные данные до создания клиента модели, освобождаем ресурсы при сбое."""
    with ExitStack() as resources:
        store = connect_store(config.store)
        resources.callback(store.close)
        name = collection_name(config.embeddings.url, config.embeddings.model, config.chunking, config.profile)
        validate_collection(store, name, config.profile.vector_size)
        tokenizer = load_tokenizer(config.tokenizer_file, config.profile)
        dictionary = load_dictionary(config.dictionary_file) if config.dictionary_file is not None else None
        embeddings = EmbeddingsClient(config.embeddings)
        resources.callback(embeddings.close)
        service = SearchService(store, name, tokenizer, embeddings, dictionary, config.raw_query, config.min_score, config.profile)
        service.resources = resources.pop_all()
        return service



def connect_hybrid_service(config: WebConfig, path: Path, spelling: bool,
                           hybrid_config: HybridConfig = HybridConfig()) -> HybridService:
    """сбой подготовки освобождает подключения до первого платного запроса."""
    with ExitStack() as resources:
        store = connect_store(config.store)
        resources.callback(store.close)
        name = collection_name(config.embeddings.url, config.embeddings.model, config.chunking, BGE_M3)
        local = TantivyService(path, spelling)
        tokenizer = load_tokenizer(config.tokenizer_file, BGE_M3)
        embeddings = EmbeddingsClient(config.embeddings)
        resources.callback(embeddings.close)
        service = HybridService(store, name, tokenizer, embeddings, local, hybrid_config)
        service.resources = resources.pop_all()
        return service


def create_app(factory: Callable[[], SearchBackend]) -> FastAPI:
    """фабрика позволяет проверять весь http-путь без настоящего ключа и вызовов mws."""
    active: dict[str, SearchBackend] = {}

    # создаём сервис один раз при запуске и закрываем его при остановке приложения
    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
        service = factory()
        active['service'] = service
        try:
            yield
        finally:
            active.clear()
            service.close()

    app = FastAPI(title='Поиск по Хабру', version='1.0', lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1', 'localhost', 'testserver'])
    app.mount('/static', StaticFiles(directory=STATIC_DIR), name='static')

    # проверяем источник запроса перед выполнением действий через локальный api
    @app.middleware('http')
    async def check_origin(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        # сторонняя страница не должна запускать платные поиски через локальный сервер
        origin = request.headers.get('origin')
        if request.url.path.startswith('/api/') and request.method == 'POST' and origin is not None:
            expected = f'{request.url.scheme}://{request.headers.get("host", "")}'
            if origin != expected:
                return JSONResponse({'detail': 'запрос разрешён только со страницы этого сервера'}, status_code=403)
        return await call_next(request)

    # не допускаем обращение к сервису до завершения его загрузки
    def service() -> SearchBackend:
        if 'service' not in active:
            raise HTTPException(503, 'поиск ещё не готов')
        return active['service']

    # отдаём страницу, стили и скрипты подключаются с этого же сервера
    @app.get('/', include_in_schema=False)
    def home() -> FileResponse:
        return FileResponse(STATIC_DIR / 'index.html')

    # передаём странице сведения о корпусе и словаре
    @app.get('/api/health')
    def health() -> HealthResponse:
        return service().health()

    # преобразуем ошибку проверки фразы в понятный http-ответ
    @app.post('/api/suggest')
    def suggest(request: QueryRequest) -> SuggestionResponse:
        try:
            return service().suggest(request.query)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None

    # сопоставляем сбои поиска с http-статусами, не раскрывая ключ модели
    @app.post('/api/search')
    def search(request: SearchRequest) -> SearchResponse:
        try:
            return service().search(request)
        except SearchBusyError as exc:
            raise HTTPException(429, str(exc)) from None
        except EmbeddingAPIError as exc:
            raise HTTPException(502, str(exc)) from None
        except (UnexpectedResponse, ResponseHandlingException):
            raise HTTPException(503, 'не удалось связаться с qdrant; проверьте контейнер') from None
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None

    return app


# параметры запуска локального сервера и подключения к индексу
class CliArgs(argparse.Namespace):
    engine: str
    tantivy_path: Path
    qdrant_url: str | None
    qdrant_path: Path | None
    tokenizer_file: Path | None
    spelling_db: Path | None
    without_spelling: bool
    port: int
    timeout: float
    raw_query: bool
    max_tokens: int
    overlap_tokens: int
    min_score: float | None
    candidate_limit: int
    rrf_k: float


# читаем настройки и ключ в терминале, затем запускаем приложение на локальном адресе
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', choices=['gemma', 'bge-m3', 'tantivy', 'hybrid'], default='gemma')
    parser.add_argument('--tantivy-path', type=Path, default=DATA_DIR / 'tantivy')
    location = parser.add_mutually_exclusive_group()
    location.add_argument('--qdrant-url')
    location.add_argument('--qdrant-path', type=Path)
    parser.add_argument('--tokenizer-file', type=Path)
    parser.add_argument('--spelling-db', type=Path)
    parser.add_argument('--without-spelling', action='store_true')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--raw-query', action='store_true')
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--overlap-tokens', type=int, default=128)
    parser.add_argument('--candidate-limit', type=int, default=20, help='кандидатов каждого источника в гибриде')
    parser.add_argument('--rrf-k', type=float, default=60.0, help='сглаживание мест в гибриде')
    parser.add_argument('--min-score', type=float, help='проверенный порог сходства; по умолчанию выключен')
    args: CliArgs = parser.parse_args(namespace=CliArgs())
    if not 1 <= args.port <= 65535 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('нужны допустимый port и положительный конечный timeout')
    try:
        if args.engine == 'tantivy':
            if args.min_score is not None or args.raw_query or args.qdrant_url or args.qdrant_path:
                raise ValueError('параметры qdrant, raw-query и min-score относятся к семантическому режиму')
            # ключ, токенизатор, docker и api модели для этого режима не нужны
            print(f'страница поиска: http://127.0.0.1:{args.port}')
            uvicorn.run(create_app(lambda: TantivyService(args.tantivy_path, not args.without_spelling)),
                        host='127.0.0.1', port=args.port, log_level='warning', access_log=False)
            return 0
        hybrid_config = HybridConfig(args.candidate_limit, args.rrf_k)
        if args.engine == 'hybrid' and (args.min_score is not None or args.raw_query or args.spelling_db is not None):
            raise ValueError('гибрид не использует min-score, raw-query и spelling-db; словарь берётся из tantivy')
        if args.engine == 'hybrid' and args.candidate_limit < 20:
            raise ValueError('для веб-выдачи до 20 статей candidate-limit должен быть не меньше 20')
        validate_min_score(args.min_score)
        store = resolve_qdrant_config(args.qdrant_url, args.qdrant_path)
        if store.path is not None and not store.path.is_dir():
            raise ValueError('локальный qdrant не найден; укажите --qdrant-url')
        profile = get_profile('bge-m3' if args.engine in ('bge-m3', 'hybrid') else 'gemma2')
        chunking = ChunkingConfig(args.max_tokens, args.overlap_tokens)
        key = read_api_key()
        embeddings = EmbeddingConfig(key, os.environ.get('MWS_EMBEDDINGS_URL', DEFAULT_URL),
                                     deployment_name(profile), args.timeout)
        config = WebConfig(store, embeddings, args.tokenizer_file,
                           None if args.without_spelling else (args.spelling_db or profile.dictionary_file),
                           chunking, args.raw_query, args.min_score, profile)
        print(f'страница поиска: http://127.0.0.1:{args.port}')
        print('получение выдачи обращается к mws; открытие страницы и проверка опечаток — нет')
        factory: Callable[[], SearchBackend]
        if args.engine == 'hybrid':
            factory = lambda: connect_hybrid_service(config, args.tantivy_path, not args.without_spelling, hybrid_config)
        else:
            factory = lambda: connect_service(config)
        uvicorn.run(create_app(factory), host='127.0.0.1', port=args.port,
                    log_level='warning', access_log=False)
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f'сервер не запущен: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
