"""ищем статьи по смыслу запроса через gemma в mws и qdrant."""
# поток поиска: запрос → инструкция gemma → вектор mws → статьи из qdrant
from __future__ import annotations

import argparse
import math
import os
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from habr_search.ingestion.chunking import DEFAULT_TOKENIZER, ChunkingConfig, load_tokenizer
from habr_search.embeddings import DEFAULT_MODEL, DEFAULT_URL, EmbeddingConfig, EmbeddingsClient, read_api_key
from habr_search.ingestion.index import VECTOR_SIZE, collection_name
from habr_search.retrieval.spelling import DEFAULT_DICTIONARY, load_dictionary
from habr_search.vector_store import connect_store, resolve_qdrant_config
from habr_search.index_version import index_revision

QUERY_INSTRUCTION: str = 'Given a web search query, retrieve relevant passages that answer the query.'
MAX_QUERY_TOKENS: int = 1024


def validate_min_score(value: float | None) -> None:
    """порог косинусного сходства должен быть конечным числом от минус одного до одного."""
    if value is not None and (not math.isfinite(value) or not -1 <= value <= 1):
        raise ValueError('min-score должен быть конечным числом от -1 до 1')


# результат представляет статью с её лучшим фрагментом
@dataclass(frozen=True)
class SearchResult:
    article_id: int  # номер исходной статьи
    title: str  # заголовок для отображения
    url: str  # ссылка на оригинал
    text: str  # текст лучшего найденного фрагмента
    score: float  # оценка алгоритма поиска, а не вероятность правильного ответа


def build_query(query: str, raw: bool = False) -> str:
    """добавляем поисковую инструкцию gemma; документы индексируются без неё."""
    query = query.strip()
    if not query:
        raise ValueError('поисковый запрос пуст')
    if raw:
        return query
    return f'<instruct>{QUERY_INSTRUCTION}\n<query>{query}'


# проверяем, что нужная коллекция существует, совместима и содержит точки
def validate_collection(store: QdrantClient, name: str, vector_size: int = VECTOR_SIZE) -> None:
    # поиск не создаёт и не изменяет коллекцию
    if not store.collection_exists(name):
        raise ValueError(f'коллекция {name} не найдена; проверьте адрес qdrant и настройки индексации')
    vectors = store.get_collection(name).config.params.vectors
    if (not isinstance(vectors, models.VectorParams) or vectors.size != vector_size
            or vectors.distance != models.Distance.COSINE):
        raise ValueError('коллекция использует другую размерность или метрику')
    if store.count(name, exact=True).count == 0:
        raise ValueError('коллекция пуста; сначала проиндексируйте статьи')


# ищем близкие векторы и получаем разные статьи с лучшими фрагментами
def search_articles(
    store: QdrantClient, name: str, vector: list[float], top_k: int = 5,
    vector_size: int = VECTOR_SIZE,
    *, min_score: float | None = None,
) -> list[SearchResult]:
    if top_k < 1:
        raise ValueError('top-k должен быть положительным')
    validate_min_score(min_score)
    if len(vector) != vector_size or not all(math.isfinite(value) for value in vector) or not any(vector):
        raise ValueError(f'нужен ненулевой конечный вектор размерности {vector_size}')
    # группируем на стороне qdrant: одна статья занимает только одну строку выдачи
    response = store.query_points_groups(
        collection_name=name, query=vector, group_by='article_id', group_size=1,
        limit=top_k, with_payload=['article_id', 'title', 'url', 'text'], with_vectors=False,
        score_threshold=min_score,
    )
    results: list[SearchResult] = []
    for group in response.groups:
        if not group.hits:
            continue
        hit = group.hits[0]
        # payload содержит текст и метаданные, сохранённые вместе с вектором
        payload = hit.payload or {}
        article_id: object = payload.get('article_id')
        title: object = payload.get('title')
        url: object = payload.get('url')
        text: object = payload.get('text')
        if (type(article_id) is not int or article_id < 1 or not isinstance(title, str)
                or not isinstance(url, str) or not isinstance(text, str)):
            raise ValueError('в найденной точке отсутствуют метаданные статьи; проверьте индекс')
        results.append(SearchResult(article_id, title, url, text, float(hit.score)))
    return results


# настройки запроса, коллекции и отображения результатов
class CliArgs(argparse.Namespace):
    query: str
    qdrant_url: str | None
    qdrant_path: Path | None
    tokenizer_file: Path
    top_k: int
    max_tokens: int
    overlap_tokens: int
    snippet_chars: int
    raw_query: bool
    spelling_mode: str
    spelling_db: Path
    min_score: float | None


# готовим запрос, получаем его вектор и печатаем найденные статьи
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--query', required=True, help='поисковый запрос')
    location = parser.add_mutually_exclusive_group()
    location.add_argument('--qdrant-url', help='адрес сервера qdrant')
    location.add_argument('--qdrant-path', type=Path, help='папка встроенного qdrant')
    parser.add_argument('--tokenizer-file', type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument('--top-k', type=int, default=5, help='сколько разных статей показать')
    parser.add_argument('--min-score', type=float, help='не показывать результаты ниже выбранного порога сходства')
    parser.add_argument('--max-tokens', type=int, default=1024, help='должно совпадать с настройкой индексатора')
    parser.add_argument('--overlap-tokens', type=int, default=128, help='должно совпадать с настройкой индексатора')
    parser.add_argument('--snippet-chars', type=int, default=400, help='длина отрывка в выводе')
    parser.add_argument('--raw-query', action='store_true', help='для сервера, который сам добавляет инструкцию gemma')
    parser.add_argument('--spelling-mode', choices=('off', 'suggest', 'apply'), default='off',
                        help='off: исходный запрос; suggest: показать варианты; apply: применить однозначные исправления')
    parser.add_argument('--spelling-db', type=Path, default=DEFAULT_DICTIONARY, help='словарь из python -m habr_search.retrieval.spellcheck build')
    args: CliArgs = parser.parse_args(namespace=CliArgs())
    if args.top_k < 1 or args.snippet_chars < 1:
        parser.error('top-k и snippet-chars должны быть положительными')
    try:
        validate_min_score(args.min_score)
        query = build_query(args.query, args.raw_query)
        chunking = ChunkingConfig(args.max_tokens, args.overlap_tokens)
        config = resolve_qdrant_config(args.qdrant_url, args.qdrant_path)
        if config.path is not None and not config.path.is_dir():
            raise ValueError('локальное хранилище не найдено; укажите адрес сервера через --qdrant-url')
        url = os.environ.get('MWS_EMBEDDINGS_URL', DEFAULT_URL)
        deployment = os.environ.get('MWS_EMBEDDINGS_MODEL', DEFAULT_MODEL)
        name = collection_name(url, deployment, chunking)
        with closing(connect_store(config)) as store:
            # проверяем индекс до запроса ключа и обращения в платный api
            validate_collection(store, name)
            if args.spelling_mode != 'off':
                dictionary = load_dictionary(args.spelling_db)
                dictionary.validate_context(name, store.count(name, exact=True).count, index_revision(store, name))
                spelling = dictionary.suggest(args.query)
                print(f'исходный запрос: {spelling.original}')
                for suggestion in spelling.suggestions:
                    options = ', '.join(item.word for item in suggestion.candidates)
                    print(f'возможное исправление {suggestion.original}: {options}')
                if args.spelling_mode == 'apply':
                    query = build_query(spelling.corrected, args.raw_query)
                    print(f'запрос для поиска: {spelling.corrected}')
            tokenizer = load_tokenizer(args.tokenizer_file)
            # длинный запрос отклоняем целиком, чтобы скрытое обрезание не меняло его смысл
            if tokenizer.count(query) > MAX_QUERY_TOKENS:
                raise ValueError(f'запрос вместе с инструкцией превышает {MAX_QUERY_TOKENS} токенов; сократите его')
            key = read_api_key()
            with closing(EmbeddingsClient(EmbeddingConfig(key, url, deployment))) as embeddings:
                vector = embeddings.embed([query])[0]
            results = search_articles(store, name, vector, args.top_k, min_score=args.min_score)
        print(f'коллекция: {name}')
        print(f'найдено статей: {len(results)}')
        for number, result in enumerate(results, 1):
            print(f'\n{number}. {result.title}')
            print(f'сходство: {result.score:.4f}')
            print(result.url)
            # сокращаем только отображение, исходный текст в индексе не меняется
            snippet = ' '.join(result.text.split())
            print(snippet[:args.snippet_chars] + ('…' if len(snippet) > args.snippet_chars else ''))
        return 0
    except KeyboardInterrupt:
        print('\nпоиск остановлен')
        return 130
    except UnexpectedResponse as exc:
        parser.exit(1, f'qdrant вернул http {exc.status_code}; проверьте настройки сервера\n')
    except ResponseHandlingException:
        parser.exit(1, 'не удалось связаться с qdrant; проверьте адрес и запущен ли контейнер\n')
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f'поиск не выполнен: {exc}\n')


# при прямом запуске выполняем команду; при импорте этот блок пропускается
if __name__ == '__main__':
    raise SystemExit(main())
