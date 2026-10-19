"""собираем общие пакеты и параллельно получаем векторы, записывая qdrant в одном потоке."""
from __future__ import annotations

import math
from collections.abc import Callable, Generator, Iterable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

from qdrant_client import QdrantClient, models

from habr_search.index_version import invalidate_index
from habr_search.ingestion.chunking import (
    CHUNKING_VERSION, MODEL_ID, MODEL_REVISION, ArticleChunk, ChunkingConfig, TextTokenizer, chunk_article,
)
from habr_search.ingestion.prepare import PreparedArticle

VECTOR_SIZE: int = 3584  # размерность проверенного деплоймента parser


class EmbeddingProvider(Protocol):
    """клиент возвращает один вектор на каждый текст в исходном порядке."""
    def embed(self, texts: list[str]) -> list[list[float]]:
        """вычисляем векторы одного пакета; повторов внутри очереди нет."""
        ...


@dataclass
class IndexStats:
    """считаем только подтверждённые новые точки и ранее сохранённые фрагменты."""
    saved: int = 0
    skipped: int = 0


@dataclass(frozen=True)
class IndexProgress:
    """снимок прогресса: статья готова только после записи всех её фрагментов и очистки старых."""
    articles: int
    saved: int
    skipped: int
    batches: int
    elapsed_seconds: float


@dataclass
class ArticleState:
    """помним ожидаемые точки статьи до завершения всех пакетов с её фрагментами."""
    article_id: int
    ids: list[str]
    remaining: int


@dataclass(frozen=True)
class PendingChunk:
    """связываем фрагмент с отпечатком текста и состоянием его статьи."""
    chunk: ArticleChunk
    text_hash: str
    article: ArticleState


def save_chunks(store: QdrantClient, name: str, chunks: Sequence[tuple[ArticleChunk, str]],
                vectors: list[list[float]], vector_size: int) -> None:
    """проверяем весь ответ до записи и сохраняем пакет с подтверждением qdrant."""
    if (len(vectors) != len(chunks) or any(len(vector) != vector_size or not any(vector)
            or not all(math.isfinite(value) for value in vector) for vector in vectors)):
        raise ValueError(f'ожидались ненулевые конечные векторы размерности {vector_size}; проверьте деплоймент')
    points = [models.PointStruct(id=chunk.id, vector=vector, payload={
        'article_id': chunk.article_id, 'chunk_index': chunk.chunk_index,
        'title': chunk.title, 'url': chunk.url, 'text': chunk.text,
        'hubs': list(chunk.hubs), 'published_at': chunk.published_at,
        'start_char': chunk.start_char, 'end_char': chunk.end_char,
        'token_count': chunk.token_count, 'text_hash': text_hash,
        'model': MODEL_ID, 'model_revision': MODEL_REVISION, 'chunking_version': CHUNKING_VERSION,
    }) for (chunk, text_hash), vector in zip(chunks, vectors)]
    invalidate_index(store, name)
    store.upsert(collection_name=name, points=points, wait=True)


def remove_stale_chunks(store: QdrantClient, name: str, article_id: int, ids: list[str]) -> None:
    """удаляем старую версию статьи только после подтверждения всех ожидаемых точек."""
    stale = models.Filter(
        must=[models.FieldCondition(key='article_id', match=models.MatchValue(value=article_id))],
        must_not=[models.HasIdCondition(has_id=[item for item in ids])],
    )
    if store.count(name, exact=True, count_filter=stale).count:
        invalidate_index(store, name)
        store.delete(name, points_selector=models.FilterSelector(filter=stale), wait=True)


class IndexCoordinator:
    """читаем корпус и пишем индекс в вызывающем потоке; работникам передаём только тексты."""
    def __init__(self, store: QdrantClient, name: str, vector_size: int,
                 on_progress: Callable[[IndexProgress], None] | None) -> None:
        """создаём счётчики и ограниченную таблицу незавершённых запросов."""
        self.store: QdrantClient = store
        self.name: str = name
        self.vector_size: int = vector_size
        self.on_progress: Callable[[IndexProgress], None] | None = on_progress
        self.started: float = perf_counter()
        self.articles: int = 0
        self.stats: IndexStats = IndexStats()
        self.batches: int = 0
        self.active: dict[Future[list[list[float]]], tuple[int, list[PendingChunk]]] = {}

    def progress(self) -> IndexProgress:
        """показываем скорость только фактической обработки текущего запуска."""
        return IndexProgress(self.articles, self.stats.saved, self.stats.skipped, self.batches,
                             perf_counter() - self.started)

    def notify(self) -> None:
        """передаём готовый снимок, не позволяя внешнему коду изменять счётчики."""
        if self.on_progress is not None:
            self.on_progress(self.progress())

    def complete_article(self, state: ArticleState) -> None:
        """очищаем устаревшие точки только полностью обработанной статьи."""
        remove_stale_chunks(self.store, self.name, state.article_id, state.ids)
        self.articles += 1

    def pending_chunks(self, articles: Iterable[PreparedArticle], tokenizer: TextTokenizer,
                       config: ChunkingConfig, stop_requested: Callable[[], bool] | None) -> Generator[PendingChunk, None, None]:
        """читаем статьи по мере заполнения очереди, пропуская готовые точки до оплаты api."""
        for article in articles:
            if stop_requested is not None and stop_requested():
                raise KeyboardInterrupt
            chunks = list(chunk_article(article, tokenizer, config))
            records = self.store.retrieve(self.name, ids=[chunk.id for chunk in chunks],
                                          with_payload=False, with_vectors=False)
            existing = {str(record.id) for record in records}
            pending = [chunk for chunk in chunks if chunk.id not in existing]
            self.stats.skipped += len(chunks) - len(pending)
            state = ArticleState(article.id, [chunk.id for chunk in chunks], len(pending))
            if not pending:
                self.complete_article(state)
                self.notify()
            for chunk in pending:
                yield PendingChunk(chunk, article.text_hash, state)

    def receive(self, future: Future[list[list[float]]]) -> int:
        """записываем завершённый пакет независимо от порядка ответов модели."""
        slot, pending = self.active.pop(future)
        vectors = future.result()
        save_chunks(self.store, self.name, [(item.chunk, item.text_hash) for item in pending],
                    vectors, self.vector_size)
        self.stats.saved += len(pending)
        self.batches += 1
        for item in pending:
            item.article.remaining -= 1
            if item.article.remaining == 0:
                self.complete_article(item.article)
        self.notify()
        return slot

    def ready(self, available: list[int], *, block: bool = False) -> None:
        """сначала проверяем готовые ответы и не запускаем новые пакеты после обнаруженной ошибки."""
        if not self.active:
            return
        done, _ = wait(self.active, timeout=0.25 if block else 0, return_when=FIRST_COMPLETED)
        for future in done:
            available.append(self.receive(future))

    def drain(self) -> None:
        """при остановке сохраняем успешные ответы уже отправленных пакетов без повторных запросов."""
        for future in list(self.active):
            if future.cancel():
                self.active.pop(future)
                continue
            try:
                self.receive(future)
            except Exception:
                # исходная ошибка будет передана вызывающему коду; остальные ответы всё равно забираем
                continue


def index_articles(
    articles: Iterable[PreparedArticle], tokenizer: TextTokenizer, config: ChunkingConfig,
    store: QdrantClient, name: str, providers: Sequence[EmbeddingProvider], batch_size: int = 8,
    vector_size: int = VECTOR_SIZE, *, on_progress: Callable[[IndexProgress], None] | None = None,
    on_stopping: Callable[[], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> IndexProgress:
    """объединяем статьи в пакеты, ограничиваем число запросов и сохраняем результат в одном потоке."""
    if batch_size < 1 or not providers or vector_size < 1:
        raise ValueError('нужны положительный размер пакета, размерность и хотя бы один клиент')
    if len({id(provider) for provider in providers}) != len(providers):
        raise ValueError('каждому работнику нужен отдельный клиент модели')
    coordinator = IndexCoordinator(store, name, vector_size, on_progress)
    available = list(range(len(providers)))
    executor = ThreadPoolExecutor(max_workers=len(providers), thread_name_prefix='habr-embeddings')

    def check_stop() -> None:
        """проверяем остановку между записями, чтобы не прервать сохранение готового пакета."""
        if stop_requested is not None and stop_requested():
            raise KeyboardInterrupt

    def submit(batch: list[PendingChunk]) -> None:
        """один клиент обслуживает не больше одного запроса; лишние задания в пул не накапливаются."""
        check_stop()
        coordinator.ready(available)
        while not available:
            check_stop()
            coordinator.ready(available, block=True)
        check_stop()
        slot = available.pop()
        future = executor.submit(providers[slot].embed, [item.chunk.embedding_text for item in batch])
        coordinator.active[future] = (slot, batch)

    pending = coordinator.pending_chunks(articles, tokenizer, config, stop_requested)
    try:
        batch: list[PendingChunk] = []
        for item in pending:
            check_stop()
            coordinator.ready(available)
            batch.append(item)
            if len(batch) == batch_size:
                submit(batch)
                batch = []
        if batch:
            submit(batch)
        while coordinator.active:
            check_stop()
            coordinator.ready(available, block=True)
    except BaseException:
        # прекращаем чтение корпуса и не бросаем ответы, за которые сервер уже мог списать оплату
        if on_stopping is not None:
            on_stopping()
        coordinator.drain()
        raise
    finally:
        pending.close()
        executor.shutdown(wait=True, cancel_futures=True)
    return coordinator.progress()
