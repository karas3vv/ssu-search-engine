"""объединяем статьи из bge-m3 и tantivy по местам, не складывая разные оценки."""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter

from qdrant_client import QdrantClient
from qdrant_client.conversions import common_types as qdrant_types

from habr_search.index_version import index_revision
from habr_search.model_profiles import BGE_M3
from habr_search.retrieval.search import SearchResult, search_articles, validate_collection
from habr_search.retrieval.tantivy_index import LexicalResult, TantivySearch, literal_query


@dataclass(frozen=True)
class HybridConfig:
    candidate_limit: int = 20  # каждый источник отдаёт одинаковое число кандидатов
    rrf_k: float = 60.0  # сглаживаем разницу между соседними местами

    def __post_init__(self) -> None:
        if not 1 <= self.candidate_limit <= 100:
            raise ValueError('candidate-limit должен быть от 1 до 100')
        if not math.isfinite(self.rrf_k) or self.rrf_k <= 0:
            raise ValueError('rrf-k должен быть положительным конечным числом')


@dataclass(frozen=True)
class FusedRank:
    article_id: int
    score: float
    dense_rank: int | None
    lexical_rank: int | None


def fuse_rankings(dense: Sequence[int], lexical: Sequence[int], top_k: int,
                  config: HybridConfig = HybridConfig()) -> list[FusedRank]:
    """rrf учитывает место статьи в каждом списке; одна статья появляется один раз."""
    if not 1 <= top_k <= 20 or top_k > config.candidate_limit:
        raise ValueError('top-k должен быть от 1 до 20 и не больше candidate-limit')
    for ids in (dense, lexical):
        if len(ids) > config.candidate_limit or len(ids) != len(set(ids)) or any(type(i) is not int or i < 1 for i in ids):
            raise ValueError('кандидаты должны иметь разные положительные id в пределах candidate-limit')
    dense_ranks = {article_id: rank for rank, article_id in enumerate(dense, 1)}
    lexical_ranks = {article_id: rank for rank, article_id in enumerate(lexical, 1)}
    found: list[FusedRank] = []
    for article_id in dense_ranks.keys() | lexical_ranks.keys():
        dense_rank, lexical_rank = dense_ranks.get(article_id), lexical_ranks.get(article_id)
        score = sum(1.0 / (config.rrf_k + rank) for rank in (dense_rank, lexical_rank) if rank is not None)
        found.append(FusedRank(article_id, score, dense_rank, lexical_rank))
    # при равных оценках порядок воспроизводится независимо от порядка set
    absent = config.candidate_limit + 1
    found.sort(key=lambda item: (-item.score, min(item.dense_rank or absent, item.lexical_rank or absent),
                                item.dense_rank or absent, item.article_id))
    return found[:top_k]


@dataclass(frozen=True)
class HybridResult:
    article_id: int
    title: str
    url: str
    snippet: str
    score: float


@dataclass(frozen=True)
class HybridRun:
    dense: tuple[SearchResult, ...]
    lexical: tuple[LexicalResult, ...]
    results: tuple[HybridResult, ...]
    dense_ms: float
    lexical_ms: float


class HybridRetriever:
    """используем готовый вектор запроса; этот класс никогда не вызывает модель."""
    def __init__(self, store: QdrantClient, name: str, lexical: TantivySearch,
                 config: HybridConfig = HybridConfig()) -> None:
        validate_collection(store, name, BGE_M3.vector_size)
        self.store: QdrantClient = store
        self.name: str = name
        self.lexical: TantivySearch = lexical
        self.config: HybridConfig = config
        self.points: int = store.count(name, exact=True).count
        self.revision: str = index_revision(store, name)
        # до платного поиска проверяем, что оба источника содержат одни и те же статьи
        ids: set[int] = set()
        offset: qdrant_types.PointId | None = None
        seen: set[str] = set()
        while True:
            records, next_offset = store.scroll(name, offset=offset, limit=512,
                                                with_payload=['article_id'], with_vectors=False)
            for record in records:
                article_id: object = (record.payload or {}).get('article_id')
                if type(article_id) is not int or article_id < 1 or str(record.id) in seen:
                    raise ValueError('повреждены метаданные статей в qdrant')
                seen.add(str(record.id))
                ids.add(article_id)
            if next_offset is None:
                break
            if next_offset == offset or not records:
                raise ValueError('не удалось прочитать все статьи qdrant')
            offset = next_offset
        self.check_context()
        if len(seen) != self.points or frozenset(ids) != lexical.article_ids():
            raise ValueError('наборы статей qdrant и tantivy различаются; обновите оба индекса из одного корпуса')

    def check_context(self) -> None:
        """изменившийся векторный индекс требует повторного запуска сервиса."""
        if (self.store.count(self.name, exact=True).count != self.points
                or index_revision(self.store, self.name) != self.revision):
            raise ValueError('индекс qdrant изменился; проверьте актуальность tantivy и перезапустите поиск')

    def search(self, query: str, vector: list[float], top_k: int = 5) -> HybridRun:
        """ищем кандидатов и выбираем отрывок tantivy, если он доступен."""
        literal_query(query)  # ограничения локального поиска проверяем до выполнения запросов
        if not 1 <= top_k <= min(20, self.config.candidate_limit):
            raise ValueError('top-k должен быть от 1 до 20 и не больше candidate-limit')
        self.check_context()
        start = perf_counter()
        dense = search_articles(self.store, self.name, vector, self.config.candidate_limit, BGE_M3.vector_size)
        dense_ms = (perf_counter() - start) * 1000
        start = perf_counter()
        lexical = self.lexical.candidates(query, self.config.candidate_limit)
        lexical_ms = (perf_counter() - start) * 1000
        ranks = fuse_rankings([item.article_id for item in dense], [item.article_id for item in lexical], top_k, self.config)
        dense_by_id = {item.article_id: item for item in dense}
        lexical_by_id = {item.article_id: item for item in lexical}
        results: list[HybridResult] = []
        for rank in ranks:
            local = lexical_by_id.get(rank.article_id)
            if local is not None:
                results.append(HybridResult(rank.article_id, local.title, local.url, local.snippet, rank.score))
            else:
                semantic = dense_by_id[rank.article_id]
                text = ' '.join(semantic.text.split())
                results.append(HybridResult(rank.article_id, semantic.title, semantic.url,
                                            text[:500] + ('…' if len(text) > 500 else ''), rank.score))
        self.check_context()
        return HybridRun(tuple(dense), tuple(lexical), tuple(results), dense_ms, lexical_ms)
