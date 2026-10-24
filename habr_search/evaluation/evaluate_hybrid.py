"""сравниваем три поиска на одном эмбеддинге запроса и сохраняем кандидатов в csv."""
from __future__ import annotations

import argparse
import csv
import math
import os
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, closing
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from habr_search.embeddings import DEFAULT_URL, EmbeddingAPIError, EmbeddingConfig, EmbeddingsClient, read_api_key
from habr_search.evaluation.evaluate import (
    EvaluationQuery, QueryEvaluation, load_queries, prepare_queries, print_summary, query_set_hash,
    score_ranking, validate_expected_articles,
)
from habr_search.ingestion.chunking import ChunkingConfig, load_tokenizer
from habr_search.ingestion.index import EmbeddingProvider, collection_name
from habr_search.model_profiles import BGE_M3, deployment_name
from habr_search.paths import DATA_DIR, PROJECT_ROOT
from habr_search.retrieval.hybrid import HybridConfig, HybridRetriever, fuse_rankings
from habr_search.retrieval.tantivy_index import DEFAULT_INDEX, TantivySearch, literal_query
from habr_search.vector_store import connect_store, resolve_qdrant_config

REPORT_COLUMNS: tuple[str, ...] = (
    'format_version', 'collection', 'qdrant_revision', 'tantivy_revision', 'collection_points',
    'query_set_sha256', 'top_k', 'candidate_limit', 'rrf_k', 'started_at', 'query_id',
    'category', 'query', 'relevant_article_ids', 'dense_article_ids', 'lexical_article_ids',
    'hybrid_article_ids', 'elapsed_ms', 'dense_elapsed_ms', 'lexical_elapsed_ms',
)


@dataclass(frozen=True)
class HybridContext:
    collection: str
    qdrant_revision: str
    tantivy_revision: str
    points: int
    query_hash: str
    top_k: int
    config: HybridConfig

    def cells(self) -> list[str]:
        """эти настройки должны точно совпасть при продолжении отчёта."""
        return ['1', self.collection, self.qdrant_revision, self.tantivy_revision, str(self.points),
                self.query_hash, str(self.top_k), str(self.config.candidate_limit), str(self.config.rrf_k)]


@dataclass(frozen=True)
class HybridEvaluation:
    case: EvaluationQuery
    dense_ids: tuple[int, ...]
    lexical_ids: tuple[int, ...]
    hybrid_ids: tuple[int, ...]
    elapsed_ms: float
    dense_ms: float
    lexical_ms: float
    started_at: str

    def row(self, context: HybridContext) -> list[str]:
        """кандидаты сохраняются глубже top-k, чтобы не платить за их повторное получение."""
        return context.cells() + [self.started_at, self.case.query_id, self.case.category, self.case.query,
                                  ';'.join(map(str, self.case.relevant_ids)), ';'.join(map(str, self.dense_ids)),
                                  ';'.join(map(str, self.lexical_ids)), ';'.join(map(str, self.hybrid_ids)),
                                  str(self.elapsed_ms), str(self.dense_ms), str(self.lexical_ms)]

    def metric(self, mode: str, top_k: int) -> QueryEvaluation:
        """порядок и измеренное время каждого режима передаём общим метрикам проекта."""
        if mode == 'bge-m3':
            ids, elapsed = self.dense_ids, self.dense_ms
        elif mode == 'tantivy':
            ids, elapsed = self.lexical_ids, self.lexical_ms
        elif mode == 'hybrid':
            ids, elapsed = self.hybrid_ids, self.elapsed_ms
        else:
            raise ValueError('неизвестный режим оценки')
        return replace(score_ranking(self.case, ids, top_k), elapsed_ms=elapsed)


def load_saved(path: Path, cases: Sequence[EvaluationQuery], context: HybridContext) -> dict[str, HybridEvaluation]:
    """отклоняем смешанные версии, повторы и повреждённые строки до вызова модели."""
    by_id = {case.query_id: case for case in cases}
    saved: dict[str, HybridEvaluation] = {}
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream, strict=True)
        if tuple(next(reader, list[str]())) != REPORT_COLUMNS:
            raise ValueError('resume требует отчёт evaluate_hybrid с кандидатами обоих поисков')
        for row in reader:
            if len(row) != len(REPORT_COLUMNS) or row[:9] != context.cells():
                raise ValueError('настройки или версии индексов в resume не совпали')
            case = by_id.get(row[10])
            if case is None or case.query_id in saved or row[11:14] != [case.category, case.query, ';'.join(map(str, case.relevant_ids))]:
                raise ValueError('запрос resume изменён, неизвестен или повторяется')
            datetime.fromisoformat(row[9])
            lists = [tuple(map(int, cell.split(';'))) if cell else () for cell in row[14:17]]
            dense, lexical, hybrid = lists
            # функция объединения также проверяет уникальность, знаки id и глубину списков
            expected = tuple(item.article_id for item in fuse_rankings(dense, lexical, context.top_k, context.config))
            if hybrid != expected:
                raise ValueError('выдача resume не соответствует сохранённым кандидатам')
            elapsed, dense_ms, lexical_ms = map(float, row[17:20])
            if any(not math.isfinite(value) or value < 0 for value in (elapsed, dense_ms, lexical_ms)):
                raise ValueError('в resume повреждено время запроса')
            saved[case.query_id] = HybridEvaluation(case, dense, lexical, hybrid, elapsed, dense_ms, lexical_ms, row[9])
    return saved


def evaluate_cases(prepared: Sequence[tuple[EvaluationQuery, str]], retriever: HybridRetriever,
                   embeddings: EmbeddingProvider, top_k: int) -> Iterator[HybridEvaluation]:
    """один платный вызов на запрос; оба списка и объединение получаем из этого вектора."""
    for case, text in prepared:
        retriever.check_context()
        start = perf_counter()
        try:
            vector = embeddings.embed([text])[0]
        except EmbeddingAPIError as exc:
            raise EmbeddingAPIError(f'{case.query_id}: {exc}') from None
        embedding_ms = (perf_counter() - start) * 1000
        run = retriever.search(case.query, vector, top_k)
        yield HybridEvaluation(case, tuple(item.article_id for item in run.dense),
                               tuple(item.article_id for item in run.lexical),
                               tuple(item.article_id for item in run.results), (perf_counter() - start) * 1000,
                               embedding_ms + run.dense_ms, run.lexical_ms, datetime.now(timezone.utc).isoformat())


class CliArgs(argparse.Namespace):
    queries: Path
    output: Path | None
    resume: Path | None
    qdrant_url: str | None
    qdrant_path: Path | None
    tantivy_path: Path
    tokenizer_file: Path | None
    candidate_limit: int
    rrf_k: float
    top_k: int
    timeout: float
    max_tokens: int
    overlap_tokens: int
    check_only: bool


def main() -> int:
    """проверяем готовность индексов, создаём новый отчёт и продолжаем сохранённые запросы."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queries', type=Path, default=PROJECT_ROOT / 'evaluation' / 'benchmark_dev.csv')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', type=Path)
    location = parser.add_mutually_exclusive_group()
    location.add_argument('--qdrant-url')
    location.add_argument('--qdrant-path', type=Path)
    parser.add_argument('--tantivy-path', type=Path, default=DEFAULT_INDEX)
    parser.add_argument('--tokenizer-file', type=Path)
    parser.add_argument('--candidate-limit', type=int, default=20)
    parser.add_argument('--rrf-k', type=float, default=60.0)
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--overlap-tokens', type=int, default=128)
    parser.add_argument('--timeout', type=float, default=300.0)
    parser.add_argument('--check-only', action='store_true', help='проверить источники без mws и создания отчёта')
    args = parser.parse_args(namespace=CliArgs())
    try:
        config = HybridConfig(args.candidate_limit, args.rrf_k)
        if not 1 <= args.top_k <= min(20, config.candidate_limit):
            raise ValueError('top-k должен быть от 1 до 20 и не больше candidate-limit')
        if not math.isfinite(args.timeout) or args.timeout <= 0:
            raise ValueError('timeout должен быть положительным конечным числом')
        cases = load_queries(args.queries)
        for case in cases:
            literal_query(case.query)
        url, deployment = os.environ.get('MWS_EMBEDDINGS_URL', DEFAULT_URL), deployment_name(BGE_M3)
        name = collection_name(url, deployment, ChunkingConfig(args.max_tokens, args.overlap_tokens), BGE_M3)
        with ExitStack() as resources:
            store = connect_store(resolve_qdrant_config(args.qdrant_url, args.qdrant_path))
            resources.callback(store.close)
            retriever = HybridRetriever(store, name, TantivySearch(args.tantivy_path), config)
            validate_expected_articles(store, name, cases)
            context = HybridContext(name, retriever.revision, retriever.lexical.info.revision, retriever.points,
                                    query_set_hash(cases), args.top_k, config)
            saved = load_saved(args.resume, cases, context) if args.resume else {}
            pending = [case for case in cases if case.query_id not in saved]
            prepared = prepare_queries(pending, load_tokenizer(args.tokenizer_file, BGE_M3), profile=BGE_M3) if pending else []
            print(f'коллекция: {name}; статей: {retriever.lexical.info.articles}; кандидатов каждого поиска: {config.candidate_limit}')
            print(f'из отчёта: {len(saved)}; осталось запросов: {len(pending)}')
            if args.check_only:
                print('индексы и запросы проверены; обращений к mws: 0; отчёт не создавался')
                return 0
            output = args.output or DATA_DIR / 'evaluation' / datetime.now(timezone.utc).strftime('hybrid-%Y%m%d-%H%M%S-%f.csv')
            output.parent.mkdir(parents=True, exist_ok=True)
            # существующий файл не заменяется, даже если это файл resume
            stream = resources.enter_context(output.open('x', encoding='utf-8', newline=''))
            writer = csv.writer(stream)
            writer.writerow(REPORT_COLUMNS)
            for case in cases:
                if case.query_id in saved:
                    writer.writerow(saved[case.query_id].row(context))
            stream.flush()
            print(f'отчёт: {output}', flush=True)
            if pending:
                provider = resources.enter_context(closing(EmbeddingsClient(EmbeddingConfig(read_api_key(), url, deployment, args.timeout))))
                for result in evaluate_cases(prepared, retriever, provider, args.top_k):
                    writer.writerow(result.row(context))
                    stream.flush()
                    saved[result.case.query_id] = result
                    rank = result.metric('hybrid', args.top_k).first_relevant_rank
                    print(f'{len(saved)}/{len(cases)} {result.case.query_id}: место {rank or "нет"}', flush=True)
            retriever.check_context()
        for mode in ('bge-m3', 'tantivy', 'hybrid'):
            results = [saved[case.query_id].metric(mode, args.top_k) for case in cases]
            print_summary(results, args.top_k, mode)
            for category in sorted({case.category for case in cases}):
                print_summary([result for result in results if result.case.category == category], args.top_k, f'{mode}/{category}')
        print(f'завершено запросов: {len(saved)}; новых запросов к mws: {len(pending)}')
        return 0
    except KeyboardInterrupt:
        print('\nостановлено; передайте сохранённый отчёт через --resume, чтобы продолжить')
        return 130
    except (UnexpectedResponse, ResponseHandlingException):
        parser.exit(1, 'не удалось прочитать qdrant; проверьте контейнер и адрес\n')
    except (ValueError, OSError, RuntimeError, csv.Error) as exc:
        parser.exit(1, f'оценка не завершена: {exc}\nзавершённые строки можно продолжить через --resume\n')


if __name__ == '__main__':
    raise SystemExit(main())
