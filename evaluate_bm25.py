"""оцениваем bm25 и сравниваем с готовым отчётом семантического поиска."""
from __future__ import annotations

import argparse
import csv
import os
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from bm25 import BM25_VERSION, BM25Config, BM25Index, corpus_hash, load_indexed_chunks
from chunking import ChunkingConfig
from embeddings import DEFAULT_MODEL, DEFAULT_URL
from evaluate import (
    DEFAULT_QUERIES, ROOT, QueryEvaluation, ReportContext, ReportEntry, load_queries,
    load_report, query_set_hash, score_ranking, summarize,
)
from index import collection_name
from search import validate_collection
from vector_store import connect_store, resolve_qdrant_config

REPORT_COLUMNS: tuple[str, ...] = (
    'started_at', 'method', 'bm25_version', 'k1', 'b', 'collection', 'collection_points',
    'corpus_sha256', 'query_set_sha256', 'top_k', 'query_id', 'category', 'query',
    'relevant_article_ids', 'retrieved_article_ids', 'first_relevant_rank', 'reciprocal_rank',
    'semantic_report', 'semantic_first_relevant_rank', 'semantic_reciprocal_rank',
)


def print_summary(label: str, results: list[QueryEvaluation], top_k: int) -> None:
    summary = summarize(results)
    print(f'{label}: Hit@{top_k} {summary.hit_rate:.2%} ({summary.hits}/{summary.count}); MRR@{top_k} {summary.mrr:.4f}')
    for category in sorted({result.case.category for result in results}):
        group = summarize([result for result in results if result.case.category == category])
        print(f'  {category}: запросов {group.count}; Hit@{top_k} {group.hit_rate:.2%}; MRR@{top_k} {group.mrr:.4f}')


class CliArgs(argparse.Namespace):
    queries: Path
    output: Path | None
    semantic_report: Path | None
    semantic_raw_query: bool
    top_k: int
    limit: int | None
    qdrant_url: str | None
    qdrant_path: Path | None
    max_tokens: int
    overlap_tokens: int
    k1: float
    b: float


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queries', type=Path, default=DEFAULT_QUERIES)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--semantic-report', type=Path, help='полный отчёт evaluate.py по тому же набору')
    parser.add_argument('--semantic-raw-query', action='store_true', help='если отчёт Gemma получен с --raw-query')
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--limit', type=int)
    location = parser.add_mutually_exclusive_group()
    location.add_argument('--qdrant-url')
    location.add_argument('--qdrant-path', type=Path)
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--overlap-tokens', type=int, default=128)
    parser.add_argument('--k1', type=float, default=1.2)
    parser.add_argument('--b', type=float, default=0.75)
    args: CliArgs = parser.parse_args(namespace=CliArgs())
    if args.top_k < 1 or (args.limit is not None and args.limit < 1):
        parser.error('top-k и limit должны быть положительными')
    try:
        cases = load_queries(args.queries)
        if args.limit is not None:
            cases = cases[:args.limit]
        config = BM25Config(args.k1, args.b)
        started = datetime.now(timezone.utc)
        output = args.output or ROOT / 'data' / 'evaluation' / f'bm25-{started:%Y%m%d-%H%M%S-%f}.csv'
        if output.exists():
            raise ValueError(f'отчёт уже существует: {output}; задайте новый --output')
        chunking = ChunkingConfig(args.max_tokens, args.overlap_tokens)
        name = collection_name(os.environ.get('MWS_EMBEDDINGS_URL', DEFAULT_URL),
                               os.environ.get('MWS_EMBEDDINGS_MODEL', DEFAULT_MODEL), chunking)
        connection = resolve_qdrant_config(args.qdrant_url, args.qdrant_path)
        if connection.path is not None and not connection.path.is_dir():
            raise ValueError('локальное хранилище не найдено; задайте --qdrant-url')
        with closing(connect_store(connection)) as store:
            validate_collection(store, name)
            chunks = load_indexed_chunks(store, name)
            point_count = len(chunks)
            present = {chunk.article_id for chunk in chunks}
            missing = {value for case in cases for value in case.relevant_ids} - present
            if missing:
                raise ValueError('в индексе отсутствуют ожидаемые статьи: ' + ', '.join(map(str, sorted(missing))))
            fingerprint = query_set_hash(cases)
            semantic: dict[str, ReportEntry] = {}
            if args.semantic_report is not None:
                semantic = load_report(args.semantic_report, cases, ReportContext(
                    name, point_count, fingerprint, args.semantic_raw_query, args.top_k,
                ))
                if len(semantic) != len(cases):
                    raise ValueError('семантический отчёт неполный; сначала завершите evaluate.py')
            index = BM25Index(chunks, config)
            results = [score_ranking(case, [item.article_id for item in index.search(case.query, args.top_k)], args.top_k)
                       for case in cases]
            # повторное чтение выявляет изменение текстов и метаданных во время оценки
            snapshot_hash = corpus_hash(chunks)
            if corpus_hash(load_indexed_chunks(store, name)) != snapshot_hash:
                raise ValueError('индекс изменился во время оценки; отчёт не записан')
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x', encoding='utf-8', newline='') as report:
            writer = csv.writer(report)
            writer.writerow(REPORT_COLUMNS)
            for result in results:
                original = semantic.get(result.case.query_id)
                original_rank = original.result.first_relevant_rank if original is not None else None
                writer.writerow([
                    started.isoformat(), 'bm25', BM25_VERSION, config.k1, config.b, name, point_count,
                    snapshot_hash, fingerprint, args.top_k, result.case.query_id, result.case.category,
                    result.case.query, ';'.join(map(str, result.case.relevant_ids)),
                    ';'.join(map(str, result.retrieved_ids)),
                    result.first_relevant_rank if result.first_relevant_rank is not None else '', result.reciprocal_rank,
                    str(args.semantic_report) if args.semantic_report is not None else '',
                    original_rank if original_rank is not None else '',
                    original.result.reciprocal_rank if original is not None else '',
                ])
        print(f'коллекция: {name}; фрагментов: {point_count}; статей: {len(present)}')
        print(f'отчёт: {output}')
        print_summary('BM25', results, args.top_k)
        if semantic:
            baseline = [semantic[case.query_id].result for case in cases]
            print_summary('Gemma (сохранённый отчёт)', baseline, args.top_k)
            improved = sum(item.reciprocal_rank > semantic[item.case.query_id].result.reciprocal_rank for item in results)
            worse = sum(item.reciprocal_rank < semantic[item.case.query_id].result.reciprocal_rank for item in results)
            print(f'по первому подходящему результату: bm25 лучше {improved}; хуже {worse}; одинаково {len(results) - improved - worse}')
        print('обращений к MWS: 0; индекс не изменялся')
        return 0
    except KeyboardInterrupt:
        print('\nоценка остановлена')
        return 130
    except UnexpectedResponse as exc:
        parser.exit(1, f'qdrant вернул http {exc.status_code}\n')
    except ResponseHandlingException:
        parser.exit(1, 'не удалось связаться с qdrant\n')
    except (ValueError, RuntimeError, OSError, csv.Error) as exc:
        parser.exit(1, f'оценка bm25 не завершена: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
