"""оцениваем локальный поиск без запросов к mws и сохраняем отдельный отчёт bm25."""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from habr_search.evaluation.evaluate import QueryEvaluation, load_queries, print_summary, query_set_hash, score_ranking
from habr_search.paths import DATA_DIR, PROJECT_ROOT
from habr_search.retrieval.tantivy_index import DEFAULT_INDEX, TantivySearch


class CliArgs(argparse.Namespace):
    index_path: Path
    queries: Path
    output: Path | None
    top_k: int


def main() -> int:
    """проверяем исходные запросы, не применяя подсказки и не меняя индекс."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index-path', type=Path, default=DEFAULT_INDEX)
    parser.add_argument('--queries', type=Path, default=PROJECT_ROOT / 'evaluation' / 'benchmark_dev.csv')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--top-k', type=int, choices=range(1, 21), default=5)
    args = parser.parse_args(namespace=CliArgs())
    try:
        cases = load_queries(args.queries)
        backend = TantivySearch(args.index_path)
        output = args.output or DATA_DIR / 'evaluation' / datetime.now(timezone.utc).strftime('tantivy-%Y%m%d-%H%M%S-%f.csv')
        output.parent.mkdir(parents=True, exist_ok=True)
        results: list[QueryEvaluation] = []
        # отдельный формат исключает смешивание bm25 с косинусными оценками модели
        with output.open('x', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['engine', 'index_revision', 'query_set_sha256', 'articles', 'top_k',
                             'query_id', 'category', 'query', 'relevant_article_ids',
                             'retrieved_article_ids', 'first_relevant_rank', 'elapsed_ms', 'bm25_scores'])
            for case in cases:
                start = perf_counter()
                found = backend.search(case.query, args.top_k)
                elapsed = (perf_counter() - start) * 1000
                result = replace(score_ranking(case, [item.article_id for item in found], args.top_k), elapsed_ms=elapsed)
                results.append(result)
                writer.writerow(['tantivy', backend.info.revision, query_set_hash(cases), backend.info.articles,
                                 args.top_k, case.query_id, case.category, case.query,
                                 ';'.join(map(str, case.relevant_ids)), ';'.join(map(str, result.retrieved_ids)),
                                 result.first_relevant_rank or '', round(elapsed, 3),
                                 ';'.join(str(item.score) for item in found)])
                stream.flush()
        print(f'статей: {backend.info.articles}; отчёт: {output}')
        print_summary(results, args.top_k)
        return 0
    except (ValueError, OSError) as exc:
        parser.exit(1, f'оценка не завершена: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
