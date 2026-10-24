"""сравниваем веса гибрида по сохранённым кандидатам без mws и открытия индексов."""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from habr_search.evaluation.evaluate import (
    EvaluationSummary, QueryEvaluation, load_queries, print_summary, score_ranking, summarize,
)
from habr_search.evaluation.evaluate_hybrid import HybridEvaluation, read_saved_report
from habr_search.paths import DATA_DIR, PROJECT_ROOT
from habr_search.retrieval.hybrid import HybridConfig, fuse_rankings


@dataclass(frozen=True)
class FusionTrial:
    config: HybridConfig
    results: tuple[QueryEvaluation, ...]

    @property
    def summary(self) -> EvaluationSummary:
        return summarize(self.results)


def replay(records: Sequence[HybridEvaluation], config: HybridConfig, top_k: int) -> FusionTrial:
    """не переносим старое время в новый эксперимент: заново измеряется только качество."""
    results: list[QueryEvaluation] = []
    for record in records:
        found = fuse_rankings(record.dense_ids, record.lexical_ids, top_k, config)
        results.append(score_ranking(record.case, [item.article_id for item in found], top_k))
    return FusionTrial(config, tuple(results))


def changed_ranks(before: Sequence[QueryEvaluation], after: Sequence[QueryEvaluation]) -> tuple[int, int, int, int]:
    """отдельно считаем новые попадания, потери и перестановки ожидаемых ответов."""
    if len(before) != len(after) or any(left.case != right.case for left, right in zip(before, after)):
        raise ValueError('для сравнения нужны одинаковые запросы в одинаковом порядке')
    gained = lost = improved = worsened = 0
    for left, right in zip(before, after):
        if left.case.category == 'no_answer':
            continue
        gained += left.first_relevant_rank is None and right.first_relevant_rank is not None
        lost += left.first_relevant_rank is not None and right.first_relevant_rank is None
        improved += right.reciprocal_rank > left.reciprocal_rank
        worsened += right.reciprocal_rank < left.reciprocal_rank
    return gained, lost, improved, worsened


class CliArgs(argparse.Namespace):
    report: Path
    queries: Path
    output: Path | None
    dense_weights: list[float]
    rrf_values: list[float]


def main() -> int:
    """сохраняем все варианты, а лучшую настройку оставляем кандидатом для holdout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--queries', type=Path, default=PROJECT_ROOT / 'evaluation' / 'benchmark_dev.csv')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--dense-weights', nargs='+', type=float, default=[1.0, 1.5, 2.0, 3.0])
    parser.add_argument('--rrf-values', nargs='+', type=float, default=[2.0, 10.0, 60.0])
    args = parser.parse_args(namespace=CliArgs())
    try:
        if any(not math.isfinite(value) or value <= 0 for value in (*args.dense_weights, *args.rrf_values)):
            raise ValueError('веса и rrf-k должны быть положительными конечными числами')
        weights, constants = list(dict.fromkeys(args.dense_weights)), list(dict.fromkeys(args.rrf_values))
        if len(weights) * len(constants) > 100:
            raise ValueError('сравнивайте не больше 100 вариантов за один запуск')
        cases = load_queries(args.queries)
        context, saved = read_saved_report(args.report, cases)
        if len(saved) != len(cases):
            raise ValueError('для сравнения весов нужен завершённый отчёт по всему выбранному набору')
        records = [saved[case.query_id] for case in cases]
        baseline = replay(records, context.config, context.top_k)
        trials = [replay(records, HybridConfig(context.config.candidate_limit, k, weight, 1.0), context.top_k)
                  for weight in weights for k in constants]
        trials.sort(key=lambda trial: (-trial.summary.hits, -trial.summary.mrr, trial.config.rrf_k, trial.config.dense_weight))
        best = trials[0]
        output = args.output or DATA_DIR / 'evaluation' / datetime.now(timezone.utc).strftime('fusion-grid-%Y%m%d-%H%M%S-%f.csv')
        details = output.with_name(output.stem + '-queries.csv')
        if output.exists() or details.exists() or output.resolve() == args.report.resolve():
            raise ValueError('выходной отчёт уже существует; укажите другой --output')
        source_hash = hashlib.sha256(args.report.read_bytes()).hexdigest()
        coverage = sum(bool(set(record.case.relevant_ids) & (set(record.dense_ids) | set(record.lexical_ids)))
                       for record in records if record.case.category != 'no_answer')
        metadata = [source_hash, context.collection, context.qdrant_revision, context.tantivy_revision,
                    context.query_hash, context.config.candidate_limit, context.top_k]
        output.parent.mkdir(parents=True, exist_ok=True)
        with ExitStack() as resources:
            grid = resources.enter_context(output.open('x', encoding='utf-8', newline=''))
            per_query = resources.enter_context(details.open('x', encoding='utf-8', newline=''))
            writer = csv.writer(grid)
            writer.writerow(['source_report_sha256', 'collection', 'qdrant_revision', 'tantivy_revision',
                             'query_set_sha256', 'candidate_limit', 'top_k', 'rrf_k', 'dense_weight', 'lexical_weight',
                             'positive_queries', 'hits', 'hit_rate', 'mrr', 'union_hits',
                             'gained_hits', 'lost_hits', 'improved_rank', 'worsened_rank'])
            for trial in trials:
                summary = trial.summary
                writer.writerow([*metadata, trial.config.rrf_k, trial.config.dense_weight, trial.config.lexical_weight,
                                 summary.count, summary.hits, summary.hit_rate, summary.mrr, coverage,
                                 *changed_ranks(baseline.results, trial.results)])
                print(f'bge-m3:tantivy={trial.config.dense_weight:g}:1; rrf-k={trial.config.rrf_k:g}; '
                      f'Hit@{context.top_k}: {summary.hit_rate:.2%} ({summary.hits}/{summary.count}); MRR: {summary.mrr:.4f}')
            writer = csv.writer(per_query)
            writer.writerow(['query_id', 'category', 'query', 'relevant_article_ids', 'baseline_rank', 'candidate_rank',
                             'baseline_article_ids', 'candidate_article_ids', 'rrf_k', 'dense_weight', 'lexical_weight'])
            for left, right in zip(baseline.results, best.results):
                writer.writerow([left.case.query_id, left.case.category, left.case.query, ';'.join(map(str, left.case.relevant_ids)),
                                 left.first_relevant_rank or '', right.first_relevant_rank or '',
                                 ';'.join(map(str, left.retrieved_ids)), ';'.join(map(str, right.retrieved_ids)),
                                 best.config.rrf_k, best.config.dense_weight, best.config.lexical_weight])
        print_summary(baseline.results, context.top_k, 'исходный гибрид')
        print_summary(best.results, context.top_k, 'лучший из проверенных на этом наборе')
        for category in sorted({case.category for case in cases}):
            print_summary([result for result in best.results if result.case.category == category], context.top_k, category)
        gained, lost, improved, worsened = changed_ranks(baseline.results, best.results)
        print(f'новых попаданий: {gained}; потеряно: {lost}; место лучше: {improved}; хуже: {worsened}')
        print(f'ожидаемый ответ среди кандидатов: {coverage}/{best.summary.count}')
        print(f'отчёт вариантов: {output}\nизменения запросов: {details}')
        print('обращений к mws: 0; индексы не открывались; время нового поиска не измерялось; настройки сервера не менялись')
        print('это подбор на текущем наборе; перед применением нужна отдельная проверка holdout')
        return 0
    except (ValueError, OSError, RuntimeError, csv.Error) as exc:
        parser.exit(1, f'сравнение не завершено: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
