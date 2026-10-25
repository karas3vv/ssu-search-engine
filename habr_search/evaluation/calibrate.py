"""проверяем пороги по сохранённым оценкам и показываем потери без вызовов модели."""
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

from habr_search.evaluation.evaluate import (
    EvaluationSummary, QueryEvaluation, load_queries, score_ranking, summarize,
)
from habr_search.evaluation.evaluate_spelling import read_complete_report
from habr_search.retrieval.search import validate_min_score


@dataclass(frozen=True)
class ThresholdResult:
    min_score: float | None
    summary: EvaluationSummary
    positive_empty: int = 0  # пустая выдача на вопросах, для которых ответ в корпусе есть


def apply_threshold(result: QueryEvaluation, min_score: float | None, top_k: int) -> QueryEvaluation:
    """отсекаем слабые результаты, не меняя их порядок и связь с оценками."""
    validate_min_score(min_score)
    if result.scores is None or len(result.scores) != len(result.retrieved_ids):
        raise ValueError('в отчёте нет оценок сходства; нужен новый запуск evaluate без старого --resume')
    kept = [(article_id, score) for article_id, score in zip(result.retrieved_ids, result.scores)
            if min_score is None or score >= min_score]
    scored = score_ranking(result.case, [article_id for article_id, _ in kept], top_k)
    # прежнее время запроса не выдаём за измерение нового правила
    return QueryEvaluation(result.case, scored.retrieved_ids, scored.first_relevant_rank,
                           scores=tuple(score for _, score in kept))


def compare_thresholds(results: list[QueryEvaluation], top_k: int, *, threshold_step: float = 0.01
                       ) -> tuple[list[ThresholdResult], ThresholdResult]:
    """выбираем предварительный порог только без потери исходных hit и mrr."""
    if not math.isfinite(threshold_step) or not 0.0001 <= threshold_step <= 1:
        raise ValueError('step должен быть конечным числом от 0.0001 до 1')

    def measure(threshold: float | None) -> ThresholdResult:
        filtered = [apply_threshold(result, threshold, top_k) for result in results]
        return ThresholdResult(threshold, summarize(filtered),
                               sum(not item.retrieved_ids for item in filtered if item.case.category != 'no_answer'))

    baseline = measure(None)
    if not baseline.summary.count or not baseline.summary.negative_count:
        raise ValueError('для подбора нужны и положительные запросы, и категория no_answer')
    rows = [baseline]
    best = baseline
    for step in range(math.ceil(2 / threshold_step) + 1):
        threshold = min(1.0, round(-1 + step * threshold_step, 12))
        candidate = measure(threshold)
        rows.append(candidate)
        summary = candidate.summary
        preserves_answers = (summary.hit_rate + 1e-12 >= baseline.summary.hit_rate
                             and summary.mrr + 1e-12 >= baseline.summary.mrr)
        # более мягкий порог остаётся первым; без выигрыша сохраняем исходный поиск
        if preserves_answers and summary.abstentions > best.summary.abstentions:
            best = candidate
    return rows, best


def calibrate(queries: Path, report: Path, output: Path, *, threshold_step: float = 0.01,
              details: Path | None = None) -> ThresholdResult:
    """сохраняем таблицу порогов и необязательный разбор запросов, не меняя настройки."""
    if output.exists() or (details is not None and details.exists()):
        raise ValueError('таблица уже существует; укажите другой --output или --details-output')
    if details is not None and output.resolve() == details.resolve():
        raise ValueError('таблица порогов и подробный разбор должны иметь разные пути')
    context, results = read_complete_report(report, load_queries(queries))
    if context.min_score is not None:
        raise ValueError('нужен исходный отчёт без min-score: отсечённые результаты восстановить нельзя')
    rows, best = compare_thresholds(results, context.top_k, threshold_step=threshold_step)
    baseline = rows[0].summary
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['collection', 'collection_points', 'index_revision', 'raw_query', 'top_k', 'query_set_sha256', 'min_score',
                         'positive_queries', 'hit_at_k', 'mrr_at_k', 'no_answer_queries', 'empty_results',
                         'positive_empty_results', 'lost_hits', 'step'])
        for row in rows:
            writer.writerow([context.collection, context.collection_points, context.index_revision,
                             context.raw_query, context.top_k, context.query_set_sha256,
                             '' if row.min_score is None else row.min_score, row.summary.count,
                             row.summary.hit_rate, row.summary.mrr, row.summary.negative_count, row.summary.abstentions,
                             row.positive_empty, baseline.hits - row.summary.hits, threshold_step])
    if details is not None:
        details.parent.mkdir(parents=True, exist_ok=True)
        with details.open('x', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['collection', 'index_revision', 'query_id', 'category', 'query', 'min_score',
                             'before_rank', 'after_rank', 'before_article_ids', 'after_article_ids',
                             'before_empty', 'after_empty'])
            for original in results:
                filtered = apply_threshold(original, best.min_score, context.top_k)
                writer.writerow([context.collection, context.index_revision, original.case.query_id,
                                 original.case.category, original.case.query, best.min_score,
                                 original.first_relevant_rank or '', filtered.first_relevant_rank or '',
                                 ';'.join(map(str, original.retrieved_ids)), ';'.join(map(str, filtered.retrieved_ids)),
                                 not original.retrieved_ids, not filtered.retrieved_ids])
    return best


class CliArgs(argparse.Namespace):
    queries: Path
    report: Path
    output: Path
    details_output: Path | None
    step: float


def main() -> int:
    """предложение по dev не включается автоматически и требует отдельной проверки."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queries', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--details-output', type=Path)
    parser.add_argument('--step', type=float, default=0.01)
    args = parser.parse_args(namespace=CliArgs())
    try:
        best = calibrate(args.queries, args.report, args.output, threshold_step=args.step, details=args.details_output)
        if best.min_score is None:
            print('порог не выбран: на этой сетке нет улучшения отказов без потери hit и mrr')
        else:
            print(f'предварительный min-score: {best.min_score:g}; '
                  f'пустая выдача {best.summary.abstentions}/{best.summary.negative_count}; '
                  f'пустых положительных запросов: {best.positive_empty}')
        print(f'таблица сравнения: {args.output}')
        if args.details_output is not None:
            print(f'разбор запросов: {args.details_output}')
        print('это подбор на dev; настройки приложения не менялись; mws и qdrant не вызывались')
        return 0
    except (ValueError, OSError, csv.Error) as exc:
        parser.exit(1, f'порог не подобран: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
