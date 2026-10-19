"""сравниваем пороги по сохранённым оценкам сходства без mws и подключения к qdrant."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from habr_search.evaluation.evaluate import (
    EvaluationSummary, QueryEvaluation, load_queries, score_ranking, summarize,
)
from habr_search.evaluation.evaluate_spelling import read_complete_report
from habr_search.retrieval.search import validate_min_score


@dataclass(frozen=True)
class ThresholdResult:
    """одна строка эксперимента содержит порог и метрики исходной разметки."""
    min_score: float | None
    summary: EvaluationSummary


def apply_threshold(result: QueryEvaluation, min_score: float | None, top_k: int) -> QueryEvaluation:
    """отсекаем слабые результаты, сохраняя их порядок и связь оценок с номерами статей."""
    validate_min_score(min_score)
    if result.scores is None or len(result.scores) != len(result.retrieved_ids):
        raise ValueError('в отчёте нет оценок сходства; нужен новый запуск evaluate без старого --resume')
    kept = [(article_id, score) for article_id, score in zip(result.retrieved_ids, result.scores)
            if min_score is None or score >= min_score]
    scored = score_ranking(result.case, [article_id for article_id, _ in kept], top_k)
    # прежнее время запроса не выдаём за измерение скорости нового правила
    return QueryEvaluation(result.case, scored.retrieved_ids, scored.first_relevant_rank,
                           scores=tuple(score for _, score in kept))


def compare_thresholds(results: list[QueryEvaluation], top_k: int) -> tuple[list[ThresholdResult], ThresholdResult]:
    """проверяем сетку с шагом 0.01 и сохраняем hit и mrr не ниже исходного уровня."""
    baseline = ThresholdResult(None, summarize([apply_threshold(result, None, top_k) for result in results]))
    if not baseline.summary.count or not baseline.summary.negative_count:
        raise ValueError('для подбора нужны и положительные запросы, и категория no_answer')
    rows = [baseline]
    best = baseline
    for step in range(-100, 101):
        threshold = step / 100
        summary = summarize([apply_threshold(result, threshold, top_k) for result in results])
        candidate = ThresholdResult(threshold, summary)
        rows.append(candidate)
        preserves_answers = (summary.hit_rate + 1e-12 >= baseline.summary.hit_rate
                             and summary.mrr + 1e-12 >= baseline.summary.mrr)
        # при одинаковом результате оставляем более мягкий порог; без выигрыша сохраняем исходный поиск
        if preserves_answers and summary.abstentions > best.summary.abstentions:
            best = candidate
    return rows, best


def calibrate(queries: Path, report: Path, output: Path) -> ThresholdResult:
    """создаём отдельную таблицу сравнения, не меняя индекс, отчёт и настройки приложения."""
    if output.exists():
        raise ValueError('таблица уже существует; укажите другой --output')
    context, results = read_complete_report(report, load_queries(queries))
    if context.min_score is not None:
        raise ValueError('нужен исходный отчёт без min-score: отсечённые результаты восстановить нельзя')
    rows, best = compare_thresholds(results, context.top_k)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['collection', 'collection_points', 'index_revision', 'raw_query', 'top_k', 'query_set_sha256', 'min_score',
                         'positive_queries', 'hit_at_k', 'mrr_at_k', 'no_answer_queries', 'empty_results'])
        for row in rows:
            writer.writerow([context.collection, context.collection_points, context.index_revision,
                             context.raw_query, context.top_k, context.query_set_sha256,
                             '' if row.min_score is None else row.min_score, row.summary.count,
                             row.summary.hit_rate, row.summary.mrr, row.summary.negative_count, row.summary.abstentions])
    return best


class CliArgs(argparse.Namespace):
    """команда использует только сохранённые файлы и не принимает ключи модели."""
    queries: Path
    report: Path
    output: Path


def main() -> int:
    """показываем предварительный вариант порога для проверки, не включая его в рабочем поиске."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queries', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args: CliArgs = parser.parse_args(namespace=CliArgs())
    try:
        best = calibrate(args.queries, args.report, args.output)
        if best.min_score is None:
            print('порог не выбран: на этой сетке нет улучшения отказов без потери hit и mrr')
        else:
            print(f'предварительный min-score: {best.min_score:.2f}; '
                  f'пустая выдача {best.summary.abstentions}/{best.summary.negative_count}')
        print(f'таблица сравнения: {args.output}')
        print('результат зависит от разметки; настройки приложения не менялись; mws и qdrant не вызывались')
        return 0
    except (ValueError, OSError, csv.Error) as exc:
        parser.exit(1, f'порог не подобран: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
