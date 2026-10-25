"""проверяем подбор порога по сохранённым оценкам без запросов к модели и индексу."""
from __future__ import annotations

import csv
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from habr_search.evaluation.calibrate import apply_threshold, calibrate, compare_thresholds
from habr_search.evaluation.evaluate import (
    REPORT_COLUMNS, EvaluationQuery, QueryEvaluation, ReportContext, ReportEntry,
    query_set_hash, report_row, score_ranking,
)
from tests.test_evaluate import write_queries


def results(positive_score: float, negative_score: float) -> list[QueryEvaluation]:
    """создаём известный ответ и запрос без ответа с управляемым сходством."""
    positive = EvaluationQuery('known', 'exact', 'нужная статья', (1,))
    negative = EvaluationQuery('unknown', 'no_answer', 'вымышленная технология', ())
    return [replace(score_ranking(positive, [1], 5), scores=(positive_score,), elapsed_ms=100),
            replace(score_ranking(negative, [2], 5), scores=(negative_score,), elapsed_ms=200)]


class CalibrationTests(unittest.TestCase):
    """сохраняем полезные ответы, отделяем отказы и проверяем совместимость файлов."""

    def test_threshold_rejects_negative_without_losing_known_answer(self) -> None:
        """выбираем первый мягкий порог, который улучшает число отказов."""
        rows, best = compare_thresholds(results(0.8, 0.3), 5)
        self.assertEqual(len(rows), 202)
        self.assertEqual(best.min_score, 0.31)
        self.assertEqual(best.summary.abstentions, 1)
        self.assertEqual(best.summary.hit_rate, 1)
        self.assertEqual(best.summary.mrr, 1)
        filtered = apply_threshold(results(0.8, 0.3)[1], 0.31, 5)
        self.assertEqual(filtered.retrieved_ids, ())
        self.assertEqual(filtered.scores, ())
        self.assertIsNone(filtered.elapsed_ms)

    def test_no_threshold_is_recommended_when_it_loses_answers(self) -> None:
        """сильный ложный ответ нельзя скрыть ценой потери известного полезного ответа."""
        _, best = compare_thresholds(results(0.3, 0.8), 5)
        self.assertIsNone(best.min_score)
        self.assertEqual(best.summary.abstentions, 0)

    def test_missing_scores_or_negative_queries_prevent_calibration(self) -> None:
        """без чисел или примеров отказа нельзя обосновать выбранный порог."""
        missing = [replace(item, scores=None) for item in results(0.8, 0.3)]
        with self.assertRaisesRegex(ValueError, 'нет оценок'):
            compare_thresholds(missing, 5)
        with self.assertRaisesRegex(ValueError, 'no_answer'):
            compare_thresholds(results(0.8, 0.3)[:1], 5)

    def test_csv_export_preserves_inputs_and_rejects_prefiltered_or_existing_output(self) -> None:
        """подбор читает отчёт целиком, пишет отдельную таблицу и не перезаписывает файлы."""
        items = results(0.8, 0.3)
        cases = [item.case for item in items]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            queries, report, output = root / 'queries.csv', root / 'report.csv', root / 'thresholds.csv'
            write_queries(queries, [[case.query_id, case.category, case.query,
                                     ';'.join(map(str, case.relevant_ids))] for case in cases])
            context = ReportContext('test', 2, query_set_hash(cases), False, 5)
            with report.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(REPORT_COLUMNS)
                for item in items:
                    writer.writerow(report_row(ReportEntry('2026-10-20T00:00:00+00:00', item), context))
            original = report.read_bytes()
            self.assertEqual(calibrate(queries, report, output).min_score, 0.31)
            self.assertEqual(report.read_bytes(), original)
            with output.open(encoding='utf-8', newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 202)
            self.assertEqual(rows[0]['min_score'], '')
            self.assertEqual(rows[0]['collection_points'], '2')
            with self.assertRaisesRegex(ValueError, 'существует'):
                calibrate(queries, report, output)
            with report.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(REPORT_COLUMNS)
                for item in items:
                    writer.writerow(report_row(ReportEntry('2026-10-20T00:00:00+00:00', item),
                                               replace(context, min_score=0.1)))
            with self.assertRaisesRegex(ValueError, 'отсечённые'):
                calibrate(queries, report, root / 'other.csv')

    def test_fine_grid_finds_narrow_candidate_without_rounding_to_wrong_value(self) -> None:
        """маленький разрыв оценок виден на мелкой сетке, но не означает надёжность правила."""
        items = results(0.5080922, 0.5073422)
        _, coarse = compare_thresholds(items, 5)
        rows, fine = compare_thresholds(items, 5, threshold_step=0.001)
        self.assertIsNone(coarse.min_score)
        self.assertEqual(len(rows), 2002)
        self.assertEqual(fine.min_score, 0.508)
        self.assertEqual(fine.summary.hits, 1)
        self.assertEqual(fine.summary.abstentions, 1)
        self.assertEqual(fine.positive_empty, 0)
        for step in [0, -0.1, 0.00001, 2, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                compare_thresholds(items, 5, threshold_step=step)

    def test_step_and_details_show_false_abstentions_without_changing_source(self) -> None:
        """csv показывает потери полезных ответов, а отдельный разбор сохраняет связь с запросами."""
        items = results(0.5080922, 0.5073422)
        cases = [item.case for item in items]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            queries, report, output, details = [root / name for name in ('queries.csv', 'report.csv', 'grid.csv', 'details.csv')]
            write_queries(queries, [[case.query_id, case.category, case.query,
                                    ';'.join(map(str, case.relevant_ids))] for case in cases])
            context = ReportContext('bge-m3-test', 2, query_set_hash(cases), False, 5)
            with report.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(REPORT_COLUMNS)
                for item in items:
                    writer.writerow(report_row(ReportEntry('2026-10-25T00:00:00+00:00', item), context))
            original = report.read_bytes()
            chosen = calibrate(queries, report, output, threshold_step=0.001, details=details)
            self.assertEqual(chosen.min_score, 0.508)
            self.assertEqual(original, report.read_bytes())
            with output.open() as stream:
                grid = list(csv.DictReader(stream))
            row = next(row for row in grid if row['min_score'] == '0.51')
            self.assertEqual(row['positive_empty_results'], '1')
            self.assertEqual(row['lost_hits'], '1')
            with details.open() as stream:
                checked = list(csv.DictReader(stream))
            self.assertEqual(checked[0]['after_rank'], '1')
            self.assertEqual(checked[1]['after_empty'], 'True')
            with self.assertRaises(ValueError):
                calibrate(queries, report, root / 'new.csv', details=details)
            self.assertFalse((root / 'new.csv').exists())


if __name__ == '__main__':
    unittest.main()
