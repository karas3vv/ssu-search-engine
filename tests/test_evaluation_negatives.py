"""проверяем отрицательную разметку, версии отчёта и остановку до лишних вызовов модели."""
from __future__ import annotations

import csv
import io
import tempfile
import unittest
from contextlib import closing, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from qdrant_client import QdrantClient

from habr_search.evaluation.evaluate import (
    REPORT_COLUMNS, EvaluationQuery, ReportContext, ReportEntry, evaluate_queries,
    load_queries, load_report, print_summary, query_set_hash, report_row, score_ranking, summarize,
)
from habr_search.ingestion.index import ensure_collection
from tests.test_evaluate import FakeEmbeddings, add_point, write_queries


# отделяем правильный отказ от промаха по положительному запросу
class NegativeEvaluationTests(unittest.TestCase):
    # пустая разметка требует явной категории, иначе потерянные id выглядели бы правильным отказом
    def test_empty_ids_are_only_allowed_for_explicit_no_answer(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'queries.csv'
            write_queries(path, [['missing', 'no_answer', 'несуществующая технология', '']])
            self.assertEqual(load_queries(path)[0].relevant_ids, ())
            for category, ids in [('exact', ''), ('no_answer', '1'), ('no_answer', ';')]:
                write_queries(path, [['missing', category, 'запрос', ids]])
                with self.assertRaises(ValueError):
                    load_queries(path)

    # отрицательные запросы не попадают в знаменатель hit и mrr
    def test_negative_results_have_separate_abstention_metric(self) -> None:
        positive = EvaluationQuery('positive', 'exact', 'известная задача', (1,))
        missing = EvaluationQuery('missing', 'no_answer', 'нет подходящей статьи', ())
        results = [score_ranking(positive, [1], 5), score_ranking(missing, [], 5),
                   score_ranking(missing, [2, 3], 5)]
        summary = summarize(results)
        self.assertEqual((summary.count, summary.hit_rate, summary.mrr), (1, 1.0, 1.0))
        self.assertEqual((summary.negative_count, summary.abstentions), (2, 1))
        with redirect_stdout(io.StringIO()) as output:
            print_summary(results[1:], 5)
        self.assertIn('пустая выдача 1/2', output.getvalue())
        self.assertNotIn('Hit@', output.getvalue())

    # продолжение отклоняет иной индекс даже при прежнем числе точек и проверяет время запроса
    def test_resume_checks_revision_and_finite_nonnegative_latency(self) -> None:
        case = EvaluationQuery('one', 'no_answer', 'нет ответа', ())
        context = ReportContext('test', 1, query_set_hash([case]), False, 5, 'revision-1')
        row = report_row(ReportEntry('2026-10-19T10:00:00+00:00', score_ranking(case, [], 5)), context)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'report.csv'
            for field, value in [('index_revision', 'revision-2'), ('elapsed_ms', 'nan'), ('elapsed_ms', '-1')]:
                bad = row.copy()
                bad[REPORT_COLUMNS.index(field)] = value
                with path.open('w', newline='') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(REPORT_COLUMNS)
                    writer.writerow(bad)
                with self.assertRaises(ValueError):
                    load_report(path, [case], context)

    # при смене версии после первого результата второй платный запрос не отправляется
    def test_revision_change_stops_before_second_embedding(self) -> None:
        with closing(QdrantClient(':memory:')) as store:
            ensure_collection(store, 'test', 3)
            add_point(store, 1, 1, [1., 0., 0.])
            case = EvaluationQuery('one', 'exact', 'запрос', (1,))
            backend = FakeEmbeddings()
            with patch('habr_search.evaluation.evaluate.index_revision', side_effect=['v1', 'v1', 'v1', 'v2']):
                results = evaluate_queries([(case, 'один'), (case, 'два')], store, 'test', backend, vector_size=3)
                first = next(results)
                self.assertIsNotNone(first.elapsed_ms)
                with self.assertRaisesRegex(ValueError, 'версия индекса'):
                    next(results)
            self.assertEqual(backend.calls, [['один']])


if __name__ == '__main__':
    unittest.main()
