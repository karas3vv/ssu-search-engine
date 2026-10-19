"""проверяем независимую разметку исправлений и сравнение готовых выдач."""
from __future__ import annotations

import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from evaluate import REPORT_COLUMNS, EvaluationQuery, ReportContext, ReportEntry, query_set_hash, report_row, score_ranking
from evaluate_spelling import (
    CASE_COLUMNS, SpellingCase, compare_reports, evaluate_local, load_cases, write_queries,
)
from test_spelling import dictionary


def report(path: Path, cases: list[EvaluationQuery], ranks: list[list[int]], points: int = 20) -> None:
    context = ReportContext('test', points, query_set_hash(cases), False, 5)
    with path.open('w', encoding='utf-8', newline='') as target:
        writer = csv.writer(target)
        writer.writerow(REPORT_COLUMNS)
        for case, ids in zip(cases, ranks):
            writer.writerow(report_row(ReportEntry('2026-10-19T00:00:00+00:00', score_ranking(case, ids, 5)), context))


class SpellingEvaluationTests(unittest.TestCase):
    def test_local_metrics_and_exports_keep_incomplete_corrections_visible(self) -> None:
        cases = [SpellingCase(EvaluationQuery('one', 'typo', 'pyhton', (1,)), 'python'),
                 SpellingCase(EvaluationQuery('two', 'typo', 'афтаматизация', (2,)), 'автоматизация'),
                 SpellingCase(EvaluationQuery('three', 'clean', 'docker', (3,)), 'docker')]
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'results'
            self.assertEqual(evaluate_local(dictionary(), cases, output, 'cases-hash'), (1, 2, 0, 1))
            with (output / 'local.csv').open(newline='', encoding='utf-8') as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(rows[1]['corrected_query'], 'афтаматизация')
            self.assertEqual(rows[1]['exact_match'], 'False')
            self.assertEqual(rows[0]['cases_sha256'], 'cases-hash')
            before = (output / 'local.csv').read_bytes()
            with self.assertRaises(FileExistsError):
                evaluate_local(dictionary(), cases, output, 'cases-hash')
            self.assertEqual((output / 'local.csv').read_bytes(), before)

    def test_cases_reject_bad_labels_duplicates_and_empty_groups(self) -> None:
        valid = ['one', 'typo', 'pyhton', 'python', '1']
        clean = ['two', 'clean', 'docker', 'docker', '2']
        variants = [[], [valid], [valid, valid, clean], [valid, ['two', 'clean', 'docker', 'python', '2']],
                    [valid, ['two', 'clean', 'docker', 'docker', '0']]]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'cases.csv'
            for rows in variants:
                with path.open('w', encoding='utf-8', newline='') as target:
                    writer = csv.writer(target)
                    writer.writerow(CASE_COLUMNS)
                    writer.writerows(rows)
                with self.assertRaises(ValueError):
                    load_cases(path)

    def test_comparison_rejects_partial_changed_index_and_changed_labels(self) -> None:
        original = [EvaluationQuery('one', 'typo', 'pyhton', (1,)), EvaluationQuery('two', 'clean', 'docker', (2,))]
        corrected = [EvaluationQuery('one', 'typo', 'python', (1,)), original[1]]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            a, b, ar, br = [root / name for name in ['a.csv', 'b.csv', 'ar.csv', 'br.csv']]
            write_queries(a, original)
            write_queries(b, corrected)
            report(ar, original, [[3, 1], [2]])
            report(br, corrected, [[1], [2]])
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                compare_reports(a, b, ar, br)
            self.assertIn('лучше: 1; хуже: 0; одинаково: 1', stdout.getvalue())
            report(br, corrected, [[1], [2]], points=21)
            with self.assertRaises(ValueError):
                compare_reports(a, b, ar, br)
            report(br, corrected, [[1]])
            with self.assertRaises(ValueError):
                compare_reports(a, b, ar, br)


if __name__ == '__main__':
    unittest.main()
