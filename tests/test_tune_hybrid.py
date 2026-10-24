"""проверяем веса и повторный расчёт кандидатов без сети и изменения исходного отчёта."""
from __future__ import annotations

import csv
import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from habr_search.evaluation.evaluate import EvaluationQuery, query_set_hash
from habr_search.evaluation.evaluate_hybrid import (
    LEGACY_REPORT_COLUMNS, REPORT_COLUMNS, HybridContext, HybridEvaluation, load_saved, read_saved_report,
)
from habr_search.evaluation.tune_hybrid import changed_ranks, main, replay
from habr_search.retrieval.hybrid import HybridConfig, fuse_rankings


class WeightTests(unittest.TestCase):
    def test_weight_changes_order_and_common_scaling_keeps_it(self) -> None:
        equal = fuse_rankings([1], [2], 2)
        weighted = fuse_rankings([1], [2], 2, HybridConfig(20, 60, 1, 2))
        scaled = fuse_rankings([1], [2], 2, HybridConfig(20, 60, 2, 4))
        self.assertEqual([item.article_id for item in equal], [1, 2])
        self.assertEqual([item.article_id for item in weighted], [2, 1])
        self.assertEqual([item.article_id for item in scaled], [2, 1])
        self.assertAlmostEqual(scaled[0].score, 2 * weighted[0].score)
        for weight in [0, -1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                HybridConfig(dense_weight=weight)
            with self.assertRaises(ValueError):
                HybridConfig(lexical_weight=weight)


class ReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.queries = self.root / 'queries.csv'
        self.queries.write_text('query_id,category,query,relevant_article_ids\n'
                                'one,exact,первый,1\ntwo,paraphrase,второй,2\n')
        self.cases = [EvaluationQuery('one', 'exact', 'первый', (1,)),
                      EvaluationQuery('two', 'paraphrase', 'второй', (2,))]
        self.context = HybridContext('m3', 'dense-v1', 'lexical-v1', 3, query_set_hash(self.cases), 2, HybridConfig())
        self.records: list[HybridEvaluation] = []
        for case, dense, lexical in [(self.cases[0], (3, 1), (1,)), (self.cases[1], (2,), (3, 2))]:
            found = fuse_rankings(dense, lexical, 2)
            self.records.append(HybridEvaluation(case, dense, lexical, tuple(item.article_id for item in found),
                                                 150.0, 110.0, 40.0, '2026-10-24T12:00:00+00:00'))

    def write_report(self, path: Path, legacy: bool = False, records: list[HybridEvaluation] | None = None) -> None:
        columns = LEGACY_REPORT_COLUMNS if legacy else REPORT_COLUMNS
        with path.open('x', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            for record in self.records if records is None else records:
                values = dict(zip(REPORT_COLUMNS, record.row(self.context)))
                if legacy:
                    values['format_version'] = '1'
                writer.writerow([values[key] for key in columns])

    def test_v1_report_remains_readable_and_weighted_resume_is_rejected(self) -> None:
        path = self.root / 'legacy.csv'
        self.write_report(path, legacy=True)
        context, records = read_saved_report(path, self.cases)
        self.assertEqual(context, self.context)
        self.assertEqual(records['one'].dense_ids, (3, 1))
        weighted = HybridContext('m3', 'dense-v1', 'lexical-v1', 3, query_set_hash(self.cases), 2,
                                 HybridConfig(dense_weight=2))
        with self.assertRaises(ValueError):
            load_saved(path, self.cases, weighted)

    def test_v2_report_tracks_weights_and_detects_tampering(self) -> None:
        path = self.root / 'current.csv'
        self.context = HybridContext('m3', 'dense-v1', 'lexical-v1', 3, query_set_hash(self.cases), 2,
                                     HybridConfig(20, 10, 1.5, 1))
        self.records = [HybridEvaluation(record.case, record.dense_ids, record.lexical_ids,
                                        tuple(item.article_id for item in fuse_rankings(record.dense_ids, record.lexical_ids,
                                                                                      2, self.context.config)),
                                        150.0, 110.0, 40.0, record.started_at) for record in self.records]
        self.write_report(path)
        context, records = read_saved_report(path, self.cases)
        self.assertEqual(context.config.dense_weight, 1.5)
        self.assertEqual(len(records), 2)
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        rows[0]['dense_weight'] = '2.0'
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=REPORT_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaises(ValueError):
            read_saved_report(path, self.cases)

    def test_offline_cli_never_connects_and_preserves_source(self) -> None:
        report = self.root / 'source.csv'
        self.write_report(report, legacy=True)
        original = report.read_bytes()
        output = self.root / 'weights.csv'
        argv = ['tune', '--report', str(report), '--queries', str(self.queries), '--output', str(output),
                '--dense-weights', '1', '2', '--rrf-values', '2', '60']
        with patch('sys.argv', argv), redirect_stdout(io.StringIO()), \
                patch('habr_search.evaluation.evaluate_hybrid.connect_store', side_effect=AssertionError('qdrant не нужен')), \
                patch('habr_search.evaluation.evaluate_hybrid.TantivySearch', side_effect=AssertionError('tantivy не нужен')), \
                patch('habr_search.evaluation.evaluate_hybrid.read_api_key', side_effect=AssertionError('ключ не нужен')), \
                patch('habr_search.evaluation.evaluate_hybrid.EmbeddingsClient', side_effect=AssertionError('модель не нужна')):
            self.assertEqual(main(), 0)
        self.assertEqual(report.read_bytes(), original)
        with output.open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 4)
        self.assertNotIn('elapsed_ms', rows[0])
        self.assertTrue(output.with_name('weights-queries.csv').is_file())
        with patch('sys.argv', argv), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main()
        self.assertEqual(report.read_bytes(), original)

    def test_partial_report_cannot_be_used_as_a_complete_benchmark(self) -> None:
        report = self.root / 'partial.csv'
        self.write_report(report, records=self.records[:1])
        output = self.root / 'weights.csv'
        with patch('sys.argv', ['tune', '--report', str(report), '--queries', str(self.queries), '--output', str(output)]), \
                redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main()
        self.assertFalse(output.exists())

    def test_replay_does_not_reuse_old_latency(self) -> None:
        baseline = replay(self.records, self.context.config, 2)
        trial = replay(self.records, HybridConfig(20, 2, 2, 1), 2)
        self.assertTrue(all(result.elapsed_ms is None for result in trial.results))
        self.assertEqual(changed_ranks(baseline.results, baseline.results), (0, 0, 0, 0))
        with self.assertRaises(ValueError):
            changed_ranks(baseline.results, tuple(reversed(trial.results)))


if __name__ == '__main__':
    unittest.main()
