"""проверяем отсутствие пересечения тем между наборами и целостность исходной разметки."""
from __future__ import annotations

import csv
import json
import unittest

from habr_search.evaluation.evaluate import load_queries
from habr_search.paths import PROJECT_ROOT


# проверки работают и на другом компьютере без локального корпуса и qdrant
class BenchmarkTests(unittest.TestCase):
    # целевые статьи и формулировки отложены целиком, искусственные отрицательные случаи помечены явно
    def test_splits_have_disjoint_articles_and_consistent_provenance(self) -> None:
        folder = PROJECT_ROOT / 'evaluation'
        dev = load_queries(folder / 'benchmark_dev.csv')
        holdout = load_queries(folder / 'benchmark_holdout.csv')
        dev_ids = {value for case in dev for value in case.relevant_ids}
        holdout_ids = {value for case in holdout for value in case.relevant_ids}
        self.assertFalse(dev_ids & holdout_ids)
        self.assertFalse({case.query.casefold() for case in dev} & {case.query.casefold() for case in holdout})
        manifest = json.loads((folder / 'benchmark_manifest.json').read_text())
        self.assertEqual(len(dev), manifest['dev_queries'])
        self.assertEqual(len(holdout), manifest['holdout_queries'])
        self.assertEqual(len(dev_ids | holdout_ids), manifest['target_articles'])
        self.assertFalse(manifest['retrieval_results_reviewed'])
        with (folder / 'benchmark_articles.csv').open() as stream:
            articles = list(csv.DictReader(stream))
        self.assertEqual({int(row['article_id']) for row in articles}, dev_ids | holdout_ids)
        for row in articles:
            self.assertEqual(len(row['text_sha256']), 64)
            self.assertTrue(row['evidence_note'])
        for cases in (dev, holdout):
            self.assertTrue(any(case.category == 'no_answer' for case in cases))
            self.assertTrue(all(bool(case.relevant_ids) == (case.category != 'no_answer') for case in cases))


if __name__ == '__main__':
    unittest.main()
