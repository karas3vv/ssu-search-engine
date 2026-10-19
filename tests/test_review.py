"""проверяем выгрузку ручной разметки без сети, готового корпуса и платного поиска."""
from __future__ import annotations

import csv
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from habr_search.evaluation.evaluate import (
    REPORT_COLUMNS, EvaluationQuery, ReportContext, ReportEntry, query_set_hash, report_row, score_ranking,
)
from habr_search.evaluation.review import export_review
from tests.test_evaluate import write_queries


# таблица сохраняет порядок выдачи и не превращает отсутствие метки в отрицательную оценку
class ReviewTests(unittest.TestCase):
    # найденные и пропущенные эталоны включаются в таблицу, оценки остаются пустыми
    def test_export_keeps_candidates_expected_articles_and_empty_results(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, queries, report, output = [root / name for name in (
                'corpus.sqlite3', 'queries.csv', 'report.csv', 'review.csv',
            )]
            with closing(sqlite3.connect(source)) as db:
                db.execute('CREATE TABLE articles(id INTEGER PRIMARY KEY, payload TEXT)')
                for article_id in (1, 2):
                    db.execute('INSERT INTO articles VALUES (?, ?)', (article_id, json.dumps({
                        'id': article_id, 'url': f'https://habr.com/ru/articles/{article_id}/',
                        'title': '=опасная формула' if article_id == 2 else 'ожидаемая статья',
                        'text': f'полный текст {article_id}',
                    })))
                db.commit()
            before = source.read_bytes()
            cases = [EvaluationQuery('one', 'exact', 'нужная статья', (1,)),
                     EvaluationQuery('none', 'no_answer', 'нет ответа', ())]
            write_queries(queries, [[case.query_id, case.category, case.query,
                                     ';'.join(map(str, case.relevant_ids))] for case in cases])
            context = ReportContext('test', 2, query_set_hash(cases), False, 5, 'revision')
            with report.open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(REPORT_COLUMNS)
                for case, ids in zip(cases, [[2], []]):
                    writer.writerow(report_row(ReportEntry('2026-10-19T10:00:00+00:00',
                                                           score_ranking(case, ids, 5)), context))
            self.assertEqual(export_review(queries, report, source, output), 3)
            with output.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row['article_id'] for row in rows], ['2', '1', ''])
            self.assertEqual([row['rank'] for row in rows], ['1', '', ''])
            self.assertEqual([row['seed_relevant'] for row in rows], ['', 'yes', ''])
            self.assertTrue(rows[0]['title'].startswith("'="))
            self.assertTrue(all(row['relevance'] == '' and row['comment'] == '' for row in rows))
            self.assertEqual(source.read_bytes(), before)
            saved = output.read_bytes()
            with self.assertRaises(ValueError):
                export_review(queries, report, source, output)
            self.assertEqual(output.read_bytes(), saved)


if __name__ == '__main__':
    unittest.main()
