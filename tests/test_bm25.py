"""проверяем формулу, группировку статей и чтение фрагментов без обращений к модели."""
from __future__ import annotations

import csv
import io
import math
import tempfile
import unittest
from contextlib import closing, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from qdrant_client import QdrantClient, models

from bm25 import BM25Config, BM25Index, IndexedChunk, corpus_hash, load_indexed_chunks, tokenize
from chunking import ChunkingConfig
from embeddings import DEFAULT_MODEL, DEFAULT_URL
from evaluate import (
    QUERY_COLUMNS, REPORT_COLUMNS, EvaluationQuery, ReportContext, ReportEntry,
    query_set_hash, report_row, score_ranking,
)
from evaluate_bm25 import main
from index import VECTOR_SIZE, collection_name, ensure_collection


def chunk(point_id: str, article_id: int, text: str, title: str = '') -> IndexedChunk:
    return IndexedChunk(point_id, article_id, title, f'https://example.com/{article_id}', text)


class BM25Tests(unittest.TestCase):
    def test_formula_matches_hand_calculation_and_query_repeats_do_not_boost(self) -> None:
        index = BM25Index([chunk('a', 1, 'excel excel'), chunk('b', 2, 'excel'), chunk('c', 3, 'word')])
        found = index.search('EXCEL')
        idf = math.log(1 + (3 - 2 + 0.5) / (2 + 0.5))
        expected = idf * 2 * (1.2 + 1) / (2 + 1.2 * (1 - 0.75 + 0.75 * 2 / (4 / 3)))
        self.assertEqual([item.article_id for item in found], [1, 2])
        self.assertAlmostEqual(found[0].score, expected, places=12)
        self.assertEqual(index.search('excel excel'), found)
        self.assertEqual(index.search('неизвестно'), [])
        self.assertEqual(index.search('?!'), [])

    def test_normalization_handles_russian_unicode_and_technical_names(self) -> None:
        self.assertEqual(tokenize('ЁЖ ёж ＥＸＣＥＬ Python C++ C# execute_excel4macro'),
                         ['еж', 'еж', 'excel', 'python', 'c++', 'c#', 'execute', 'excel4macro'])

    def test_grouping_returns_best_chunk_of_each_article_with_stable_ties(self) -> None:
        chunks = [chunk('b', 1, 'excel'), chunk('a', 1, 'excel'), chunk('c', 2, 'excel')]
        index = BM25Index(chunks)
        self.assertEqual([item.article_id for item in index.search('excel', 2)], [1, 2])
        self.assertEqual(index.search('excel'), BM25Index(list(reversed(chunks))).search('excel'))
        titled = BM25Index([chunk('x', 3, 'обычный текст', title='ExecuteExcel4Macro')])
        self.assertEqual(titled.search('ExecuteExcel4Macro')[0].article_id, 3)

    def test_length_normalization_prefers_shorter_chunk_for_same_frequency(self) -> None:
        chunks = [chunk('a', 1, 'excel ' + 'word ' * 40), chunk('b', 2, 'excel word')]
        self.assertEqual(BM25Index(chunks).search('excel')[0].article_id, 2)
        scores = BM25Index(chunks, BM25Config(b=0)).search('excel')
        self.assertAlmostEqual(scores[0].score, scores[1].score)

    def test_invalid_settings_and_duplicate_chunks_are_rejected(self) -> None:
        for k1, b in [(-1.0, 0.75), (float('nan'), 0.75), (1.2, 1.1), (1.2, float('inf'))]:
            with self.subTest(k1=k1, b=b), self.assertRaises(ValueError):
                BM25Config(k1, b)
        with self.assertRaises(ValueError):
            BM25Index([])
        with self.assertRaises(ValueError):
            BM25Index([chunk('a', 1, 'word'), chunk('a', 1, 'word')])
        with self.assertRaises(ValueError):
            BM25Index([chunk('a', 1, 'word')]).search(' ')
        with self.assertRaises(ValueError):
            BM25Index([chunk('a', 1, 'word')]).search('word', 0)

    def test_scroll_reads_all_pages_without_mutating_points(self) -> None:
        with closing(QdrantClient(location=':memory:')) as store:
            ensure_collection(store, 'test', 3)
            store.upsert('test', points=[models.PointStruct(
                id=i, vector=[1.0, 0.0, 0.0], payload={
                    'article_id': i, 'title': 'тест', 'url': f'https://example.com/{i}', 'text': f'текст {i}',
                },
            ) for i in range(1, 6)])
            chunks = load_indexed_chunks(store, 'test', batch_size=2)
            self.assertEqual({item.article_id for item in chunks}, {1, 2, 3, 4, 5})
            self.assertEqual(store.count('test', exact=True).count, 5)
            self.assertEqual(corpus_hash(chunks), corpus_hash(list(reversed(chunks))))
            changed = [chunk('1', 1, 'изменённый текст')] + chunks[1:]
            self.assertNotEqual(corpus_hash(chunks), corpus_hash(changed))
            store.set_payload('test', payload={'article_id': True}, points=[1])
            with self.assertRaises(ValueError):
                load_indexed_chunks(store, 'test')

    def test_cli_compares_saved_semantic_report_without_model_or_key(self) -> None:
        for complete in [True, False]:
            with self.subTest(complete=complete), tempfile.TemporaryDirectory() as folder:
                source = Path(folder) / 'queries.csv'
                baseline = Path(folder) / 'semantic.csv'
                output = Path(folder) / 'bm25.csv'
                cases = [EvaluationQuery('excel', 'exact', 'excel', (1,)),
                         EvaluationQuery('python', 'alias', 'python', (2,))]
                name = collection_name(DEFAULT_URL, DEFAULT_MODEL, ChunkingConfig())
                context = ReportContext(name, 2, query_set_hash(cases), False, 5)
                with source.open('w', encoding='utf-8', newline='') as target:
                    writer = csv.writer(target)
                    writer.writerow(QUERY_COLUMNS)
                    for case in cases:
                        writer.writerow([case.query_id, case.category, case.query, ';'.join(map(str, case.relevant_ids))])
                with baseline.open('w', encoding='utf-8', newline='') as target:
                    writer = csv.writer(target)
                    writer.writerow(REPORT_COLUMNS)
                    for case in cases if complete else cases[:1]:
                        ids = [1] if case.query_id == 'excel' else []
                        writer.writerow(report_row(ReportEntry('2026-10-19T09:00:00+00:00', score_ranking(case, ids, 5)), context))
                before = baseline.read_bytes()
                with closing(QdrantClient(location=':memory:')) as store:
                    ensure_collection(store, name)
                    store.upsert(name, points=[models.PointStruct(
                        id=i, vector=[1.0] + [0.0] * (VECTOR_SIZE - 1), payload={
                            'article_id': i, 'title': word, 'url': f'https://example.com/{i}', 'text': word,
                        },
                    ) for i, word in [(1, 'excel'), (2, 'python')]])
                    argv = ['evaluate_bm25.py', '--queries', str(source), '--output', str(output),
                            '--semantic-report', str(baseline), '--qdrant-url', 'http://localhost:6333']
                    with (patch('sys.argv', argv), patch.dict('os.environ', {}, clear=True),
                          patch('evaluate_bm25.connect_store', return_value=store),
                          patch('evaluate.EmbeddingsClient', side_effect=AssertionError('mws не нужен')),
                          patch('evaluate.read_api_key', side_effect=AssertionError('ключ не нужен')),
                          patch('evaluate.load_tokenizer', side_effect=AssertionError('модель не нужна')),
                          redirect_stdout(io.StringIO()) as stdout, redirect_stderr(io.StringIO()) as stderr):
                        if complete:
                            self.assertEqual(main(), 0)
                        else:
                            with self.assertRaises(SystemExit) as error:
                                main()
                            self.assertEqual(error.exception.code, 1)
                            self.assertIn('неполный', stderr.getvalue())
                self.assertEqual(baseline.read_bytes(), before)
                if complete:
                    with output.open(encoding='utf-8', newline='') as report:
                        rows = list(csv.DictReader(report))
                    self.assertEqual(len(rows), 2)
                    self.assertEqual(rows[1]['semantic_reciprocal_rank'], '0.0')
                    self.assertEqual(rows[1]['first_relevant_rank'], '1')
                    self.assertEqual(rows[0]['method'], 'bm25')
                    self.assertIn('bm25 лучше 1; хуже 0; одинаково 1', stdout.getvalue())
                    self.assertIn('обращений к MWS: 0', stdout.getvalue())
                else:
                    self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
