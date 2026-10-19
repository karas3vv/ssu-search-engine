"""проверяем метрики и запуск оценки на синтетическом индексе без api mws."""
from __future__ import annotations

import csv
import io
import tempfile
import unittest
from contextlib import closing, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from qdrant_client import QdrantClient, models

from chunking import ChunkingConfig
from embeddings import DEFAULT_MODEL, DEFAULT_URL, EmbeddingAPIError
from evaluate import (
    DEFAULT_QUERIES, QUERY_COLUMNS, EvaluationQuery, evaluate_queries, load_queries,
    main, prepare_queries, query_set_hash, score_ranking, summarize, validate_expected_articles,
)
from index import VECTOR_SIZE, collection_name, ensure_collection
from search import build_query


class CharacterTokenizer:
    def count(self, text: str) -> int:
        return len(text) + 2

    def offsets(self, text: str) -> list[tuple[int, int]]:
        return [(i, i + 1) for i in range(len(text))]


class FakeEmbeddings:
    def __init__(self, vector_size: int = 3, fail_on_call: int | None = None) -> None:
        self.calls: list[list[str]] = []
        self.vector_size: int = vector_size
        self.fail_on_call: int | None = fail_on_call
        self.closed: bool = False

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        if len(self.calls) == self.fail_on_call:
            raise EmbeddingAPIError('синтетический сбой')
        return [[1.0] + [0.0] * (self.vector_size - 1) for _ in texts]

    def close(self) -> None:
        self.closed = True


def write_queries(path: Path, rows: list[list[str]]) -> None:
    with path.open('w', encoding='utf-8', newline='') as target:
        writer = csv.writer(target)
        writer.writerow(QUERY_COLUMNS)
        writer.writerows(rows)


def add_point(store: QdrantClient, point_id: int, article_id: int, vector: list[float]) -> None:
    store.upsert('test', points=[models.PointStruct(id=point_id, vector=vector, payload={
        'article_id': article_id, 'title': f'статья {article_id}',
        'url': f'https://habr.com/ru/articles/{article_id}/', 'text': 'тестовый фрагмент',
    })])


class EvaluationTests(unittest.TestCase):
    def test_metrics_handle_misses_cutoff_and_multiple_relevant_articles(self) -> None:
        case = EvaluationQuery('one', 'exact', 'запрос', (10, 20))
        first = score_ranking(case, [20, 30, 10], 3)
        second = score_ranking(case, [30, 10, 20], 3)
        missed = score_ranking(case, [30, 40, 10], 2)
        empty = score_ranking(case, [], 3)
        self.assertEqual(first.first_relevant_rank, 1)
        self.assertEqual(second.first_relevant_rank, 2)
        self.assertIsNone(missed.first_relevant_rank)
        self.assertEqual(missed.retrieved_ids, (30, 40))
        self.assertEqual(empty.reciprocal_rank, 0.0)
        summary = summarize([first, second, missed, empty])
        self.assertEqual(summary.hits, 2)
        self.assertEqual(summary.hit_rate, 0.5)
        self.assertEqual(summary.mrr, 0.375)
        with self.assertRaises(ValueError):
            score_ranking(case, [10, 10], 3)
        with self.assertRaises(ValueError):
            summarize([])

    def test_csv_supports_quotes_and_multiple_relevant_ids(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'queries.csv'
            write_queries(path, [['case', 'paraphrase', 'как заполнить "шаблон", автоматически?', '10;20']])
            cases = load_queries(path)
            self.assertEqual(cases[0].relevant_ids, (10, 20))
            self.assertEqual(cases[0].query, 'как заполнить "шаблон", автоматически?')
            self.assertEqual(query_set_hash(cases), query_set_hash(load_queries(path)))
            different = [EvaluationQuery('case', 'paraphrase', 'другой запрос', (10, 20))]
            self.assertNotEqual(query_set_hash(cases), query_set_hash(different))

    def test_csv_rejects_duplicate_queries_identifiers_and_invalid_ids(self) -> None:
        variants: list[list[list[str]]] = [
            [], [['case', 'exact', '', '1']], [['case', 'exact', 'запрос', '0']],
            [['case', 'exact', 'запрос', '1;1']], [['case', 'exact', 'запрос', 'True']],
            [['case', 'exact', 'один', '1'], ['case', 'exact', 'два', '2']],
            [['one', 'exact', 'Запрос', '1'], ['two', 'exact', 'запрос', '2']],
        ]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'queries.csv'
            for rows in variants:
                write_queries(path, rows)
                with self.subTest(rows=rows), self.assertRaises(ValueError):
                    load_queries(path)

    def test_preflight_rejects_long_queries_and_checks_raw_prompt(self) -> None:
        case = EvaluationQuery('short', 'exact', 'запрос', (1,))
        prepared = prepare_queries([case], CharacterTokenizer())
        self.assertEqual(prepared[0][1], build_query(case.query))
        self.assertEqual(prepare_queries([case], CharacterTokenizer(), True)[0][1], case.query)
        long = EvaluationQuery('long', 'exact', 'я' * 1100, (1,))
        with self.assertRaisesRegex(ValueError, 'long'):
            prepare_queries([case, long], CharacterTokenizer())

    def test_evaluation_reuses_grouped_search_and_does_not_modify_index(self) -> None:
        case = EvaluationQuery('find', 'paraphrase', 'запрос', (2,))
        backend = FakeEmbeddings()
        with closing(QdrantClient(location=':memory:')) as store:
            ensure_collection(store, 'test', 3)
            add_point(store, 1, 1, [1.0, 0.0, 0.0])
            add_point(store, 2, 1, [0.99, 0.01, 0.0])
            add_point(store, 3, 2, [0.8, 0.2, 0.0])
            validate_expected_articles(store, 'test', [case])
            results = list(evaluate_queries(prepare_queries([case], CharacterTokenizer()), store, 'test', backend, 2, 3))
            self.assertEqual(results[0].retrieved_ids, (1, 2))
            self.assertEqual(results[0].first_relevant_rank, 2)
            self.assertEqual(backend.calls, [[build_query(case.query)]])
            self.assertEqual(store.count('test', exact=True).count, 3)
            missing = EvaluationQuery('missing', 'exact', 'другой', (99,))
            with self.assertRaisesRegex(ValueError, '99'):
                validate_expected_articles(store, 'test', [case, missing])

    def test_dry_run_does_not_connect_prompt_download_or_write_report(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'results.csv'
            argv = ['evaluate.py', '--queries', str(DEFAULT_QUERIES), '--output', str(output), '--dry-run']
            with (patch('sys.argv', argv), redirect_stdout(io.StringIO()) as stdout,
                  patch('evaluate.connect_store', side_effect=AssertionError('qdrant не нужен')),
                  patch('evaluate.load_tokenizer', side_effect=AssertionError('скачивание не нужно')),
                  patch('evaluate.read_api_key', side_effect=AssertionError('ключ не нужен')),
                  patch('evaluate.EmbeddingsClient', side_effect=AssertionError('mws не нужен'))):
                self.assertEqual(main(), 0)
            self.assertIn('контрольных запросов: 18', stdout.getvalue())
            self.assertFalse(output.exists())

    def test_existing_report_is_preserved_before_api_access(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'results.csv'
            output.write_text('предыдущий отчёт', encoding='utf-8')
            argv = ['evaluate.py', '--queries', str(DEFAULT_QUERIES), '--output', str(output)]
            with (patch('sys.argv', argv), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()),
                  patch('evaluate.connect_store', side_effect=AssertionError('подключение не нужно')),
                  patch('evaluate.read_api_key', side_effect=AssertionError('ключ не нужен')),
                  self.assertRaises(SystemExit) as error):
                main()
            self.assertEqual(error.exception.code, 1)
            self.assertEqual(output.read_text(encoding='utf-8'), 'предыдущий отчёт')

    def test_missing_expected_article_stops_before_prompt_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as folder, closing(QdrantClient(location=':memory:')) as store:
            source = Path(folder) / 'queries.csv'
            output = Path(folder) / 'report.csv'
            write_queries(source, [['missing', 'exact', 'запрос', '99']])
            name = collection_name(DEFAULT_URL, DEFAULT_MODEL, ChunkingConfig())
            ensure_collection(store, name)
            store.upsert(name, points=[models.PointStruct(
                id=1, vector=[1.0] + [0.0] * (VECTOR_SIZE - 1), payload={'article_id': 1},
            )])
            argv = ['evaluate.py', '--queries', str(source), '--output', str(output),
                    '--qdrant-url', 'http://localhost:6333']
            with (patch('sys.argv', argv), patch.dict('os.environ', {}, clear=True),
                  patch('evaluate.connect_store', return_value=store), redirect_stdout(io.StringIO()),
                  redirect_stderr(io.StringIO()) as stderr,
                  patch('evaluate.read_api_key', side_effect=AssertionError('ключ не нужен')),
                  patch('evaluate.EmbeddingsClient', side_effect=AssertionError('mws не нужен')),
                  self.assertRaises(SystemExit) as error):
                main()
            self.assertEqual(error.exception.code, 1)
            self.assertIn('99', stderr.getvalue())
            self.assertFalse(output.exists())

    def test_cli_preserves_completed_rows_on_failure_and_summarizes_only_success(self) -> None:
        for failure in [None, 2]:
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as folder:
                source = Path(folder) / 'queries.csv'
                output = Path(folder) / 'report.csv'
                write_queries(source, [
                    ['one', 'exact', 'первый запрос', '1'],
                    ['two', 'paraphrase', 'второй запрос', '1'],
                ])
                backend = FakeEmbeddings(VECTOR_SIZE, failure)
                with closing(QdrantClient(location=':memory:')) as store:
                    name = collection_name(DEFAULT_URL, DEFAULT_MODEL, ChunkingConfig())
                    ensure_collection(store, name)
                    store.upsert(name, points=[models.PointStruct(
                        id=1, vector=[1.0] + [0.0] * (VECTOR_SIZE - 1), payload={
                            'article_id': 1, 'title': 'тест', 'url': 'https://example.com/1', 'text': 'текст',
                        },
                    )])
                    argv = ['evaluate.py', '--queries', str(source), '--output', str(output),
                            '--qdrant-url', 'http://localhost:6333']
                    with (patch('sys.argv', argv), patch.dict('os.environ', {}, clear=True),
                          patch('evaluate.connect_store', return_value=store),
                          patch('evaluate.load_tokenizer', return_value=CharacterTokenizer()),
                          patch('evaluate.read_api_key', return_value='test-key'),
                          patch('evaluate.EmbeddingsClient', return_value=backend),
                          redirect_stdout(io.StringIO()) as stdout, redirect_stderr(io.StringIO())):
                        if failure is None:
                            self.assertEqual(main(), 0)
                        else:
                            with self.assertRaises(SystemExit) as error:
                                main()
                            self.assertEqual(error.exception.code, 1)
                with output.open(encoding='utf-8', newline='') as report:
                    rows = list(csv.DictReader(report))
                self.assertTrue(backend.closed)
                self.assertEqual(len(rows), 2 if failure is None else 1)
                self.assertEqual(rows[0]['first_relevant_rank'], '1')
                self.assertEqual(rows[0]['retrieved_article_ids'], '1')
                self.assertEqual(rows[0]['collection_points'], '1')
                self.assertNotIn('test-key', output.read_text(encoding='utf-8'))
                if failure is None:
                    self.assertIn('MRR@5: 1.0000', stdout.getvalue())
                else:
                    self.assertNotIn('MRR@5:', stdout.getvalue())


if __name__ == '__main__':
    unittest.main()
