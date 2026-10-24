"""проверяем настоящий rust-индекс и http-поиск на небольшом локальном корпусе."""
from __future__ import annotations

import csv
import io
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from habr_search.retrieval.tantivy_index import TantivySearch, build_index
from habr_search.web.app import TantivyService, create_app, main
from habr_search.evaluation.evaluate_tantivy import main as evaluate_main


class TantivyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'corpus.sqlite3'
        self.index = self.root / 'index'
        with sqlite3.connect(self.source) as connection:
            connection.execute('CREATE TABLE articles (id INTEGER PRIMARY KEY, payload TEXT NOT NULL)')
            for article_id, title, text in [
                (1, 'Автоматизация документов', 'Начало статьи. ' * 100 + 'Документами управляет программа Python.'),
                (2, 'Телефония дома', 'Настройка домашней телефонии и подключение телефонного аппарата.'),
            ]:
                payload = {'id': article_id, 'url': f'https://habr.com/ru/articles/{article_id}/',
                           'title': title, 'text': text}
                connection.execute('INSERT INTO articles VALUES (?, ?)', (article_id, json.dumps(payload)))
        build_index(self.source, self.index)

    def test_persistent_index_stemming_and_contextual_snippet(self) -> None:
        backend = TantivySearch(self.index)
        found = backend.search('документами Python')
        self.assertEqual([item.article_id for item in found], [1])
        self.assertIn('Python', found[0].snippet)
        self.assertEqual(backend.info.articles, 2)
        self.assertEqual(backend.search('телефонию')[0].article_id, 2)

    def test_query_syntax_is_literal_and_bounds_are_checked(self) -> None:
        backend = TantivySearch(self.index)
        self.assertEqual(backend.search('*:*'), [])
        self.assertEqual(backend.search('zzunknownzz'), [])
        self.assertEqual(backend.search('"Python" OR body:*')[0].article_id, 1)
        for query in ['', 'a' * 81, ' '.join(str(i) for i in range(65))]:
            with self.assertRaises(ValueError):
                backend.search(query)
        with self.assertRaises(ValueError):
            backend.search('python', 21)

    def test_existing_index_is_preserved_and_source_is_read_only(self) -> None:
        original = self.source.read_bytes()
        revision = TantivySearch(self.index).info.revision
        with self.assertRaises(ValueError):
            build_index(self.source, self.index)
        self.assertEqual(original, self.source.read_bytes())
        self.assertEqual(revision, TantivySearch(self.index).info.revision)
        self.assertEqual(list(self.root.glob('.tantivy-*')), [])

    def test_failed_build_leaves_no_published_index(self) -> None:
        destination = self.root / 'failed'
        with self.assertRaises((ValueError, OSError)):
            build_index(self.root / 'missing.sqlite3', destination)
        self.assertFalse(destination.exists())
        self.assertEqual(list(self.root.glob('.tantivy-*')), [])

    def test_web_needs_no_model_or_qdrant_and_preserves_original_query(self) -> None:
        with patch('habr_search.web.app.read_api_key', side_effect=AssertionError('ключ не нужен')), \
                patch('habr_search.web.app.connect_store', side_effect=AssertionError('база не нужна')), \
                patch('habr_search.web.app.EmbeddingsClient', side_effect=AssertionError('модель не нужна')), \
                TestClient(create_app(lambda: TantivyService(self.index))) as client:
            self.assertEqual(client.get('/api/health').json()['articles'], 2)
            response = client.post('/api/search', json={'query': 'Python', 'top_k': 20}).json()
            self.assertEqual(response['used_query'], 'Python')
            self.assertFalse(response['corrected'])
            self.assertTrue(response['deferred_suggestion'])
            self.assertEqual(response['results'][0]['article_id'], 1)
            self.assertEqual(client.post('/api/suggest', json={'query': 'Python'}).status_code, 200)

    def test_cli_starts_local_service_before_reading_any_key(self) -> None:
        def serve(app: FastAPI, **_options: object) -> None:
            with TestClient(app) as client:
                self.assertEqual(client.post('/api/search', json={'query': 'Python'}).status_code, 200)

        with patch('sys.argv', ['app', '--engine', 'tantivy', '--tantivy-path', str(self.index)]), \
                patch('habr_search.web.app.read_api_key', side_effect=AssertionError('ключ не нужен')), \
                patch('habr_search.web.app.uvicorn.run', side_effect=serve) as server, redirect_stdout(io.StringIO()):
            self.assertEqual(main(), 0)
            server.assert_called_once()

    def test_evaluation_writes_separate_report_and_never_overwrites_it(self) -> None:
        queries = self.root / 'queries.csv'
        queries.write_text('query_id,category,query,relevant_article_ids\n'
                           'python,exact,Python,1\nmissing,no_answer,zzunknownzz,\n')
        output = self.root / 'report.csv'
        arguments = ['evaluate', '--index-path', str(self.index), '--queries', str(queries), '--output', str(output)]
        with patch('sys.argv', arguments), redirect_stdout(io.StringIO()):
            self.assertEqual(evaluate_main(), 0)
        with output.open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([row['engine'] for row in rows], ['tantivy', 'tantivy'])
        self.assertEqual(rows[0]['first_relevant_rank'], '1')
        self.assertEqual(rows[1]['retrieved_article_ids'], '')
        before = output.read_bytes()
        with patch('sys.argv', arguments), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            evaluate_main()
        self.assertEqual(output.read_bytes(), before)

    def test_corrupt_manifest_is_rejected(self) -> None:
        (self.index / 'manifest.json').write_text('{}')
        with self.assertRaises(ValueError):
            TantivySearch(self.index)


if __name__ == '__main__':
    unittest.main()
