"""проверяем объединение, один вызов модели и продолжение оценки после сбоя."""
from __future__ import annotations

import csv
import io
import json
import os
import sqlite3
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from habr_search.evaluation.evaluate import EvaluationQuery, prepare_queries
from habr_search.evaluation.evaluate_hybrid import HybridContext, evaluate_cases, load_saved, main as evaluate_main
from habr_search.index_version import invalidate_index
from habr_search.ingestion.chunking import ChunkingConfig, chunk_article
from habr_search.ingestion.index import collection_name, ensure_collection, index_chunks
from habr_search.embeddings import DEFAULT_URL
from habr_search.model_profiles import BGE_M3
from habr_search.retrieval.hybrid import HybridConfig, HybridRetriever, fuse_rankings
from habr_search.retrieval.tantivy_index import TantivySearch, build_index
from habr_search.web.app import HybridService, TantivyService, create_app, main as web_main
from tests.test_chunking import CharacterTokenizer, article
from tests.test_evaluate import FakeEmbeddings


class FusionTests(unittest.TestCase):
    def test_both_sources_contribute_without_duplicate_articles(self) -> None:
        config = HybridConfig(3, 60)
        results = fuse_rankings([1, 2, 3], [3, 4, 2], 3, config)
        self.assertEqual(results[0].article_id, 3)
        self.assertEqual(results[0].dense_rank, 3)
        self.assertEqual(results[0].lexical_rank, 1)
        self.assertAlmostEqual(results[0].score, 1 / 63 + 1 / 61)
        self.assertEqual(len({item.article_id for item in results}), 3)
        self.assertEqual(fuse_rankings([], [], 1), [])
        self.assertEqual([item.article_id for item in fuse_rankings([], [3, 1], 2)], [3, 1])

    def test_ties_are_deterministic_and_invalid_ranks_are_rejected(self) -> None:
        self.assertEqual([item.article_id for item in fuse_rankings([2], [1], 2)], [2, 1])
        for dense in ([1, 1], [0], [-1], [True]):
            with self.assertRaises(ValueError):
                fuse_rankings(dense, [], 1)
        for config in ((0, 60), (101, 60), (20, 0), (20, float('nan'))):
            with self.assertRaises(ValueError):
                HybridConfig(*config)


class HybridTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'corpus.sqlite3'
        self.local_path = self.root / 'tantivy'
        self.store_path = self.root / 'qdrant'
        self.name = collection_name(DEFAULT_URL, 'bge-m3', ChunkingConfig(), BGE_M3)
        documents = [article('автоматизация документов на Python', article_id=1),
                     article('автоматизация звонков и телефония', article_id=2),
                     article('автоматизация учета товаров', article_id=3)]
        with sqlite3.connect(self.source) as connection:
            connection.execute('CREATE TABLE articles (id INTEGER PRIMARY KEY, payload TEXT NOT NULL)')
            for document in documents:
                connection.execute('INSERT INTO articles VALUES (?, ?)', (document.id, json.dumps({
                    'id': document.id, 'url': document.url, 'title': document.title, 'text': document.text})))
        build_index(self.source, self.local_path)
        self.store = QdrantClient(path=str(self.store_path))
        self.addCleanup(self.store.close)
        ensure_collection(self.store, self.name, 1024)
        self.embeddings = FakeEmbeddings(1024)
        for document in documents:
            chunks = list(chunk_article(document, CharacterTokenizer(), ChunkingConfig(), BGE_M3))
            index_chunks(chunks, document.text_hash, self.store, self.name, self.embeddings, profile=BGE_M3)
        self.embeddings.calls.clear()
        self.local = TantivySearch(self.local_path)

    def test_retrieval_and_evaluation_reuse_one_vector(self) -> None:
        retriever = HybridRetriever(self.store, self.name, self.local)
        cases = [EvaluationQuery('python', 'exact', 'Python', (1,))]
        result = list(evaluate_cases(prepare_queries(cases, CharacterTokenizer(), profile=BGE_M3),
                                     retriever, self.embeddings, 3))[0]
        self.assertEqual(self.embeddings.calls, [['Python']])
        self.assertEqual(result.lexical_ids, (1,))
        self.assertEqual(result.hybrid_ids[0], 1)
        self.assertEqual(result.metric('hybrid', 2).first_relevant_rank, 1)
        self.assertEqual(result.metric('bge-m3', 3).first_relevant_rank, result.dense_ids.index(1) + 1)

    def test_web_preserves_query_and_defers_spelling_without_extra_embeddings(self) -> None:
        local = TantivyService(self.local_path)
        service = HybridService(self.store, self.name, CharacterTokenizer(), self.embeddings, local)
        with TestClient(create_app(lambda: service)) as client:
            result = client.post('/api/search', json={'query': 'автоматизаця документов', 'top_k': 20})
            self.assertEqual(result.status_code, 200)
            self.assertEqual(result.json()['used_query'], 'автоматизаця документов')
            self.assertTrue(result.json()['deferred_suggestion'])
            self.assertEqual(len(self.embeddings.calls), 1)
            suggestion = client.post('/api/suggest', json={'query': 'автоматизаця документов'}).json()
            self.assertEqual(suggestion['corrected_query'], 'автоматизация документов')
            self.assertEqual(len(self.embeddings.calls), 1)
            corrected = client.post('/api/search', json={'query': 'автоматизаця документов', 'correct_typos': True}).json()
            self.assertEqual(corrected['used_query'], 'автоматизация документов')
            self.assertEqual(len(self.embeddings.calls), 2)

    def test_mismatched_or_changed_sources_are_rejected_before_payment(self) -> None:
        self.store.delete_collection(self.name)
        ensure_collection(self.store, self.name, 1024)
        document = article('другая статья', article_id=3)
        index_chunks(list(chunk_article(document, CharacterTokenizer(), ChunkingConfig(), BGE_M3)),
                     document.text_hash, self.store, self.name, self.embeddings, profile=BGE_M3)
        with self.assertRaisesRegex(ValueError, 'наборы статей'):
            HybridRetriever(self.store, self.name, self.local)

    def test_index_change_and_invalid_query_do_not_call_model(self) -> None:
        service = HybridService(self.store, self.name, CharacterTokenizer(), self.embeddings,
                                TantivyService(self.local_path, spelling=False))
        with TestClient(create_app(lambda: service)) as client:
            response = client.post('/api/search', json={'query': 'a' * 81})
            self.assertEqual(response.status_code, 400)
            self.assertEqual(len(self.embeddings.calls), 0)
            invalidate_index(self.store, self.name)
            response = client.post('/api/search', json={'query': 'Python'})
            self.assertEqual(response.status_code, 400)
            self.assertEqual(len(self.embeddings.calls), 0)

    def test_web_cli_uses_m3_and_local_dictionary(self) -> None:
        self.store.close()
        provider = FakeEmbeddings(1024)

        def serve(app: FastAPI, **_options: object) -> None:
            with TestClient(app) as client:
                response = client.post('/api/search', json={'query': 'Python'})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()['results'][0]['article_id'], 1)
                self.assertTrue(response.json()['deferred_suggestion'])

        with patch.dict(os.environ, {}, clear=True), redirect_stdout(io.StringIO()), \
                patch('sys.argv', ['web', '--engine', 'hybrid', '--qdrant-path', str(self.store_path),
                                   '--tantivy-path', str(self.local_path)]), \
                patch('habr_search.web.app.read_api_key', return_value='test'), \
                patch('habr_search.web.app.load_tokenizer', return_value=CharacterTokenizer()), \
                patch('habr_search.web.app.EmbeddingsClient', return_value=provider) as constructor, \
                patch('habr_search.web.app.uvicorn.run', side_effect=serve):
            self.assertEqual(web_main(), 0)
            self.assertEqual(constructor.call_args.args[0].model, 'bge-m3')
        self.assertEqual(provider.calls, [['Python']])

    def test_resume_after_failure_never_repeats_saved_paid_queries(self) -> None:
        self.store.close()
        cases = self.root / 'queries.csv'
        cases.write_text('query_id,category,query,relevant_article_ids\n'
                         'python,exact,Python,1\nphone,exact,телефония,2\n')
        first, second = self.root / 'first.csv', self.root / 'second.csv'
        common = ['evaluate', '--qdrant-path', str(self.store_path), '--tantivy-path', str(self.local_path),
                  '--queries', str(cases)]
        failing = FakeEmbeddings(1024, fail_on_call=2)
        with patch.dict(os.environ, {}, clear=True), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), \
                patch('sys.argv', [*common, '--output', str(first)]), \
                patch('habr_search.evaluation.evaluate_hybrid.load_tokenizer', return_value=CharacterTokenizer()), \
                patch('habr_search.evaluation.evaluate_hybrid.read_api_key', return_value='test'), \
                patch('habr_search.evaluation.evaluate_hybrid.EmbeddingsClient', return_value=failing), \
                self.assertRaises(SystemExit):
            evaluate_main()
        with first.open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 1)
        resumed = FakeEmbeddings(1024)
        with patch.dict(os.environ, {}, clear=True), redirect_stdout(io.StringIO()), \
                patch('sys.argv', [*common, '--output', str(second), '--resume', str(first)]), \
                patch('habr_search.evaluation.evaluate_hybrid.load_tokenizer', return_value=CharacterTokenizer()), \
                patch('habr_search.evaluation.evaluate_hybrid.read_api_key', return_value='test'), \
                patch('habr_search.evaluation.evaluate_hybrid.EmbeddingsClient', return_value=resumed):
            self.assertEqual(evaluate_main(), 0)
        self.assertEqual(resumed.calls, [['телефония']])
        with patch.dict(os.environ, {}, clear=True), redirect_stdout(io.StringIO()), \
                patch('sys.argv', [*common, '--output', str(self.root / 'third.csv'), '--resume', str(second)]), \
                patch('habr_search.evaluation.evaluate_hybrid.read_api_key', side_effect=AssertionError('ключ не нужен')):
            self.assertEqual(evaluate_main(), 0)
        # изменённые параметры исключают смешивание разных экспериментов
        context = HybridContext(self.name, 'wrong-revision', self.local.info.revision, 2, 'wrong-hash', 5, HybridConfig())
        with self.assertRaises(ValueError):
            load_saved(second, [], context)


if __name__ == '__main__':
    unittest.main()
