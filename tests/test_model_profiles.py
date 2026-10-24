"""проверяем bge-m3 на подставных векторах, не вызывая платный api."""
from __future__ import annotations

import io
import json
import os
import sqlite3
import unittest
from contextlib import closing, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from habr_search.embeddings import DEFAULT_URL, EmbeddingConfig, main as probe_main
from habr_search.evaluation.evaluate import EvaluationQuery, main as evaluate_main, prepare_queries
from habr_search.ingestion.chunking import ChunkingConfig, ModelTokenizer, chunk_article
from habr_search.ingestion.index import collection_name, ensure_collection, index_chunks, main as index_main
from habr_search.model_profiles import BGE_M3, GEMMA
from habr_search.retrieval.search import build_query, main as search_main, validate_collection
from habr_search.retrieval.spellcheck import main as spellcheck_main
from habr_search.web.app import SearchService, create_app, main as web_main
from tests.test_chunking import CharacterTokenizer, article
from tests.test_evaluate import FakeEmbeddings


class ModelProfileTests(unittest.TestCase):
    def test_gemma_collection_and_chunk_ids_are_preserved(self) -> None:
        self.assertEqual(collection_name(DEFAULT_URL, 'parser', ChunkingConfig()), 'habr_gemma2_ddb30cefb72f')
        self.assertTrue(collection_name(DEFAULT_URL, 'bge-m3', ChunkingConfig(), BGE_M3).startswith('habr_bge_m3_'))
        # один и тот же деплоймент также не позволяет смешать разные модели
        self.assertNotEqual(collection_name(DEFAULT_URL, 'parser', ChunkingConfig(), BGE_M3),
                            collection_name(DEFAULT_URL, 'parser', ChunkingConfig()))
        doc = article('полный текст статьи')
        legacy = list(chunk_article(doc, CharacterTokenizer(), ChunkingConfig()))
        explicit = list(chunk_article(doc, CharacterTokenizer(), ChunkingConfig(), GEMMA))
        changed = list(chunk_article(doc, CharacterTokenizer(), ChunkingConfig(), BGE_M3))
        self.assertEqual(legacy, explicit)
        self.assertNotEqual(legacy[0].id, changed[0].id)

    def test_m3_tokenizer_uses_its_own_special_tokens(self) -> None:
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'tokenizer.json'
            tokenizer = Tokenizer(WordLevel({'<s>': 0, '<pad>': 1, '</s>': 2, '<unk>': 3, 'текст': 4}, unk_token='<unk>'))
            tokenizer.save(str(path))
            actual = ModelTokenizer(path, BGE_M3)
            self.assertEqual(actual.tokenizer.encode('текст').ids, [0, 4, 2])
            self.assertEqual(actual.count('текст'), 3)
            with self.assertRaises(ValueError):
                ModelTokenizer(path, GEMMA)
        self.assertNotEqual(GEMMA.tokenizer_file, BGE_M3.tokenizer_file)

    def test_m3_query_and_evaluation_have_no_gemma_instruction(self) -> None:
        self.assertEqual(build_query('  документы  ', profile=BGE_M3), 'документы')
        case = EvaluationQuery('m3', 'exact', 'документы', (1,))
        self.assertEqual(prepare_queries([case], CharacterTokenizer(), profile=BGE_M3)[0][1], 'документы')
        self.assertIn('<instruct>', build_query('документы'))
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(EmbeddingConfig.from_env('test', BGE_M3).model, 'bge-m3')

    def test_m3_vectors_payload_resume_and_web_search(self) -> None:
        doc = article('текст документов ' * 100)
        chunks = list(chunk_article(doc, CharacterTokenizer(), ChunkingConfig(), BGE_M3))
        embeddings = FakeEmbeddings(BGE_M3.vector_size)
        with closing(QdrantClient(':memory:')) as store:
            ensure_collection(store, 'm3', BGE_M3.vector_size)
            stats = index_chunks(chunks, doc.text_hash, store, 'm3', embeddings, profile=BGE_M3)
            self.assertEqual(stats.saved, len(chunks))
            before = len(embeddings.calls)
            repeated = index_chunks(chunks, doc.text_hash, store, 'm3', embeddings, profile=BGE_M3)
            self.assertEqual(repeated.skipped, len(chunks))
            self.assertEqual(before, len(embeddings.calls))
            payload = store.retrieve('m3', [chunks[0].id])[0].payload
            self.assertIsNotNone(payload)
            if payload is not None:
                self.assertEqual(payload['model'], 'BAAI/bge-m3')
            with self.assertRaises(ValueError):
                validate_collection(store, 'm3')
            service = SearchService(store, 'm3', CharacterTokenizer(), embeddings, None, profile=BGE_M3)
            with TestClient(create_app(lambda: service)) as client:
                result = client.post('/api/search', json={'query': 'документы'})
                self.assertEqual(result.status_code, 200)
                self.assertEqual(result.json()['results'][0]['article_id'], doc.id)
                self.assertEqual(embeddings.calls[-1], ['документы'])

    def test_wrong_dimension_never_writes_points(self) -> None:
        doc = article('короткий текст')
        chunks = list(chunk_article(doc, CharacterTokenizer(), ChunkingConfig(), BGE_M3))
        with closing(QdrantClient(':memory:')) as store:
            ensure_collection(store, 'm3', BGE_M3.vector_size)
            with self.assertRaisesRegex(ValueError, '1024'):
                index_chunks(chunks, doc.text_hash, store, 'm3', FakeEmbeddings(GEMMA.vector_size), profile=BGE_M3)
            self.assertEqual(store.count('m3').count, 0)

    def test_all_m3_commands_use_the_same_collection(self) -> None:
        # строим небольшой постоянный индекс, затем используем его в других командах
        with TemporaryDirectory() as folder, patch.dict(os.environ, {}, clear=True), redirect_stdout(io.StringIO()):
            root = Path(folder)
            source, database = root / 'corpus.sqlite3', root / 'qdrant'
            doc = article('документы и автоматизация')
            with sqlite3.connect(source) as connection:
                connection.execute('CREATE TABLE articles (id INTEGER PRIMARY KEY, payload TEXT NOT NULL)')
                connection.execute('INSERT INTO articles VALUES (?, ?)', (doc.id, json.dumps({
                    'id': doc.id, 'url': doc.url, 'title': doc.title, 'text': doc.text})))
            common = ['--embedding-model', 'bge-m3', '--qdrant-path', str(database)]
            provider = FakeEmbeddings(BGE_M3.vector_size)
            with patch('sys.argv', ['index', *common, '--source', str(source)]), \
                    patch('habr_search.ingestion.index.load_tokenizer', return_value=CharacterTokenizer()), \
                    patch('habr_search.ingestion.index.read_api_key', return_value='test'), \
                    patch('habr_search.ingestion.index.EmbeddingsClient', return_value=provider):
                self.assertEqual(index_main(), 0)
            queries = root / 'queries.csv'
            queries.write_text('query_id,category,query,relevant_article_ids\nq,exact,документы,1\n')
            with patch('sys.argv', ['evaluate', *common, '--queries', str(queries), '--output', str(root / 'report.csv')]), \
                    patch('habr_search.evaluation.evaluate.load_tokenizer', return_value=CharacterTokenizer()), \
                    patch('habr_search.evaluation.evaluate.read_api_key', return_value='test'), \
                    patch('habr_search.evaluation.evaluate.EmbeddingsClient', return_value=FakeEmbeddings(1024)):
                self.assertEqual(evaluate_main(), 0)
            dictionary = root / 'spelling.sqlite3'
            with patch('sys.argv', ['spellcheck', 'build', *common, '--dictionary', str(dictionary)]):
                self.assertEqual(spellcheck_main(), 0)
            search_provider = FakeEmbeddings(1024)
            with patch('sys.argv', ['search', *common, '--query', 'документы']), \
                    patch('habr_search.retrieval.search.load_tokenizer', return_value=CharacterTokenizer()), \
                    patch('habr_search.retrieval.search.read_api_key', return_value='test'), \
                    patch('habr_search.retrieval.search.EmbeddingsClient', return_value=search_provider):
                self.assertEqual(search_main(), 0)
                self.assertEqual(search_provider.calls, [['документы']])
            with patch('sys.argv', ['probe', '--embedding-model', 'bge-m3']), \
                    patch('habr_search.embeddings.read_api_key', return_value='test'), \
                    patch('habr_search.embeddings.EmbeddingsClient', return_value=FakeEmbeddings(1024)) as constructor:
                self.assertEqual(probe_main(), 0)
                self.assertEqual(constructor.call_args.args[0].model, 'bge-m3')
            # сервер действительно открывает ту же коллекцию и отправляет обычный запрос
            web_provider = FakeEmbeddings(1024)

            def serve(app: FastAPI, **_options: object) -> None:
                with TestClient(app) as client:
                    response = client.post('/api/search', json={'query': 'документы'})
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.json()['results'][0]['article_id'], 1)

            with patch('sys.argv', ['web', '--engine', 'bge-m3', '--qdrant-path', str(database),
                                    '--spelling-db', str(dictionary)]), \
                    patch('habr_search.web.app.read_api_key', return_value='test'), \
                    patch('habr_search.web.app.load_tokenizer', return_value=CharacterTokenizer()), \
                    patch('habr_search.web.app.EmbeddingsClient', return_value=web_provider), \
                    patch('habr_search.web.app.uvicorn.run', side_effect=serve):
                self.assertEqual(web_main(), 0)
                self.assertEqual(web_provider.calls, [['документы']])



if __name__ == '__main__':
    unittest.main()
