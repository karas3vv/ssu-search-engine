"""проверяем поиск без mws на синтетических векторах и настоящем локальном qdrant."""
from __future__ import annotations

import io
import unittest
from contextlib import closing, redirect_stderr
from unittest.mock import patch

from qdrant_client import QdrantClient, models

from index import ensure_collection
from search import QUERY_INSTRUCTION, build_query, main, search_articles, validate_collection


def point(point_id: int, article_id: int, vector: list[float], text: str) -> models.PointStruct:
    return models.PointStruct(id=point_id, vector=vector, payload={
        'article_id': article_id, 'title': f'статья {article_id}',
        'url': f'https://habr.com/ru/articles/{article_id}/', 'text': text,
    })


class SearchTests(unittest.TestCase):
    def test_query_instruction_and_raw_mode(self) -> None:
        self.assertEqual(build_query('  запрос  '), f'<instruct>{QUERY_INSTRUCTION}\n<query>запрос')
        self.assertEqual(build_query(' запрос ', raw=True), 'запрос')
        with self.assertRaises(ValueError):
            build_query(' \n ')

    def test_groups_articles_and_returns_best_fragment_in_score_order(self) -> None:
        with closing(QdrantClient(location=':memory:')) as store:
            ensure_collection(store, 'test', 3)
            store.upsert('test', points=[
                point(1, 1, [1.0, 0.0, 0.0], 'лучший'),
                point(2, 1, [0.99, 0.01, 0.0], 'ещё один'),
                point(3, 2, [0.8, 0.2, 0.0], 'другая статья'),
                point(4, 3, [0.0, 1.0, 0.0], 'третья'),
            ])
            before = store.count('test', exact=True).count
            validate_collection(store, 'test', 3)
            results = search_articles(store, 'test', [1.0, 0.0, 0.0], 2, 3)
            self.assertEqual([result.article_id for result in results], [1, 2])
            self.assertEqual(results[0].text, 'лучший')
            self.assertAlmostEqual(results[0].score, 1.0, places=5)
            self.assertGreater(results[0].score, results[1].score)
            self.assertEqual(results[1].url, 'https://habr.com/ru/articles/2/')
            self.assertEqual(store.count('test', exact=True).count, before)

    def test_missing_empty_or_incompatible_index_is_not_modified(self) -> None:
        with closing(QdrantClient(location=':memory:')) as store:
            with self.assertRaises(ValueError):
                validate_collection(store, 'missing', 3)
            self.assertFalse(store.collection_exists('missing'))
            ensure_collection(store, 'empty', 3)
            with self.assertRaises(ValueError):
                validate_collection(store, 'empty', 3)
            with self.assertRaises(ValueError):
                validate_collection(store, 'empty', 4)
            self.assertEqual(store.count('empty', exact=True).count, 0)

    def test_invalid_vectors_and_limit_are_rejected(self) -> None:
        with closing(QdrantClient(location=':memory:')) as store:
            for vector in [[], [1.0], [0.0, 0.0, 0.0], [float('nan'), 1.0, 0.0]]:
                with self.subTest(vector=vector), self.assertRaises(ValueError):
                    search_articles(store, 'unused', vector, vector_size=3)
            with self.assertRaises(ValueError):
                search_articles(store, 'unused', [1.0, 0.0, 0.0], 0, 3)

    def test_missing_collection_does_not_request_key_or_call_mws(self) -> None:
        with closing(QdrantClient(location=':memory:')) as store:
            argv = ['search.py', '--query', 'запрос', '--qdrant-url', 'http://localhost:6333']
            with (patch('sys.argv', argv), patch('search.connect_store', return_value=store),
                  patch('search.read_api_key', side_effect=AssertionError('ключ не нужен')),
                  patch('search.EmbeddingsClient', side_effect=AssertionError('mws не нужен')),
                  patch('search.load_tokenizer', side_effect=AssertionError('токенизатор не нужен')),
                  redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error):
                main()
            self.assertEqual(error.exception.code, 1)


if __name__ == '__main__':
    unittest.main()
