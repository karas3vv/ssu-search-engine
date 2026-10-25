"""проверяем поиск без mws на синтетических векторах и настоящем локальном qdrant."""
from __future__ import annotations

import io
import unittest
from contextlib import closing, redirect_stderr
from unittest.mock import patch

from qdrant_client import QdrantClient, models

from habr_search.ingestion.index import ensure_collection
from habr_search.retrieval.search import build_query, main, search_articles, validate_collection


# создаём точку с заранее заданным вектором и метаданными статьи
def point(point_id: int, article_id: int, vector: list[float], text: str) -> models.PointStruct:
    return models.PointStruct(id=point_id, vector=vector, payload={
        'article_id': article_id, 'title': f'статья {article_id}',
        'url': f'https://habr.com/ru/articles/{article_id}/', 'text': text,
    })


# группа проверок модуля; каждый метод test_ описывает отдельный ожидаемый сценарий
class SearchTests(unittest.TestCase):
    # порог отсекает статьи по лучшему фрагменту и допускает пустую выдачу
    def test_threshold_filters_groups_without_modifying_index(self) -> None:
        with closing(QdrantClient(':memory:')) as store:
            ensure_collection(store, 'test', 3)
            store.upsert('test', points=[
                point(1, 1, [0.6, 0.8, 0.0], 'сильный'),
                point(2, 1, [0.0, 1.0, 0.0], 'слабый того же автора'),
                point(3, 2, [0.0, 1.0, 0.0], 'другая статья'),
            ])
            before = store.scroll('test', with_vectors=True)
            found = search_articles(store, 'test', [1.0, 0.0, 0.0], 5, 3, min_score=0.5)
            self.assertEqual([item.article_id for item in found], [1])
            self.assertEqual(found[0].text, 'сильный')
            self.assertEqual(search_articles(store, 'test', [1.0, 0.0, 0.0], 5, 3, min_score=0.7), [])
            self.assertEqual(store.scroll('test', with_vectors=True), before)

    # неверный порог отклоняется до подключения, загрузки токенизатора и запроса ключа
    def test_invalid_threshold_stops_before_resources_or_payment(self) -> None:
        for value in ['nan', 'inf', '-1.1', '1.1']:
            with (self.subTest(value=value), patch('sys.argv', ['search', '--query', 'тест', f'--min-score={value}']),
                  patch('habr_search.retrieval.search.connect_store', side_effect=AssertionError('qdrant не нужен')),
                  patch('habr_search.retrieval.search.read_api_key', side_effect=AssertionError('ключ не нужен')),
                  redirect_stderr(io.StringIO()), self.assertRaises(SystemExit)):
                main()

    # убираем крайние пробелы и отклоняем пустой запрос
    def test_plain_query_and_empty_input(self) -> None:
        self.assertEqual(build_query('  запрос  '), 'запрос')
        with self.assertRaises(ValueError):
            build_query(' \n ')

    # соседние фрагменты одной статьи не занимают несколько мест в выдаче
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

    # поиск отклоняет неподходящую коллекцию и не создаёт её автоматически
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

    # нулевой, нечисловой или неверный по длине вектор отклоняется до поиска
    def test_invalid_vectors_and_limit_are_rejected(self) -> None:
        with closing(QdrantClient(location=':memory:')) as store:
            for vector in [[], [1.0], [0.0, 0.0, 0.0], [float('nan'), 1.0, 0.0]]:
                with self.subTest(vector=vector), self.assertRaises(ValueError):
                    search_articles(store, 'unused', vector, vector_size=3)
            with self.assertRaises(ValueError):
                search_articles(store, 'unused', [1.0, 0.0, 0.0], 0, 3)

    # если коллекции нет, поиск останавливается до платного обращения к модели
    def test_missing_collection_does_not_request_key_or_call_mws(self) -> None:
        with closing(QdrantClient(location=':memory:')) as store:
            argv = ['search.py', '--query', 'запрос', '--qdrant-url', 'http://localhost:6333']
            with (patch('sys.argv', argv), patch('habr_search.retrieval.search.connect_store', return_value=store),
                  patch('habr_search.retrieval.search.read_api_key', side_effect=AssertionError('ключ не нужен')),
                  patch('habr_search.retrieval.search.EmbeddingsClient', side_effect=AssertionError('mws не нужен')),
                  patch('habr_search.retrieval.search.load_tokenizer', side_effect=AssertionError('токенизатор не нужен')),
                  redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error):
                main()
            self.assertEqual(error.exception.code, 1)


# при прямом запуске выполняем тесты этого файла
if __name__ == '__main__':
    unittest.main()
