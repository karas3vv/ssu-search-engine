"""проверяем http, исправления и ошибки на синтетическом индексе без mws."""
from __future__ import annotations

import unittest
from contextlib import closing
from dataclasses import replace

from fastapi.testclient import TestClient
from qdrant_client import QdrantClient, models

from habr_search.ingestion.index import VECTOR_SIZE, ensure_collection
from habr_search.retrieval.spelling import DictionaryInfo, SpellingDictionary, dictionary_hash
from tests.test_evaluate import CharacterTokenizer, FakeEmbeddings
from habr_search.web.app import SearchService, create_app
from habr_search.index_version import invalidate_index


# собираем сервис с временным индексом и подставным клиентом модели
def make_service(store: QdrantClient) -> tuple[SearchService, FakeEmbeddings]:
    ensure_collection(store, 'test')
    vector = [1.0] + [0.0] * (VECTOR_SIZE - 1)
    store.upsert('test', [models.PointStruct(id=i, vector=vector, payload={
        'article_id': i, 'title': '<img src=x onerror=alert(1)> Python',
        'url': f'https://habr.com/ru/articles/{i}/', 'text': 'текст\n' * 200,
    }) for i in range(1, 4)])
    words = {'python': 3}
    dictionary = SpellingDictionary(words, DictionaryInfo('test', 3, 3, dictionary_hash(words),
                                                          '2026-10-19T00:00:00+00:00', index_revision='unversioned'))
    embeddings = FakeEmbeddings(VECTOR_SIZE)
    service = SearchService(store, 'test', CharacterTokenizer(), embeddings, dictionary)
    service.resources.callback(embeddings.close)
    return service, embeddings


# проверяем страницу, форматы api, обработку ошибок и число вызовов модели
class WebTests(unittest.TestCase):
    # двадцать кандидатов для кнопки получаем одним вызовом модели и группируем по статьям
    def test_twenty_candidates_use_one_embedding_and_keep_articles_unique(self) -> None:
        with closing(QdrantClient(':memory:')) as store:
            service, embeddings = make_service(store)
            vector = [1.0] + [0.0] * (VECTOR_SIZE - 1)
            store.upsert('test', [models.PointStruct(id=i, vector=vector, payload={
                'article_id': i if i < 26 else 1, 'title': f'статья {i}',
                'url': f'https://habr.com/ru/articles/{i}/', 'text': 'текст',
            }) for i in range(4, 27)])
            before = store.count('test', exact=True).count
            with TestClient(create_app(lambda: service)) as client:
                response = client.post('/api/search', json={'query': 'python', 'top_k': 20})
                self.assertEqual(response.status_code, 200)
                items = response.json()['results']
                self.assertEqual(len(items), 20)
                self.assertEqual(len({item['article_id'] for item in items}), 20)
                self.assertEqual(len(embeddings.calls), 1)
                self.assertEqual(store.count('test', exact=True).count, before)

    # порог сервера разрешает пустую выдачу без повторного обращения к модели
    def test_server_threshold_returns_empty_results_with_one_embedding_call(self) -> None:
        with closing(QdrantClient(':memory:')) as store:
            initial, embeddings = make_service(store)
            store.upsert('test', [models.PointStruct(
                id=i, vector=[0.0, 1.0] + [0.0] * (VECTOR_SIZE - 2), payload={
                    'article_id': i, 'title': 'другая тема', 'url': f'https://habr.com/ru/articles/{i}/', 'text': 'текст',
                }) for i in range(1, 4)])
            service = SearchService(store, 'test', CharacterTokenizer(), embeddings, initial.dictionary, min_score=0.5)
            service.resources.callback(embeddings.close)
            with TestClient(create_app(lambda: service)) as client:
                response = client.post('/api/search', json={'query': 'python'})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()['results'], [])
                self.assertEqual(len(embeddings.calls), 1)
                self.assertEqual(store.count('test', exact=True).count, 3)

    # словарь старого формата отключает подсказки при запуске, но позволяет искать статьи
    def test_legacy_dictionary_keeps_original_search_available(self) -> None:
        with closing(QdrantClient(':memory:')) as store:
            initial, embeddings = make_service(store)
            assert initial.dictionary is not None
            legacy = SpellingDictionary(initial.dictionary.frequencies,
                                        replace(initial.dictionary.info, index_revision=None))
            with self.assertLogs(level='WARNING'):
                service = SearchService(store, 'test', CharacterTokenizer(), embeddings, legacy)
            with TestClient(create_app(lambda: service)) as client:
                self.assertFalse(client.get('/api/health').json()['spelling_available'])
                result = client.post('/api/search', json={'query': 'pyhton'}).json()
                self.assertEqual(result['used_query'], 'pyhton')
                self.assertIsNone(result['suggested_query'])
                self.assertEqual(len(embeddings.calls), 1)

    # версия меняется даже при прежнем числе точек, и обе ручки перестают предлагать старые слова
    def test_changed_revision_disables_suggestions_with_same_point_count(self) -> None:
        with closing(QdrantClient(':memory:')) as store:
            service, embeddings = make_service(store)
            with TestClient(create_app(lambda: service)) as client:
                invalidate_index(store, 'test')
                self.assertFalse(client.post('/api/suggest', json={'query': 'pyhton'}).json()['available'])
                result = client.post('/api/search', json={'query': 'pyhton'}).json()
                self.assertIsNone(result['suggested_query'])
                self.assertEqual(result['used_query'], 'pyhton')
                self.assertEqual(len(embeddings.calls), 1)

    # страница, сведения о корпусе и подсказки работают без вызова модели
    def test_page_health_and_suggestions_do_not_call_model(self) -> None:
        with closing(QdrantClient(location=':memory:')) as store:
            service, embeddings = make_service(store)
            with TestClient(create_app(lambda: service)) as client:
                response = client.get('/')
                self.assertEqual(response.status_code, 200)
                self.assertIn('Поиск по Хабру', response.text)
                self.assertEqual(client.get('/static/app.js').status_code, 200)
                self.assertEqual(client.get('/static/style.css').status_code, 200)
                self.assertEqual(client.get('/api/health').json()['fragments'], 3)
                suggestion = client.post('/api/suggest', json={'query': 'pyhton'})
                self.assertEqual(suggestion.status_code, 200)
                self.assertEqual(suggestion.json()['corrected_query'], 'python')
                self.assertEqual(embeddings.calls, [])
            self.assertTrue(embeddings.closed)

    # флаг запроса управляет исправлениями, а сам поиск не меняет индекс
    def test_search_corrects_only_when_requested_and_preserves_index(self) -> None:
        with closing(QdrantClient(location=':memory:')) as store:
            service, embeddings = make_service(store)
            before = store.scroll('test', limit=10, with_vectors=True)
            with TestClient(create_app(lambda: service)) as client:
                for apply in [False, True]:
                    response = client.post('/api/search', json={'query': 'pyhton', 'correct_typos': apply, 'top_k': 2})
                    self.assertEqual(response.status_code, 200)
                    value = response.json()
                    self.assertEqual(value['used_query'], 'python' if apply else 'pyhton')
                    self.assertEqual(value['corrected'], apply)
                    self.assertEqual(value['suggested_query'], None if apply else 'python')
                    self.assertEqual(len(value['results']), 2)
                    self.assertLessEqual(len(value['results'][0]['snippet']), 501)
                    self.assertIn('<img', value['results'][0]['title'])
                self.assertEqual(len(embeddings.calls), 2)
            self.assertEqual(store.scroll('test', limit=10, with_vectors=True), before)

    # ошибочные поля и слишком длинные запросы отклоняются до платного вызова
    def test_invalid_input_and_long_tokens_stop_before_paid_call(self) -> None:
        with closing(QdrantClient(location=':memory:')) as store:
            service, embeddings = make_service(store)
            variants: list[dict[str, object]] = [
                {'query': ''}, {'query': 'a' * 4001}, {'query': 'python', 'top_k': 0},
                {'query': 'python', 'top_k': '5'}, {'query': 'python', 'correct_typos': 'false'},
                {'query': 'python', 'api_key': 'must-not-be-accepted'},
            ]
            with TestClient(create_app(lambda: service)) as client:
                for body in variants:
                    with self.subTest(body=body):
                        self.assertEqual(client.post('/api/search', json=body).status_code, 422)
                for text in ['  ', 'a' * 1100]:
                    self.assertEqual(client.post('/api/search', json={'query': text}).status_code, 400)
                self.assertEqual(client.post('/api/suggest', json={'query': '  '}).status_code, 400)
                self.assertEqual(embeddings.calls, [])

    # параллельный поиск отклоняется, сбой модели не вызывает автоматическую повторную оплату
    def test_busy_search_is_rejected_and_model_failure_does_not_retry(self) -> None:
        with closing(QdrantClient(location=':memory:')) as store:
            service, embeddings = make_service(store)
            with TestClient(create_app(lambda: service)) as client:
                service.lock.acquire()
                try:
                    self.assertEqual(client.post('/api/search', json={'query': 'python'}).status_code, 429)
                    self.assertEqual(embeddings.calls, [])
                finally:
                    service.lock.release()
                embeddings.fail_on_call = 1
                response = client.post('/api/search', json={'query': 'python'})
                self.assertEqual(response.status_code, 502)
                self.assertEqual(len(embeddings.calls), 1)
                self.assertEqual(client.post('/api/search', json={'query': 'python'}).status_code, 200)
                self.assertEqual(len(embeddings.calls), 2)

    # исправление не применяется при изменении индекса или отсутствии словаря
    def test_changed_index_and_missing_dictionary_reject_correction(self) -> None:
        with closing(QdrantClient(location=':memory:')) as store:
            service, embeddings = make_service(store)
            with TestClient(create_app(lambda: service)) as client:
                store.delete('test', [3], wait=True)
                self.assertEqual(client.post('/api/search', json={'query': 'pyhton', 'correct_typos': True}).status_code, 400)
                self.assertEqual(embeddings.calls, [])
                service.dictionary = None
                suggestion = client.post('/api/suggest', json={'query': 'pyhton'}).json()
                self.assertFalse(suggestion['available'])
                self.assertEqual(client.post('/api/search', json={'query': 'pyhton', 'correct_typos': True}).status_code, 400)
                self.assertEqual(embeddings.calls, [])

    # запрос со сторонней страницы блокируется до обращения к модели
    def test_foreign_page_cannot_launch_paid_search(self) -> None:
        with closing(QdrantClient(location=':memory:')) as store:
            service, embeddings = make_service(store)
            with TestClient(create_app(lambda: service)) as client:
                response = client.post('/api/search', json={'query': 'python'},
                                       headers={'Origin': 'https://unrelated.example'})
                self.assertEqual(response.status_code, 403)
                self.assertEqual(client.get('/api/health', headers={'Host': 'unrelated.example'}).status_code, 400)
                self.assertEqual(embeddings.calls, [])

    # сначала ищем введённую фразу; выбранная подсказка требует второго явного запроса
    def test_suggested_query_search_requires_second_explicit_request(self) -> None:
        with closing(QdrantClient(location=':memory:')) as store:
            service, embeddings = make_service(store)
            with TestClient(create_app(lambda: service)) as client:
                original = client.post('/api/search', json={'query': 'pyhton'}).json()
                self.assertEqual(original['used_query'], 'pyhton')
                self.assertEqual(original['suggested_query'], 'python')
                self.assertFalse(original['corrected'])
                self.assertEqual(len(embeddings.calls), 1)
                selected = client.post('/api/search', json={
                    'query': original['suggested_query'], 'correct_typos': False,
                }).json()
                self.assertEqual(selected['used_query'], 'python')
                self.assertIsNone(selected['suggested_query'])
                self.assertFalse(selected['corrected'])
                self.assertEqual(len(embeddings.calls), 2)
                self.assertTrue(embeddings.calls[0][0].endswith('<query>pyhton'))
                self.assertTrue(embeddings.calls[1][0].endswith('<query>python'))

    # даже ложный вариант для редкого названия не меняет исходный поисковый запрос
    def test_rare_technology_name_is_only_suggested_and_never_silently_replaced(self) -> None:
        with closing(QdrantClient(location=':memory:')) as store:
            service, embeddings = make_service(store)
            words = {'стора': 3, 'спринга': 3}
            service.dictionary = SpellingDictionary(words, DictionaryInfo(
                'test', 3, 3, dictionary_hash(words), '2026-10-19T00:00:00+00:00',
            index_revision='unversioned'))
            with TestClient(create_app(lambda: service)) as client:
                result = client.post('/api/search', json={'query': 'Ктора без Спринга'}).json()
                self.assertEqual(result['used_query'], 'Ктора без Спринга')
                self.assertEqual(result['suggested_query'], 'Стора без Спринга')
                self.assertFalse(result['corrected'])
                self.assertEqual(len(embeddings.calls), 1)
                self.assertTrue(embeddings.calls[0][0].endswith('<query>Ктора без Спринга'))

    # при явном расхождении словаря и индекса доступен поиск исходной фразы без подсказок
    def test_changed_point_count_disables_suggestions_for_original_search(self) -> None:
        with closing(QdrantClient(location=':memory:')) as store:
            service, embeddings = make_service(store)
            with TestClient(create_app(lambda: service)) as client:
                store.delete('test', [3], wait=True)
                result = client.post('/api/search', json={'query': 'pyhton'}).json()
                self.assertEqual(result['used_query'], 'pyhton')
                self.assertIsNone(result['suggested_query'])
                self.assertEqual(len(embeddings.calls), 1)

    # неоднозначные варианты показываются пользователю без изменения поисковой фразы
    def test_ambiguous_words_are_returned_as_suggestions_and_not_applied(self) -> None:
        with closing(QdrantClient(location=':memory:')) as store:
            service, embeddings = make_service(store)
            words = {'серверы': 3, 'сервера': 3}
            service.dictionary = SpellingDictionary(words, DictionaryInfo(
                'test', 3, 3, dictionary_hash(words), '2026-10-19T00:00:00+00:00',
            index_revision='unversioned'))
            with TestClient(create_app(lambda: service)) as client:
                response = client.post('/api/search', json={'query': 'серверо', 'correct_typos': True}).json()
                self.assertEqual(response['used_query'], 'серверо')
                self.assertFalse(response['corrected'])
                self.assertEqual(set(response['alternatives']['серверо']), {'серверы', 'сервера'})
                self.assertEqual(len(embeddings.calls), 1)


if __name__ == '__main__':
    unittest.main()
