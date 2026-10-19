"""проверяем http, исправления и ошибки на синтетическом индексе без mws."""
from __future__ import annotations

import unittest
from contextlib import closing

from fastapi.testclient import TestClient
from qdrant_client import QdrantClient, models

from index import VECTOR_SIZE, ensure_collection
from spelling import DictionaryInfo, SpellingDictionary, dictionary_hash
from test_evaluate import CharacterTokenizer, FakeEmbeddings
from web import SearchService, create_app


def make_service(store: QdrantClient) -> tuple[SearchService, FakeEmbeddings]:
    ensure_collection(store, 'test')
    vector = [1.0] + [0.0] * (VECTOR_SIZE - 1)
    store.upsert('test', [models.PointStruct(id=i, vector=vector, payload={
        'article_id': i, 'title': '<img src=x onerror=alert(1)> Python',
        'url': f'https://habr.com/ru/articles/{i}/', 'text': 'текст\n' * 200,
    }) for i in range(1, 4)])
    words = {'python': 3}
    dictionary = SpellingDictionary(words, DictionaryInfo('test', 3, 3, dictionary_hash(words),
                                                          '2026-10-19T00:00:00+00:00'))
    embeddings = FakeEmbeddings(VECTOR_SIZE)
    service = SearchService(store, 'test', CharacterTokenizer(), embeddings, dictionary)
    service.resources.callback(embeddings.close)
    return service, embeddings


class WebTests(unittest.TestCase):
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
                    self.assertEqual(len(value['results']), 2)
                    self.assertLessEqual(len(value['results'][0]['snippet']), 501)
                    self.assertIn('<img', value['results'][0]['title'])
                self.assertEqual(len(embeddings.calls), 2)
            self.assertEqual(store.scroll('test', limit=10, with_vectors=True), before)

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

    def test_foreign_page_cannot_launch_paid_search(self) -> None:
        with closing(QdrantClient(location=':memory:')) as store:
            service, embeddings = make_service(store)
            with TestClient(create_app(lambda: service)) as client:
                response = client.post('/api/search', json={'query': 'python'},
                                       headers={'Origin': 'https://unrelated.example'})
                self.assertEqual(response.status_code, 403)
                self.assertEqual(client.get('/api/health', headers={'Host': 'unrelated.example'}).status_code, 400)
                self.assertEqual(embeddings.calls, [])

    # интерфейс применяет исправление явно, а повторный запрос может отменить его один раз
    def test_corrected_search_and_original_retry_send_exactly_one_query_each(self) -> None:
        with closing(QdrantClient(location=':memory:')) as store:
            service, embeddings = make_service(store)
            with TestClient(create_app(lambda: service)) as client:
                corrected = client.post('/api/search', json={'query': 'pyhton', 'correct_typos': True}).json()
                self.assertEqual(corrected['used_query'], 'python')
                self.assertEqual(corrected['original_query'], 'pyhton')
                self.assertEqual(corrected['alternatives'], {})
                original = client.post('/api/search', json={
                    'query': corrected['original_query'], 'correct_typos': False,
                }).json()
                self.assertEqual(original['used_query'], 'pyhton')
                self.assertFalse(original['corrected'])
                self.assertEqual(len(embeddings.calls), 2)
                self.assertTrue(embeddings.calls[0][0].endswith('<query>python'))
                self.assertTrue(embeddings.calls[1][0].endswith('<query>pyhton'))

    def test_ambiguous_words_are_returned_as_suggestions_and_not_applied(self) -> None:
        with closing(QdrantClient(location=':memory:')) as store:
            service, embeddings = make_service(store)
            words = {'серверы': 3, 'сервера': 3}
            service.dictionary = SpellingDictionary(words, DictionaryInfo(
                'test', 3, 3, dictionary_hash(words), '2026-10-19T00:00:00+00:00',
            ))
            with TestClient(create_app(lambda: service)) as client:
                response = client.post('/api/search', json={'query': 'серверо', 'correct_typos': True}).json()
                self.assertEqual(response['used_query'], 'серверо')
                self.assertFalse(response['corrected'])
                self.assertEqual(set(response['alternatives']['серверо']), {'серверы', 'сервера'})
                self.assertEqual(len(embeddings.calls), 1)


if __name__ == '__main__':
    unittest.main()
