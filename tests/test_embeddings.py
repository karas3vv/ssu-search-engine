"""проверяем клиент на подставных ответах, без платных запросов к серверу."""
from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import requests

from embeddings import EmbeddingAPIError, EmbeddingConfig, EmbeddingsClient, parse_vectors


# собираем подставной http-ответ requests с заданным содержимым и статусом
def response(payload: object, status: int = 200) -> requests.Response:
    result = requests.Response()
    result.status_code = status
    result._content = json.dumps(payload).encode('utf-8')
    _ = result.content  # имитируем полностью прочитанный ответ requests
    return result


# группа проверок модуля; каждый метод test_ описывает отдельный ожидаемый сценарий
class EmbeddingsTests(unittest.TestCase):
    # ответ api может быть перемешан, поэтому порядок восстанавливается по index
    def test_response_order_uses_indices(self) -> None:
        payload = {'data': [
            {'index': 1, 'embedding': [3, 4]},
            {'index': 0, 'embedding': [1, 2]},
        ]}
        self.assertEqual(parse_vectors(payload, 2), [[1.0, 2.0], [3.0, 4.0]])

    # пустые, нулевые и нечисловые векторы не попадают в индекс
    def test_malformed_vectors_are_rejected(self) -> None:
        for vector in [[], [True], ['0.1'], [float('nan')], [float('inf')], [0, 0]]:
            with self.subTest(vector=vector):
                with self.assertRaises(EmbeddingAPIError):
                    parse_vectors({'data': [{'index': 0, 'embedding': vector}]}, 1)

    # повторные номера и разные размерности в одном ответе считаются ошибкой
    def test_duplicate_indices_and_mixed_dimensions_are_rejected(self) -> None:
        for items in [
            [{'index': 0, 'embedding': [1]}, {'index': 0, 'embedding': [2]}],
            [{'index': 0, 'embedding': [1]}, {'index': 1, 'embedding': [2, 3]}],
        ]:
            with self.subTest(items=items):
                with self.assertRaises(EmbeddingAPIError):
                    parse_vectors({'data': items}, 2)

    # сверяем формат отправленного запроса и отсутствие скрытого изменения текста
    def test_request_matches_supplied_endpoint_and_preserves_input(self) -> None:
        config = EmbeddingConfig(api_key='test-key-not-real')
        client = EmbeddingsClient(config)
        try:
            with patch.object(client.session, 'post', return_value=response(
                {'data': [{'index': 0, 'embedding': [0.2, 0.3]}]}
            )) as post:
                self.assertEqual(client.embed(['тестовый текст']), [[0.2, 0.3]])
                self.assertEqual(post.call_args.kwargs['json'], {'model': 'parser', 'input': ['тестовый текст']})
                self.assertFalse(post.call_args.kwargs['allow_redirects'])
        finally:
            client.close()
        self.assertNotIn('test-key-not-real', repr(config))

    # ошибка авторизации не выводит ключ или исходное тело ответа
    def test_authentication_error_does_not_expose_response_body(self) -> None:
        client = EmbeddingsClient(EmbeddingConfig(api_key='test-key-not-real'))
        try:
            with patch.object(client.session, 'post', return_value=response({'error': 'test-key-not-real'}, 401)):
                with self.assertRaisesRegex(EmbeddingAPIError, 'http 401') as caught:
                    client.embed(['текст'])
                self.assertNotIn('test-key-not-real', str(caught.exception))
        finally:
            client.close()

    # пустые тексты отклоняются до сетевого запроса
    def test_empty_input_does_not_make_a_request(self) -> None:
        client = EmbeddingsClient(EmbeddingConfig(api_key='test-key-not-real'))
        try:
            with patch.object(client.session, 'post') as post:
                with self.assertRaises(ValueError):
                    client.embed(['   '])
                post.assert_not_called()
        finally:
            client.close()

    # различаем виды тайм-аута и проверяем, что запрос не повторяется автоматически
    def test_timeout_uses_configured_wait_without_automatic_retry(self) -> None:
        config = EmbeddingConfig(api_key='test-key-not-real', timeout_seconds=420.0)
        client = EmbeddingsClient(config)
        try:
            cases: list[tuple[requests.Timeout, str]] = [
                (requests.ConnectTimeout('private details'), 'подключения к MWS \\(10 с\\)'),
                (requests.ReadTimeout('private details'), 'данных ответа MWS \\(420 с\\)'),
                (requests.Timeout('private details'), 'ожидание ответа: 420 с'),
            ]
            for error, message in cases:
                with (self.subTest(error=type(error).__name__),
                      patch.object(client.session, 'post', side_effect=error) as post):
                    with self.assertRaisesRegex(EmbeddingAPIError, message) as caught:
                        client.embed(['текст'])
                    self.assertEqual(post.call_count, 1)
                    self.assertEqual(post.call_args.kwargs['timeout'], (10, 420.0))
                    self.assertNotIn('private details', str(caught.exception))
                    self.assertNotIn('test-key-not-real', str(caught.exception))
        finally:
            client.close()


# при прямом запуске выполняем тесты этого файла
if __name__ == '__main__':
    unittest.main()
