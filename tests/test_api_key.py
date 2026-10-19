"""проверяем общий ввод ключа без настоящих секретов."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from habr_search.embeddings import read_api_key


# группа проверок модуля; каждый метод test_ описывает отдельный ожидаемый сценарий
class ApiKeyTests(unittest.TestCase):
    # ключ из окружения используется без запроса скрытого ввода
    def test_environment_key_does_not_prompt(self) -> None:
        with patch.dict('os.environ', {'MWS_API_KEY': ' test-key-not-real '}, clear=True):
            with patch('habr_search.embeddings.getpass.getpass') as prompt:
                self.assertEqual(read_api_key(), 'test-key-not-real')
                prompt.assert_not_called()

    # пустой ввод не позволяет начать запрос к модели
    def test_empty_key_is_rejected(self) -> None:
        with patch.dict('os.environ', {}, clear=True):
            with patch('habr_search.embeddings.sys.stdin.isatty', return_value=True):
                with patch('habr_search.embeddings.getpass.getpass', return_value=' '):
                    with self.assertRaises(ValueError):
                        read_api_key()

    # без терминала и ключа команда завершается вместо ожидания ввода
    def test_noninteractive_missing_key_fails_without_prompt(self) -> None:
        with patch.dict('os.environ', {}, clear=True):
            with patch('habr_search.embeddings.sys.stdin.isatty', return_value=False):
                with patch('habr_search.embeddings.getpass.getpass') as prompt:
                    with self.assertRaises(ValueError):
                        read_api_key()
                    prompt.assert_not_called()


# при прямом запуске выполняем тесты этого файла
if __name__ == '__main__':
    unittest.main()
