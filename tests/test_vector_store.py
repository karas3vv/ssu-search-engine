"""проверяем выбор серверного и встроенного режима без сетевых запросов."""
from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from vector_store import QdrantConfig, connect_store, resolve_qdrant_config


class VectorStoreTests(unittest.TestCase):
    def test_explicit_local_path_overrides_environment_url(self) -> None:
        with patch.dict('os.environ', {'QDRANT_URL': 'https://example.com'}, clear=True):
            config = resolve_qdrant_config(None, Path('/tmp/qdrant-test'))
        self.assertIsNone(config.url)
        self.assertEqual(config.path, Path('/tmp/qdrant-test'))

    def test_environment_url_selects_server_and_hides_key(self) -> None:
        with patch.dict('os.environ', {'QDRANT_URL': 'https://example.com', 'QDRANT_API_KEY': 'fake-key-not-real'}, clear=True):
            config = resolve_qdrant_config(None, None)
        self.assertEqual(config.url, 'https://example.com')
        self.assertIsNone(config.path)
        self.assertNotIn('fake-key-not-real', repr(config))

    def test_server_factory_passes_connection_settings(self) -> None:
        config = QdrantConfig(url='https://example.com', api_key='fake-key-not-real')
        with patch('vector_store.QdrantClient') as factory:
            connect_store(config)
            factory.assert_called_once_with(url='https://example.com', api_key='fake-key-not-real', timeout=60)

    def test_local_factory_does_not_receive_server_credentials(self) -> None:
        with patch('vector_store.QdrantClient') as factory:
            connect_store(QdrantConfig(path=Path('/tmp/local-qdrant-test')))
            factory.assert_called_once_with(path='/tmp/local-qdrant-test')

    def test_conflicting_locations_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            resolve_qdrant_config('http://127.0.0.1:6333', Path('/tmp/qdrant-test'))

    def test_remote_key_requires_https(self) -> None:
        with self.assertRaises(ValueError):
            QdrantConfig(url='http://example.com', api_key='fake-key-not-real')
        self.assertEqual(QdrantConfig(url='http://127.0.0.1:6333').url, 'http://127.0.0.1:6333')


if __name__ == '__main__':
    unittest.main()
