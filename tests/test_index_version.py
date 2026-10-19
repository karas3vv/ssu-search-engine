"""проверяем версии при замене текста, сбоях записи и повторном запуске без изменений."""
from __future__ import annotations

import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from qdrant_client import QdrantClient

from habr_search.index_version import index_revision
from habr_search.ingestion.chunking import ChunkingConfig, chunk_article
from habr_search.ingestion.index import ensure_collection, index_chunks
from habr_search.retrieval.spellcheck import build_dictionary
from habr_search.retrieval.spelling import load_dictionary
from tests.test_index import CharacterTokenizer, FakeEmbeddings, article


# используем маленький индекс, чтобы проверить замену содержимого при неизменном числе точек
class IndexVersionTests(unittest.TestCase):
    # новый текст с одним фрагментом делает словарь старого текста несовместимым
    def test_equal_point_count_does_not_hide_text_replacement(self) -> None:
        with closing(QdrantClient(':memory:')) as store, tempfile.TemporaryDirectory() as folder:
            ensure_collection(store, 'test', 3)
            first = article('автоматизация документов')
            chunks = list(chunk_article(first, CharacterTokenizer(), ChunkingConfig(128, 0)))
            index_chunks(chunks, first.text_hash, store, 'test', FakeEmbeddings(), vector_size=3)
            path = Path(folder) / 'dictionary.sqlite3'
            build_dictionary(store, 'test', path)
            dictionary = load_dictionary(path)
            before = index_revision(store, 'test')
            second = article('контейнеризация приложений')
            chunks = list(chunk_article(second, CharacterTokenizer(), ChunkingConfig(128, 0)))
            index_chunks(chunks, second.text_hash, store, 'test', FakeEmbeddings(), vector_size=3)
            self.assertEqual(store.count('test', exact=True).count, 1)
            self.assertNotEqual(index_revision(store, 'test'), before)
            with self.assertRaisesRegex(ValueError, 'версии'):
                dictionary.validate_context('test', 1, index_revision(store, 'test'))

    # повтор готовой статьи не меняет версию и не отправляет тексты в модель
    def test_noop_keeps_revision_and_other_metadata(self) -> None:
        with closing(QdrantClient(':memory:')) as store:
            ensure_collection(store, 'test', 3)
            store.update_collection('test', metadata={'owner': 'учебный проект'})
            doc = article('текст статьи')
            chunks = list(chunk_article(doc, CharacterTokenizer(), ChunkingConfig(128, 0)))
            index_chunks(chunks, doc.text_hash, store, 'test', FakeEmbeddings(), vector_size=3)
            before = index_revision(store, 'test')
            backend = FakeEmbeddings()
            index_chunks(chunks, doc.text_hash, store, 'test', backend, vector_size=3)
            self.assertEqual(index_revision(store, 'test'), before)
            self.assertEqual(backend.calls, [])
            self.assertEqual((store.get_collection('test').config.metadata or {})['owner'], 'учебный проект')

    # при неопределённом результате записи прежний словарь также перестаёт считаться актуальным
    def test_write_failure_invalidates_previous_revision_before_upsert(self) -> None:
        with closing(QdrantClient(':memory:')) as store:
            ensure_collection(store, 'test', 3)
            before = index_revision(store, 'test')
            doc = article('новый текст')
            chunks = list(chunk_article(doc, CharacterTokenizer(), ChunkingConfig(128, 0)))
            with patch.object(store, 'upsert', side_effect=RuntimeError('сбой записи')):
                with self.assertRaisesRegex(RuntimeError, 'сбой'):
                    index_chunks(chunks, doc.text_hash, store, 'test', FakeEmbeddings(), vector_size=3)
            self.assertNotEqual(index_revision(store, 'test'), before)

    # изменение версии во время чтения словаря не публикует несогласованный файл
    def test_dictionary_build_detects_revision_change_with_same_count(self) -> None:
        with closing(QdrantClient(':memory:')) as store, tempfile.TemporaryDirectory() as folder:
            ensure_collection(store, 'test', 3)
            doc = article('текст статьи')
            chunks = list(chunk_article(doc, CharacterTokenizer(), ChunkingConfig(128, 0)))
            index_chunks(chunks, doc.text_hash, store, 'test', FakeEmbeddings(), vector_size=3)
            path = Path(folder) / 'dictionary.sqlite3'
            with patch('habr_search.retrieval.spellcheck.index_revision', side_effect=['before', 'after']):
                with self.assertRaisesRegex(ValueError, 'индекс изменился'):
                    build_dictionary(store, 'test', path)
            self.assertFalse(path.exists())


if __name__ == '__main__':
    unittest.main()
