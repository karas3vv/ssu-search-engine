"""проверяем сохранение и продолжение на настоящем локальном qdrant без api mws."""
from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from qdrant_client import QdrantClient

from chunking import ChunkingConfig, chunk_article
from embeddings import EmbeddingAPIError
from index import collection_name, ensure_collection, index_chunks, main
from prepare import PreparedArticle


# простой токенизатор для тестов считает символы вместо токенов настоящей модели
class CharacterTokenizer:
    # считаем каждый символ отдельным токеном с известными границами
    def offsets(self, text: str) -> list[tuple[int, int]]:
        return [(i, i + 1) for i in range(len(text))]

    # прибавляем два условных специальных токена к числу символов
    def count(self, text: str) -> int:
        return len(text) + 2


# подставной клиент хранит историю вызовов и позволяет имитировать сбой модели
class FakeEmbeddings:
    # сохраняем параметры подставного клиента и готовим историю вызовов
    def __init__(self, fail_on_call: int | None = None) -> None:
        self.calls: list[list[str]] = []
        self.fail_on_call: int | None = fail_on_call

    # возвращаем подставные векторы и при заданной настройке имитируем ошибку
    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        if len(self.calls) == self.fail_on_call:
            raise EmbeddingAPIError('подставной сбой api')
        return [[float(len(text)), 1.0, 1.0] for text in texts]


# создаём подготовленную статью с известным текстом и отпечатком для теста
def article(text: str, article_id: int = 1) -> PreparedArticle:
    return PreparedArticle(
        id=article_id, url=f'https://habr.com/ru/articles/{article_id}/',
        title='тест', text=text, author=None, published_at=None, hubs=(), tags=(),
        text_hash=hashlib.sha256(text.encode()).hexdigest(),
    )


# группа проверок модуля; каждый метод test_ описывает отдельный ожидаемый сценарий
class IndexTests(unittest.TestCase):
    # сохранённые точки доступны после перезапуска и не отправляются в модель повторно
    def test_saved_points_survive_restart_and_are_not_embedded_twice(self) -> None:
        doc = article('фрагмент ' * 80)
        chunks = list(chunk_article(doc, CharacterTokenizer(), ChunkingConfig(128, 12)))
        backend = FakeEmbeddings()
        with tempfile.TemporaryDirectory() as folder:
            with closing(QdrantClient(path=folder)) as store:
                ensure_collection(store, 'test', 3)
                stats = index_chunks(chunks, doc.text_hash, store, 'test', backend, 2, 3)
                self.assertEqual(stats.saved, len(chunks))
            with closing(QdrantClient(path=folder)) as store:
                repeated = FakeEmbeddings()
                stats = index_chunks(chunks, doc.text_hash, store, 'test', repeated, 2, 3)
                self.assertEqual(stats.skipped, len(chunks))
                self.assertEqual(repeated.calls, [])
                self.assertEqual(store.count('test', exact=True).count, len(chunks))

    # после сбоя индексатор начинает с несохранённого пакета
    def test_partial_failure_resumes_without_resending_saved_batch(self) -> None:
        doc = article('длинная статья ' * 80)
        chunks = list(chunk_article(doc, CharacterTokenizer(), ChunkingConfig(128, 12)))
        with closing(QdrantClient(location=':memory:')) as store:
            ensure_collection(store, 'test', 3)
            with self.assertRaises(EmbeddingAPIError):
                index_chunks(chunks, doc.text_hash, store, 'test', FakeEmbeddings(2), 2, 3)
            self.assertEqual(store.count('test', exact=True).count, 2)
            resumed = FakeEmbeddings()
            stats = index_chunks(chunks, doc.text_hash, store, 'test', resumed, 2, 3)
            self.assertEqual(stats.skipped, 2)
            self.assertEqual(store.count('test', exact=True).count, len(chunks))
            self.assertEqual(resumed.calls[0], [chunk.embedding_text for chunk in chunks[2:4]])

    # обновление статьи удаляет её старые фрагменты и сохраняет остальные статьи
    def test_changed_article_replaces_its_stale_points_only(self) -> None:
        original = article('старый текст ' * 80)
        changed = article('новый короткий текст')
        other = article('другая статья', 2)
        config = ChunkingConfig(128, 12)
        with closing(QdrantClient(location=':memory:')) as store:
            ensure_collection(store, 'test', 3)
            for doc in [original, other, changed]:
                chunks = list(chunk_article(doc, CharacterTokenizer(), config))
                index_chunks(chunks, doc.text_hash, store, 'test', FakeEmbeddings(), 2, 3)
            records, _ = store.scroll('test', limit=100, with_payload=True)
            self.assertEqual(len(records), 2)
            hashes = {(record.payload or {}).get('text_hash') for record in records}
            self.assertEqual(hashes, {changed.text_hash, other.text_hash})

    # несовместимая коллекция отклоняется без изменения её размерности
    def test_existing_collection_dimension_is_not_overwritten(self) -> None:
        with closing(QdrantClient(location=':memory:')) as store:
            ensure_collection(store, 'test', 3)
            with self.assertRaises(ValueError):
                ensure_collection(store, 'test', 4)

    # смена деплоймента или разбиения выбирает другую коллекцию
    def test_collection_depends_on_deployment_and_chunk_settings(self) -> None:
        initial = collection_name('https://example.com/embeddings', 'parser', ChunkingConfig())
        self.assertEqual(initial, collection_name('https://example.com/embeddings', 'parser', ChunkingConfig()))
        self.assertNotEqual(initial, collection_name('https://example.com/embeddings', 'other', ChunkingConfig()))
        self.assertNotEqual(initial, collection_name('https://example.com/embeddings', 'parser', ChunkingConfig(512, 48)))

    # пробная индексация читает корпус без обращения к модели и хранилищу векторов
    def test_dry_run_does_not_request_key_or_open_qdrant(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'source.sqlite3'
            db = sqlite3.connect(source)
            db.execute('CREATE TABLE articles (id INTEGER PRIMARY KEY, payload TEXT)')
            db.execute('INSERT INTO articles VALUES (?,?)', (1, json.dumps({
                'id': 1, 'url': 'https://habr.com/ru/articles/1/', 'title': 'тест', 'text': 'текст',
            })))
            db.commit()
            db.close()
            before = source.read_bytes()
            argv = ['index.py', '--source', str(source), '--dry-run', '--limit', '1']
            with (patch('sys.argv', argv),
                  patch('index.load_tokenizer', return_value=CharacterTokenizer()),
                  patch('index.read_api_key', side_effect=AssertionError('ключ не нужен')),
                  patch('index.connect_store', side_effect=AssertionError('qdrant не нужен')),
                  redirect_stdout(io.StringIO())):
                self.assertEqual(main(), 0)
            self.assertEqual(before, source.read_bytes())


# при прямом запуске выполняем тесты этого файла
if __name__ == '__main__':
    unittest.main()
