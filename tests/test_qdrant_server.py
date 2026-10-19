"""необязательная проверка настоящего сервера: задайте qdrant_test_url для запуска."""
from __future__ import annotations

import hashlib
import os
import unittest
from contextlib import closing
from uuid import uuid4

from chunking import ChunkingConfig, chunk_article
from index import ensure_collection, index_chunks
from prepare import PreparedArticle
from search import search_articles, validate_collection
from vector_store import QdrantConfig, connect_store

TEST_URL: str | None = os.environ.get('QDRANT_TEST_URL')


# подставной токенизатор делает размер фрагментов предсказуемым в тесте
class TestTokenizer:
    # возвращаем границы символов вместо границ настоящих токенов
    def offsets(self, text: str) -> list[tuple[int, int]]:
        return [(i, i + 1) for i in range(len(text))]

    # имитируем учёт двух специальных токенов модели
    def count(self, text: str) -> int:
        return len(text) + 2


# подставной клиент выдаёт короткие векторы для проверки серверного qdrant
class TestEmbeddings:
    # начинаем подсчёт обращений к подставному клиенту
    def __init__(self) -> None:
        self.calls: int = 0

    # строим короткие векторы из длины текста без вызова удалённой модели
    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [[float(len(text)), 1.0, 2.0] for text in texts]


# группа проверок модуля; каждый метод test_ описывает отдельный ожидаемый сценарий
@unittest.skipUnless(TEST_URL, 'задайте QDRANT_TEST_URL для проверки настоящего сервера')
class QdrantServerTests(unittest.TestCase):
    # на настоящем сервере проверяем запись, продолжение, поиск и удаление устаревших точек
    def test_http_index_resume_search_and_stale_cleanup(self) -> None:
        name = f'test_habr_{uuid4().hex}'
        text = 'синтетический текст для проверки ' * 30
        doc = PreparedArticle(
            id=1, url='https://habr.com/ru/articles/1/', title='тест', text=text,
            author=None, published_at=None, hubs=(), tags=(),
            text_hash=hashlib.sha256(text.encode()).hexdigest(),
        )
        chunks = list(chunk_article(doc, TestTokenizer(), ChunkingConfig(128, 12)))
        config = QdrantConfig(url=TEST_URL, api_key=os.environ.get('QDRANT_TEST_API_KEY') or None)
        with closing(connect_store(config)) as store:
            try:
                ensure_collection(store, name, 3, create_article_index=True)
                self.assertIn('article_id', store.get_collection(name).payload_schema)
                backend = TestEmbeddings()
                saved = index_chunks(chunks, doc.text_hash, store, name, backend, 2, 3)
                self.assertEqual(saved.saved, len(chunks))
                previous_calls = backend.calls
                repeated = index_chunks(chunks, doc.text_hash, store, name, backend, 2, 3)
                self.assertEqual(repeated.skipped, len(chunks))
                self.assertEqual(backend.calls, previous_calls)
                validate_collection(store, name, 3)
                results = search_articles(store, name, [1.0, 1.0, 1.0], 3, 3)
                self.assertEqual([result.article_id for result in results], [1])
                self.assertEqual(results[0].title, doc.title)
                # имитируем новую версию статьи с меньшим числом фрагментов
                index_chunks(chunks[:1], doc.text_hash, store, name, backend, 2, 3)
                self.assertEqual(store.count(name, exact=True).count, 1)
            finally:
                if store.collection_exists(name):
                    store.delete_collection(name)


# при прямом запуске выполняем тесты этого файла
if __name__ == '__main__':
    unittest.main()
