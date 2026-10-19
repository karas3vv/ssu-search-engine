"""проверяем общие пакеты, ограничение параллельности и продолжение без платного api."""
from __future__ import annotations

import unittest
from collections.abc import Iterator
from contextlib import closing
from threading import Event, Lock, get_ident
from typing import cast
from unittest.mock import patch

from qdrant_client import QdrantClient, models

from habr_search.embeddings import EmbeddingAPIError
from habr_search.index_version import index_revision
from habr_search.ingestion.chunking import ChunkingConfig, chunk_article
from habr_search.ingestion.index import ensure_collection, index_chunks
from habr_search.ingestion.pipeline import IndexProgress, index_articles
from habr_search.ingestion.prepare import PreparedArticle
from tests.test_index import CharacterTokenizer, FakeEmbeddings, article


class ConcurrentBackend:
    """подставной клиент ждёт соседний запрос и записывает порядок завершения под блокировкой."""
    def __init__(self, first_started: Event, second_finished: Event, lock: Lock,
                 completed: list[str], *, first: bool, fail: bool = False) -> None:
        """разделяем события между двумя разными клиентами без общей http-сессии."""
        self.first_started: Event = first_started
        self.second_finished: Event = second_finished
        self.lock: Lock = lock
        self.completed: list[str] = completed
        self.first: bool = first
        self.fail: bool = fail
        self.calls: int = 0
        self.thread_id: int | None = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        """первый запрос завершается после второго; ошибка не отменяет ответ соседнего клиента."""
        self.calls += 1
        self.thread_id = get_ident()
        if self.first:
            self.first_started.set()
            if not self.second_finished.wait(5):
                raise AssertionError('второй запрос не выполнялся параллельно')
        else:
            if not self.first_started.wait(5):
                raise AssertionError('первый запрос не стартовал')
        with self.lock:
            self.completed.append('first' if self.first else 'second')
        if not self.first and not self.fail:
            self.second_finished.set()
        if self.fail:
            raise EmbeddingAPIError('подставная ошибка без повторной отправки')
        return [[float(len(text)), 1.0, 1.0] for text in texts]


class PipelineTests(unittest.TestCase):
    """сверяем содержимое qdrant, порядок очистки и сохранение оплаченных результатов после сбоя."""
    def test_cross_article_batches_skip_existing_and_do_not_change_ids(self) -> None:
        """короткие статьи наполняют общий пакет; смена размера пакета не требует переиндексации."""
        docs = [article(f'короткий текст {i}', i) for i in range(1, 11)]
        config = ChunkingConfig(128, 12)
        backend = FakeEmbeddings()
        with closing(QdrantClient(':memory:')) as store:
            ensure_collection(store, 'test', 3)
            first_chunks = list(chunk_article(docs[0], CharacterTokenizer(), config))
            index_chunks(first_chunks, docs[0].text_hash, store, 'test', FakeEmbeddings(), vector_size=3)
            result = index_articles(docs, CharacterTokenizer(), config, store, 'test', [backend], 4, 3)
            self.assertEqual([len(batch) for batch in backend.calls], [4, 4, 1])
            self.assertEqual((result.articles, result.saved, result.skipped, result.batches), (10, 9, 1, 3))
            records, _ = store.scroll('test', limit=100, with_payload=True, with_vectors=True)
            expected = {chunk.id for doc in docs for chunk in chunk_article(doc, CharacterTokenizer(), config)}
            self.assertEqual({str(record.id) for record in records}, expected)
            revision = index_revision(store, 'test')
            repeated = FakeEmbeddings()
            result = index_articles(docs, CharacterTokenizer(), config, store, 'test', [repeated], 8, 3)
            self.assertEqual((result.saved, result.skipped), (0, 10))
            self.assertEqual(repeated.calls, [])
            self.assertEqual(index_revision(store, 'test'), revision)
            self.assertEqual(store.scroll('test', limit=100, with_payload=True, with_vectors=True)[0], records)

    def test_concurrent_out_of_order_responses_write_store_in_coordinator_thread(self) -> None:
        """запросы перекрываются, а qdrant и уведомления обслуживает исходный поток."""
        started, finished, lock = Event(), Event(), Lock()
        completed: list[str] = []
        # очередь начинает с последнего свободного клиента
        providers = [ConcurrentBackend(started, finished, lock, completed, first=False),
                     ConcurrentBackend(started, finished, lock, completed, first=True)]
        owner = get_ident()
        snapshots: list[IndexProgress] = []

        def notify(progress: IndexProgress) -> None:
            """проверяем поток обратного вызова вместе с его подтверждённым прогрессом."""
            self.assertEqual(get_ident(), owner)
            snapshots.append(progress)

        docs = [article('текст ' * i, i) for i in range(1, 5)]
        with closing(QdrantClient(':memory:')) as store:
            ensure_collection(store, 'test', 3)
            original_upsert = store.upsert

            def record_upsert(collection_name: str, points: list[models.PointStruct], wait: bool) -> models.UpdateResult:
                """проверяем, что даже пришедший раньше ответ записывается исходным потоком."""
                self.assertEqual(get_ident(), owner)
                return original_upsert(collection_name=collection_name, points=points, wait=wait)

            with patch.object(store, 'upsert', side_effect=record_upsert):
                result = index_articles(docs, CharacterTokenizer(), ChunkingConfig(128, 12),
                                        store, 'test', providers, 2, 3, on_progress=notify)
            self.assertEqual((result.articles, result.saved, result.batches), (4, 4, 2))
            self.assertEqual(completed, ['second', 'first'])
            self.assertEqual([provider.calls for provider in providers], [1, 1])
            self.assertTrue(all(provider.thread_id != owner for provider in providers))
            self.assertEqual(snapshots[-1].saved, 4)
            expected = {chunk.id: len(chunk.embedding_text) for doc in docs
                        for chunk in chunk_article(doc, CharacterTokenizer(), ChunkingConfig(128, 12))}
            records, _ = store.scroll('test', limit=100, with_vectors=True)
            for record in records:
                vector = cast(list[float], record.vector)
                self.assertAlmostEqual(vector[0] / vector[1], expected[str(record.id)], places=4)

    def test_failure_drains_other_response_and_resume_skips_confirmed_points(self) -> None:
        """после сбоя сохраняем успешный соседний пакет и не отправляем всю очередь заново."""
        started, finished, lock = Event(), Event(), Lock()
        completed: list[str] = []
        providers = [ConcurrentBackend(started, finished, lock, completed, first=False, fail=True),
                     ConcurrentBackend(started, finished, lock, completed, first=True)]
        docs = [article(f'текст {i}', i) for i in range(1, 21)]
        consumed: list[int] = []

        def source() -> Iterator[PreparedArticle]:
            """считаем чтения корпуса, чтобы обнаружить неограниченное накопление заданий."""
            for doc in docs:
                consumed.append(doc.id)
                yield doc

        config = ChunkingConfig(128, 12)
        with closing(QdrantClient(':memory:')) as store:
            ensure_collection(store, 'test', 3)
            with self.assertRaises(EmbeddingAPIError):
                index_articles(source(), CharacterTokenizer(), config, store, 'test', providers, 2, 3,
                               on_stopping=finished.set)
            self.assertEqual(sum(provider.calls for provider in providers), 2)
            self.assertLessEqual(len(consumed), 6)
            self.assertEqual(store.count('test', exact=True).count, 2)
            resumed = FakeEmbeddings()
            result = index_articles(docs, CharacterTokenizer(), config, store, 'test', [resumed], 4, 3)
            self.assertEqual((result.saved, result.skipped, result.articles), (18, 2, 20))
            saved_texts = [chunk.embedding_text for doc in docs[:2]
                           for chunk in chunk_article(doc, CharacterTokenizer(), config)]
            self.assertTrue(all(text not in saved_texts for batch in resumed.calls for text in batch))

    def test_stopping_saves_started_batch_and_does_not_submit_next(self) -> None:
        """остановка ждёт ответ уже запущенного запроса вместо потери готовых векторов."""
        stopped = Event()

        class StoppingBackend(FakeEmbeddings):
            """первый вызов просит остановить чтение новых данных."""
            def embed(self, texts: list[str]) -> list[list[float]]:
                """возвращаем успешный ответ после запроса остановки."""
                stopped.set()
                return super().embed(texts)

        backend = StoppingBackend()
        docs = [article(f'текст {i}', i) for i in range(1, 7)]
        with closing(QdrantClient(':memory:')) as store:
            ensure_collection(store, 'test', 3)
            with self.assertRaises(KeyboardInterrupt):
                index_articles(docs, CharacterTokenizer(), ChunkingConfig(128, 12), store, 'test',
                               [backend], 2, 3, stop_requested=stopped.is_set)
            self.assertEqual(len(backend.calls), 1)
            self.assertEqual(store.count('test', exact=True).count, 2)

    def test_failed_article_keeps_old_chunks_until_all_new_chunks_are_saved(self) -> None:
        """изменённая статья не теряет старую версию при частичном сбое новой."""
        config = ChunkingConfig(128, 12)
        old, changed = article('старый текст ' * 30), article('новый текст ' * 50)
        old_chunks = list(chunk_article(old, CharacterTokenizer(), config))
        new_chunks = list(chunk_article(changed, CharacterTokenizer(), config))
        with closing(QdrantClient(':memory:')) as store:
            ensure_collection(store, 'test', 3)
            index_chunks(old_chunks, old.text_hash, store, 'test', FakeEmbeddings(), 2, 3)
            with self.assertRaises(EmbeddingAPIError):
                index_articles([changed], CharacterTokenizer(), config, store, 'test', [FakeEmbeddings(2)], 2, 3)
            stored, _ = store.scroll('test', limit=100)
            self.assertTrue({chunk.id for chunk in old_chunks}.issubset({str(point.id) for point in stored}))
            index_articles([changed], CharacterTokenizer(), config, store, 'test', [FakeEmbeddings()], 2, 3)
            stored, _ = store.scroll('test', limit=100)
            self.assertEqual({str(point.id) for point in stored}, {chunk.id for chunk in new_chunks})


if __name__ == '__main__':
    unittest.main()
