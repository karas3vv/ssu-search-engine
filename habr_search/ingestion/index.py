"""индексируем фрагменты через mws и сохраняем их в qdrant."""
# поток обработки: sqlite → подготовка → фрагменты → mws → qdrant
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
from contextlib import ExitStack, closing
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Protocol

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from habr_search.ingestion.chunking import (
    CHUNKING_VERSION, DEFAULT_TOKENIZER, MODEL_ID, MODEL_REVISION,
    ArticleChunk, ChunkingConfig, chunk_article, load_tokenizer,
)
from habr_search.embeddings import (
    DEFAULT_MODEL, DEFAULT_URL, EmbeddingConfig, EmbeddingsClient, read_api_key,
)
from habr_search.ingestion.prepare import DEFAULT_SOURCE, PreparationStats, iter_prepared_articles
from habr_search.vector_store import connect_store, resolve_qdrant_config

VECTOR_SIZE: int = 3584  # размерность, которую вернул проверенный деплоймент parser


# интерфейс позволяет проверять запись в qdrant без настоящих запросов в mws
class EmbeddingProvider(Protocol):
    # получаем по одному вектору на каждую переданную строку
    def embed(self, texts: list[str]) -> list[list[float]]: ...


# считаем новые и уже существовавшие фрагменты одной операции индексации
@dataclass
class IndexStats:
    saved: int = 0  # новые точки, запись которых подтверждена qdrant
    skipped: int = 0  # точки, которые уже были сохранены раньше


# вычисляем имя коллекции из настроек модели и разбиения
def collection_name(url: str, deployment: str, config: ChunkingConfig) -> str:
    # другой деплоймент или настройки разбиения получают отдельную коллекцию
    # настройки входят в отпечаток, чтобы случайно не смешать несовместимые векторы
    settings = {
        'url': url, 'deployment': deployment, 'model': MODEL_ID,
        'revision': MODEL_REVISION, 'chunking_version': CHUNKING_VERSION,
        'max_tokens': config.max_tokens, 'overlap_tokens': config.overlap_tokens,
        'vector_size': VECTOR_SIZE,
    }
    signature = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]
    return f'habr_gemma2_{signature}'


# создаём подходящую коллекцию либо проверяем совместимость существующей
def ensure_collection(
    store: QdrantClient, name: str, vector_size: int = VECTOR_SIZE,
    *, create_article_index: bool = False,
) -> None:
    if not store.collection_exists(name):
        store.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )
    info = store.get_collection(name)
    vectors = info.config.params.vectors
    if (not isinstance(vectors, models.VectorParams) or vectors.size != vector_size
            or vectors.distance != models.Distance.COSINE):
        raise ValueError('существующая коллекция имеет другую размерность или метрику')
    # сервер использует индекс поля для фильтрации и удаления старых фрагментов статьи
    if create_article_index and 'article_id' not in info.payload_schema:
        store.create_payload_index(name, 'article_id', models.PayloadSchemaType.INTEGER, wait=True)


def index_chunks(
    chunks: list[ArticleChunk], text_hash: str, store: QdrantClient,
    name: str, embeddings: EmbeddingProvider, batch_size: int = 4,
    vector_size: int = VECTOR_SIZE,
) -> IndexStats:
    """сохраняем одну статью пакетами; повторный запуск пропускает готовые точки."""
    if batch_size < 1 or not chunks:
        raise ValueError('нужны фрагменты статьи и положительный размер пакета')
    article_id = chunks[0].article_id
    if any(chunk.article_id != article_id for chunk in chunks):
        raise ValueError('один вызов должен содержать фрагменты только одной статьи')
    stats = IndexStats()
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        # qdrant служит журналом прогресса: отдельная копия корпуса не нужна
        records = store.retrieve(
            name, ids=[chunk.id for chunk in batch], with_payload=False, with_vectors=False,
        )
        # уже записанные точки можно пропустить без повторной оплаты вычисления векторов
        existing = {str(record.id) for record in records}
        pending = [chunk for chunk in batch if chunk.id not in existing]
        stats.skipped += len(batch) - len(pending)
        if not pending:
            continue
        vectors = embeddings.embed([chunk.embedding_text for chunk in pending])
        if len(vectors) != len(pending) or any(len(vector) != vector_size for vector in vectors):
            raise ValueError(f'ожидались векторы размерности {vector_size}; проверьте модель деплоймента')
        points = [
            models.PointStruct(
                id=chunk.id, vector=vector,
                payload={
                    'article_id': chunk.article_id, 'chunk_index': chunk.chunk_index,
                    'title': chunk.title, 'url': chunk.url, 'text': chunk.text,
                    'hubs': list(chunk.hubs), 'published_at': chunk.published_at,
                    'start_char': chunk.start_char, 'end_char': chunk.end_char,
                    'token_count': chunk.token_count, 'text_hash': text_hash,
                    'model': MODEL_ID, 'model_revision': MODEL_REVISION,
                    'chunking_version': CHUNKING_VERSION,
                },
            )
            for chunk, vector in zip(pending, vectors)
        ]
        # ждём подтверждения записи перед переходом к следующему пакету
        store.upsert(collection_name=name, points=points, wait=True)
        stats.saved += len(points)
    # если текст статьи изменился, убираем её старые точки только после успешной записи новых
    store.delete(
        collection_name=name,
        points_selector=models.FilterSelector(filter=models.Filter(
            must=[models.FieldCondition(key='article_id', match=models.MatchValue(value=article_id))],
            must_not=[models.HasIdCondition(has_id=[chunk.id for chunk in chunks])],
        )),
        wait=True,
    )
    return stats


# настройки чтения корпуса, разбиения, отправки пакетов и подключения к qdrant
class CliArgs(argparse.Namespace):
    source: Path
    tokenizer_file: Path
    qdrant_path: Path | None
    qdrant_url: str | None
    limit: int
    batch_size: int
    max_tokens: int
    overlap_tokens: int
    timeout_seconds: float  # ожидание данных ответа mws для одного пакета
    dry_run: bool


# обрабатываем статьи по очереди, отправляем новые фрагменты в модель и сохраняем результат
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--tokenizer-file', type=Path, default=DEFAULT_TOKENIZER)
    location = parser.add_mutually_exclusive_group()
    location.add_argument('--qdrant-path', type=Path, help='папка встроенного режима')
    location.add_argument('--qdrant-url', help='адрес сервера, например http://127.0.0.1:6333')
    parser.add_argument('--limit', type=int, default=5, help='общий лимит статей, включая уже обработанные')
    parser.add_argument('--batch-size', type=int, default=4, help='число фрагментов в одном запросе mws')
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--overlap-tokens', type=int, default=128)
    parser.add_argument('--timeout', '--timeout-seconds', dest='timeout_seconds', type=float, default=300.0,
                        help='ожидание данных ответа MWS в секундах, по умолчанию 300')
    parser.add_argument('--dry-run', action='store_true', help='посчитать фрагменты без mws и без записи индекса')
    args: CliArgs = parser.parse_args(namespace=CliArgs())
    if args.limit < 1 or args.batch_size < 1:
        parser.error('limit и batch-size должны быть положительными')
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        parser.error('timeout должен быть положительным конечным числом')
    if not args.source.is_file():
        parser.error(f'исходная база не найдена: {args.source}')
    try:
        config = ChunkingConfig(args.max_tokens, args.overlap_tokens)
        store_config = resolve_qdrant_config(args.qdrant_url, args.qdrant_path)
        tokenizer = load_tokenizer(args.tokenizer_file)
        url = os.environ.get('MWS_EMBEDDINGS_URL', DEFAULT_URL)
        deployment = os.environ.get('MWS_EMBEDDINGS_MODEL', DEFAULT_MODEL)
        name = collection_name(url, deployment, config)
        prepared_stats = PreparationStats()
        total = IndexStats()
        article_count = 0
        chunk_count = 0
        with ExitStack() as stack:
            prepared = stack.enter_context(closing(iter_prepared_articles(args.source, prepared_stats)))
            store: QdrantClient | None = None
            embeddings: EmbeddingsClient | None = None
            if not args.dry_run:
                # проверяем qdrant до запроса ключа и платных обращений в mws
                store = connect_store(store_config)
                stack.callback(store.close)
                ensure_collection(store, name, create_article_index=store_config.url is not None)
                key = read_api_key()
                # ожидание не входит в имя коллекции и не меняет id ранее записанных фрагментов
                embedding_config = EmbeddingConfig(key, url, deployment, timeout_seconds=args.timeout_seconds)
                embeddings = stack.enter_context(closing(EmbeddingsClient(embedding_config)))
            for article in islice(prepared, args.limit):
                chunks = list(chunk_article(article, tokenizer, config))
                chunk_count += len(chunks)
                if store is not None and embeddings is not None:
                    stats = index_chunks(chunks, article.text_hash, store, name, embeddings, args.batch_size)
                    total.saved += stats.saved
                    total.skipped += stats.skipped
                article_count += 1
                print(f'статей: {article_count}; фрагментов: {chunk_count}; записано: {total.saved}; уже было: {total.skipped}', flush=True)
        print(f'коллекция: {name}')
        print(f'отклонено записей корпуса: {sum(prepared_stats.rejected.values())}')
        if args.dry_run:
            print('пробный запуск: api не вызывался, индекс не создавался')
        else:
            print(f'qdrant: {store_config.url or store_config.path}')
        return 0 if article_count and not prepared_stats.rejected else 1
    except KeyboardInterrupt:
        print('\nостановлено; сохранённые пакеты останутся в qdrant, повторите ту же команду для продолжения')
        return 130
    except UnexpectedResponse as exc:
        parser.exit(1, f'qdrant вернул http {exc.status_code}; проверьте адрес, ключ и настройки сервера\n')
    except ResponseHandlingException:
        parser.exit(1, 'не удалось связаться с qdrant; проверьте адрес и запущен ли контейнер\n')
    except (ValueError, RuntimeError, OSError, sqlite3.Error) as exc:
        parser.exit(1, f'индексация остановлена: {exc}\nсохранённые пакеты можно использовать при следующем запуске\n')


# при прямом запуске выполняем команду; при импорте этот блок пропускается
if __name__ == '__main__':
    raise SystemExit(main())
