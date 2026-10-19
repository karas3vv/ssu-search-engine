"""ищем по словам в тех же фрагментах qdrant, без обращений к модели."""
# поток поиска: тексты qdrant → словарный индекс в памяти → оценки bm25 → разные статьи
from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.conversions import common_types as qdrant_types
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from chunking import ChunkingConfig
from embeddings import DEFAULT_MODEL, DEFAULT_URL
from index import collection_name
from search import SearchResult, validate_collection
from vector_store import connect_store, resolve_qdrant_config

# выделяем слова и числа, сохраняя окончания технических имён вроде c++ и c#
TOKEN_PATTERN: re.Pattern[str] = re.compile(r'[^\W_]+(?:\+\+|#)?', re.UNICODE)
BM25_VERSION: str = '1'  # меняем при изменении формулы или нормализации


# текст и метаданные точки qdrant, достаточные для поиска по словам
@dataclass(frozen=True)
class IndexedChunk:
    point_id: str  # id отдельного фрагмента в qdrant
    article_id: int  # id исходной статьи для группировки результатов
    title: str  # заголовок, добавляемый к тексту фрагмента
    url: str  # ссылка на оригинальную статью
    text: str  # текст фрагмента из поискового индекса


# параметры насыщения частоты слова и поправки на длину фрагмента
@dataclass(frozen=True)
class BM25Config:
    k1: float = 1.2  # насыщение частоты: повторы слова дают всё меньшую прибавку
    b: float = 0.75  # поправка на длину фрагмента

    # проверяем границы параметров, чтобы формула давала конечные значения
    def __post_init__(self) -> None:
        if not math.isfinite(self.k1) or self.k1 < 0:
            raise ValueError('k1 должен быть неотрицательным конечным числом')
        if not math.isfinite(self.b) or not 0 <= self.b <= 1:
            raise ValueError('b должен быть конечным числом от 0 до 1')


# приводим написание к общему виду и выделяем слова и технические названия
def tokenize(text: str) -> list[str]:
    # одна нормализация применяется к запросам и документам; словари пока не используются
    normalized = unicodedata.normalize('NFKC', text).casefold().replace('ё', 'е')
    return TOKEN_PATTERN.findall(normalized)


# читаем все страницы точек qdrant с текстами, не скачивая векторы
def load_indexed_chunks(store: QdrantClient, name: str, batch_size: int = 256) -> list[IndexedChunk]:
    if batch_size < 1:
        raise ValueError('размер страницы должен быть положительным')
    # проверяем число точек до и после чтения всех страниц
    initial_count = store.count(name, exact=True).count
    if not initial_count:
        raise ValueError('коллекция пуста')
    chunks: list[IndexedChunk] = []
    offset: qdrant_types.PointId | None = None
    seen: set[str] = set()
    while True:
        records, offset = store.scroll(
            name, limit=batch_size, offset=offset,
            with_payload=['article_id', 'title', 'url', 'text'], with_vectors=False,
        )
        for record in records:
            payload = record.payload or {}
            article_id: object = payload.get('article_id')
            title: object = payload.get('title')
            url: object = payload.get('url')
            text: object = payload.get('text')
            if (type(article_id) is not int or article_id < 1 or not isinstance(title, str)
                    or not isinstance(url, str) or not isinstance(text, str) or not text.strip()):
                raise ValueError(f'точка {record.id}: отсутствуют метаданные или текст статьи')
            point_id = str(record.id)
            if point_id in seen:
                raise ValueError('при чтении повторилась точка; проверьте, что индекс не изменяется')
            seen.add(point_id)
            chunks.append(IndexedChunk(point_id, article_id, title, url, text))
        if offset is None:
            break
    if len(chunks) != initial_count or store.count(name, exact=True).count != initial_count:
        raise ValueError('состав коллекции изменился во время чтения; повторите на неизменном индексе')
    return chunks


# фиксируем id, тексты и метаданные прочитанного корпуса независимо от порядка чтения
def corpus_hash(chunks: Sequence[IndexedChunk]) -> str:
    digest = hashlib.sha256()
    for chunk in sorted(chunks, key=lambda item: item.point_id):
        for value in [chunk.point_id, str(chunk.article_id), chunk.title, chunk.url, chunk.text]:
            encoded = value.encode('utf-8')
            # длина перед содержимым разделяет поля без неоднозначного склеивания
            digest.update(len(encoded).to_bytes(8, 'big'))
            digest.update(encoded)
    return digest.hexdigest()


# индекс в памяти связывает слова с фрагментами и частотами их появления
class BM25Index:
    # строим списки вхождений слов и считаем длины фрагментов
    def __init__(self, chunks: Sequence[IndexedChunk], config: BM25Config = BM25Config()) -> None:
        if not chunks:
            raise ValueError('для bm25 нужен непустой набор фрагментов')
        self.chunks: tuple[IndexedChunk, ...] = tuple(chunks)
        self.config: BM25Config = config
        self.lengths: list[int] = []
        # для каждого слова храним номера фрагментов и число повторов в них
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        ids: set[str] = set()
        for position, chunk in enumerate(self.chunks):
            if chunk.point_id in ids:
                raise ValueError('id фрагмента повторяется')
            ids.add(chunk.point_id)
            # заголовок учитывается один раз на фрагмент, как во входе семантической модели
            frequencies = Counter(tokenize(chunk.title + '\n\n' + chunk.text))
            self.lengths.append(sum(frequencies.values()))
            for term, frequency in frequencies.items():
                self.postings[term].append((position, frequency))
        # средняя длина нужна, чтобы длинные фрагменты не выигрывали только из-за размера
        self.average_length: float = sum(self.lengths) / len(self.chunks)
        if self.average_length == 0:
            raise ValueError('в корпусе нет слов для поиска')

    # считаем bm25 для совпавших слов и оставляем лучший фрагмент каждой статьи
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if top_k < 1:
            raise ValueError('top-k должен быть положительным')
        if not query.strip():
            raise ValueError('поисковый запрос пуст')
        scores: dict[int, float] = defaultdict(float)
        # повтор одного слова в запросе не увеличивает его вес
        for term in sorted(set(tokenize(query))):
            matches = self.postings.get(term, [])
            if not matches:
                continue
            # редкое слово получает больший вес, чем слово почти во всех фрагментах
            idf = math.log1p((len(self.chunks) - len(matches) + 0.5) / (len(matches) + 0.5))
            for position, frequency in matches:
                # параметры k1 и b управляют насыщением повторов и поправкой на длину
                norm = self.config.k1 * (1 - self.config.b + self.config.b * self.lengths[position] / self.average_length)
                scores[position] += idf * frequency * (self.config.k1 + 1) / (frequency + norm)
        # сортируем по убыванию оценки, а равенства разрешаем устойчивыми id
        ranked = sorted(scores, key=lambda position: (
            -scores[position], self.chunks[position].article_id, self.chunks[position].point_id,
        ))
        found: list[SearchResult] = []
        # после ранжирования оставляем каждой статье только её лучший фрагмент
        seen: set[int] = set()
        for position in ranked:
            chunk = self.chunks[position]
            if chunk.article_id in seen:
                continue
            seen.add(chunk.article_id)
            found.append(SearchResult(chunk.article_id, chunk.title, chunk.url, chunk.text, scores[position]))
            if len(found) == top_k:
                break
        return found


# настройки поиска по словам и подключения к существующей коллекции
class CliArgs(argparse.Namespace):
    query: str
    top_k: int
    qdrant_url: str | None
    qdrant_path: Path | None
    max_tokens: int
    overlap_tokens: int
    k1: float
    b: float


# загружаем фрагменты в память и показываем выдачу bm25 для одного запроса
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--query', required=True)
    parser.add_argument('--top-k', type=int, default=5)
    location = parser.add_mutually_exclusive_group()
    location.add_argument('--qdrant-url')
    location.add_argument('--qdrant-path', type=Path)
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--overlap-tokens', type=int, default=128)
    parser.add_argument('--k1', type=float, default=1.2)
    parser.add_argument('--b', type=float, default=0.75)
    args: CliArgs = parser.parse_args(namespace=CliArgs())
    if not args.query.strip() or args.top_k < 1:
        parser.error('нужны непустой запрос и положительный top-k')
    try:
        config = BM25Config(args.k1, args.b)
        chunking = ChunkingConfig(args.max_tokens, args.overlap_tokens)
        name = collection_name(os.environ.get('MWS_EMBEDDINGS_URL', DEFAULT_URL),
                               os.environ.get('MWS_EMBEDDINGS_MODEL', DEFAULT_MODEL), chunking)
        connection = resolve_qdrant_config(args.qdrant_url, args.qdrant_path)
        if connection.path is not None and not connection.path.is_dir():
            raise ValueError('локальное хранилище не найдено; задайте --qdrant-url')
        with closing(connect_store(connection)) as store:
            validate_collection(store, name)
            chunks = load_indexed_chunks(store, name)
        found = BM25Index(chunks, config).search(args.query, args.top_k)
        print(f'коллекция: {name}; найдено статей: {len(found)}')
        for number, result in enumerate(found, 1):
            print(f'\n{number}. {result.title}\nоценка bm25: {result.score:.4f}\n{result.url}')
            snippet = ' '.join(result.text.split())
            print(snippet[:400] + ('…' if len(snippet) > 400 else ''))
        return 0
    except KeyboardInterrupt:
        print('\nпоиск остановлен')
        return 130
    except UnexpectedResponse as exc:
        parser.exit(1, f'qdrant вернул http {exc.status_code}\n')
    except ResponseHandlingException:
        parser.exit(1, 'не удалось связаться с qdrant\n')
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f'поиск bm25 не выполнен: {exc}\n')


# при прямом запуске выполняем команду; при импорте этот блок пропускается
if __name__ == '__main__':
    raise SystemExit(main())
