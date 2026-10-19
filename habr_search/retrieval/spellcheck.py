"""строим словарь из qdrant, предлагаем исправления и готовим отдельный набор оценки."""
from __future__ import annotations

import argparse
import csv
import os
from collections import Counter, defaultdict
from contextlib import closing
from dataclasses import replace
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.conversions import common_types as qdrant_types
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from habr_search.ingestion.chunking import ChunkingConfig
from habr_search.embeddings import DEFAULT_MODEL, DEFAULT_URL
from habr_search.evaluation.evaluate import QUERY_COLUMNS, EvaluationQuery, load_queries
from habr_search.ingestion.index import collection_name
from habr_search.retrieval.spelling import (
    DEFAULT_DICTIONARY, DictionaryInfo, SpellingDictionary, SpellingResult,
    load_dictionary, save_dictionary, vocabulary_words,
)
from habr_search.vector_store import connect_store, resolve_qdrant_config
from habr_search.index_version import index_revision


def build_dictionary(store: QdrantClient, name: str, output: Path, batch_size: int = 256) -> DictionaryInfo:
    """читаем только тексты и заголовки, не скачиваем векторы и не изменяем qdrant."""
    if batch_size < 1:
        raise ValueError('размер пакета должен быть положительным')
    if output.exists():
        raise ValueError(f'словарь уже существует: {output}; укажите другой --dictionary')
    if not store.collection_exists(name):
        raise ValueError(f'коллекция {name} не найдена')
    expected_points = store.count(name, exact=True).count
    revision = index_revision(store, name)
    if expected_points == 0:
        raise ValueError('коллекция пуста; сначала выполните индексацию')
    article_words: dict[int, set[str]] = defaultdict(set)
    seen: set[str] = set()
    offset: qdrant_types.PointId | None = None
    while True:
        records, next_offset = store.scroll(
            name, offset=offset, limit=batch_size, with_payload=['article_id', 'title', 'text'],
            with_vectors=False,
        )
        for record in records:
            point_id = str(record.id)
            payload = record.payload or {}
            article_id: object = payload.get('article_id')
            title: object = payload.get('title')
            text: object = payload.get('text')
            if (type(article_id) is not int or article_id < 1 or not isinstance(title, str)
                    or not isinstance(text, str) or point_id in seen):
                raise ValueError('в коллекции повторяются точки или отсутствуют поля статьи')
            seen.add(point_id)
            article_words[article_id].update(vocabulary_words(title + '\n' + text))
        if next_offset is None:
            break
        if next_offset == offset or not records:
            raise ValueError('qdrant не продвинулся при чтении словаря')
        offset = next_offset
    if (len(seen) != expected_points or store.count(name, exact=True).count != expected_points
            or index_revision(store, name) != revision):
        raise ValueError('индекс изменился; не обновляйте его при построении словаря')
    frequencies: Counter[str] = Counter()
    for words in article_words.values():
        frequencies.update(words)
    return save_dictionary(output, frequencies, name, expected_points, len(article_words), revision)


def prepare_evaluation(dictionary: SpellingDictionary, source: Path, output: Path) -> int:
    """сохраняем исправленные запросы отдельно, не меняя исходную разметку и отчёты."""
    cases = load_queries(source)
    corrected: list[EvaluationQuery] = []
    changed = 0
    seen: set[str] = set()
    for case in cases:
        result = dictionary.suggest(case.query)
        if result.corrected.casefold() in seen:
            raise ValueError('после исправления запросы совпали; подготовьте отдельный набор без повторов')
        seen.add(result.corrected.casefold())
        corrected.append(replace(case, query=result.corrected))
        changed += result.corrected != result.original
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8', newline='') as target:
        writer = csv.writer(target)
        writer.writerow(QUERY_COLUMNS)
        for case in corrected:
            writer.writerow([case.query_id, case.category, case.query, ';'.join(map(str, case.relevant_ids))])
    return changed


# показываем варианты написания и отличаем предложения от применённых замен
def print_suggestions(result: SpellingResult) -> None:
    print(f'исходный запрос: {result.original}')
    print(f'предложенный запрос: {result.corrected}')
    for suggestion in result.suggestions:
        options = ', '.join(f'{item.word} (правок: {item.distance}; статей: {item.article_frequency})'
                            for item in suggestion.candidates)
        action = f'замена: {suggestion.replacement}' if suggestion.replacement else 'только предложение'
        print(f'{suggestion.original}: {options}; {action}')
    if not result.suggestions:
        print('подходящих исправлений нет')


# параметры построения словаря, проверки фразы и подготовки контрольных запросов
class CliArgs(argparse.Namespace):
    command: str
    dictionary: Path
    qdrant_url: str | None
    qdrant_path: Path | None
    max_tokens: int
    overlap_tokens: int
    query: str
    queries: Path
    output: Path


# выбираем операцию; модель не вызывается ни в одной из этих команд
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    build = commands.add_parser('build', help='собрать словарь из существующей коллекции')
    build.add_argument('--dictionary', type=Path, default=DEFAULT_DICTIONARY)
    location = build.add_mutually_exclusive_group()
    location.add_argument('--qdrant-url')
    location.add_argument('--qdrant-path', type=Path)
    build.add_argument('--max-tokens', type=int, default=1024)
    build.add_argument('--overlap-tokens', type=int, default=128)
    suggest = commands.add_parser('suggest', help='проверить запрос без mws и qdrant')
    suggest.add_argument('--dictionary', type=Path, default=DEFAULT_DICTIONARY)
    suggest.add_argument('--query', required=True)
    prepare = commands.add_parser('prepare-evaluation', help='создать исправленную копию контрольных запросов')
    prepare.add_argument('--dictionary', type=Path, default=DEFAULT_DICTIONARY)
    prepare.add_argument('--queries', type=Path, required=True)
    prepare.add_argument('--output', type=Path, required=True)
    args: CliArgs = parser.parse_args(namespace=CliArgs())
    try:
        if args.command == 'build':
            config = resolve_qdrant_config(args.qdrant_url, args.qdrant_path)
            if config.path is not None and not config.path.is_dir():
                raise ValueError('локальный qdrant не найден; укажите --qdrant-url')
            url = os.environ.get('MWS_EMBEDDINGS_URL', DEFAULT_URL)
            deployment = os.environ.get('MWS_EMBEDDINGS_MODEL', DEFAULT_MODEL)
            name = collection_name(url, deployment, ChunkingConfig(args.max_tokens, args.overlap_tokens))
            with closing(connect_store(config)) as store:
                info = build_dictionary(store, name, args.dictionary)
            print(f'словарь: {args.dictionary}')
            print(f'коллекция: {info.collection}; фрагментов: {info.points}; статей: {info.articles}')
            print(f'отпечаток словаря: {info.sha256}')
        else:
            dictionary = load_dictionary(args.dictionary)
            if args.command == 'suggest':
                print_suggestions(dictionary.suggest(args.query))
            else:
                changed = prepare_evaluation(dictionary, args.queries, args.output)
                print(f'набор для оценки: {args.output}; изменено запросов: {changed}')
                print(f'отпечаток словаря: {dictionary.info.sha256}')
        print('обращений к mws: 0; поисковый индекс не изменялся')
        return 0
    except KeyboardInterrupt:
        print('\nпроверка остановлена')
        return 130
    except UnexpectedResponse as exc:
        parser.exit(1, f'qdrant вернул http {exc.status_code}; проверьте настройки\n')
    except ResponseHandlingException:
        parser.exit(1, 'не удалось связаться с qdrant; проверьте адрес и запущен ли контейнер\n')
    except (ValueError, OSError, RuntimeError, csv.Error) as exc:
        parser.exit(1, f'проверка опечаток не выполнена: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
