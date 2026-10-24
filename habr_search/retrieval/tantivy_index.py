"""строим обратный индекс целых статей и ищем локально через rust-библиотеку."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import tantivy

from habr_search.ingestion.prepare import DEFAULT_SOURCE, PreparationStats, iter_prepared_articles
from habr_search.paths import DATA_DIR
from habr_search.retrieval.spelling import MAX_QUERY_CHARS, save_dictionary, vocabulary_words

DEFAULT_INDEX: Path = DATA_DIR / 'tantivy'
FORMAT_VERSION: int = 1
WORD_PATTERN: re.Pattern[str] = re.compile(r'[^\W_]+', re.UNICODE)


@dataclass(frozen=True)
class IndexInfo:
    articles: int
    revision: str


@dataclass(frozen=True)
class LexicalResult:
    article_id: int
    title: str
    url: str
    snippet: str
    score: float  # bm25 не ограничен диапазоном косинусного сходства


def register_analyzer(index: tantivy.Index) -> None:
    """регистрируем одинаковую обработку слов при записи и открытии индекса."""
    analyzer = (tantivy.TextAnalyzerBuilder(tantivy.Tokenizer.simple())
                .filter(tantivy.Filter.remove_long(80))
                .filter(tantivy.Filter.lowercase())
                .filter(tantivy.Filter.stemmer('russian')).build())
    index.register_tokenizer('russian', analyzer)


def literal_query(text: str) -> str:
    """принимаем обычный текст, запрещая выполнение поискового синтаксиса пользователя."""
    if not text.strip() or len(text) > MAX_QUERY_CHARS:
        raise ValueError('нужен непустой запрос до 4000 символов')
    words = list(dict.fromkeys(WORD_PATTERN.findall(text.lower().replace('ё', 'е'))))
    if len(words) > 64 or any(len(word) > 80 for word in words):
        raise ValueError('запрос слишком длинный: максимум 64 слова по 80 символов')
    # or сохраняет частичные совпадения, не меняя написание и смысл слов
    return ' OR '.join(f'"{word}"' for word in words)


def build_index(source: Path, destination: Path) -> PreparationStats:
    """публикуем готовую папку; существующий индекс никогда не перезаписываем."""
    if destination.exists():
        raise ValueError(f'индекс уже существует: {destination}; укажите новый --index-path')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.tantivy-', dir=destination.parent))
    stats = PreparationStats()
    try:
        builder = tantivy.SchemaBuilder()
        builder.add_unsigned_field('article_id', stored=True)
        builder.add_text_field('title', stored=True, tokenizer_name='russian')
        builder.add_text_field('body', stored=True, tokenizer_name='russian')
        builder.add_text_field('url', stored=True, tokenizer_name='raw')
        index = tantivy.Index(builder.build(), path=str(temporary), reuse=False)
        register_analyzer(index)
        writer = index.writer(heap_size=64_000_000, num_threads=1)
        digest = hashlib.sha256(f'tantivy:{FORMAT_VERSION}'.encode())
        frequencies: Counter[str] = Counter()
        with closing(iter_prepared_articles(source, stats)) as articles:
            for article in articles:
                document = tantivy.Document()
                document.add_unsigned('article_id', article.id)
                document.add_text('title', article.title.replace('ё', 'е').replace('Ё', 'Е'))
                document.add_text('body', article.text.replace('ё', 'е').replace('Ё', 'Е'))
                document.add_text('url', article.url)
                writer.add_document(document)
                digest.update(json.dumps([article.id, article.url, article.title, article.text],
                                         ensure_ascii=False).encode())
                frequencies.update(vocabulary_words(article.title + '\n' + article.text))
        if not stats.accepted:
            raise ValueError('в корпусе нет подходящих статей')
        writer.commit()
        writer.wait_merging_threads()
        revision = digest.hexdigest()
        # словарь строится из тех же статей и получает версию именно этого индекса
        save_dictionary(temporary / 'spelling.sqlite3', frequencies, 'tantivy',
                        stats.accepted, stats.accepted, revision)
        metadata = {'format': FORMAT_VERSION, 'articles': stats.accepted, 'revision': revision}
        (temporary / 'manifest.json').write_text(json.dumps(metadata), encoding='utf-8')
        # готовность проверяем до публикации; при ошибке исходный корпус не меняется
        loaded = TantivySearch(temporary)
        del loaded, writer, index
        os.rename(temporary, destination)
        return stats
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


class TantivySearch:
    """держим неизменяемый снимок индекса в памяти между запросами."""
    def __init__(self, path: Path = DEFAULT_INDEX) -> None:
        if not (path / 'manifest.json').is_file():
            raise ValueError(f'индекс не найден: {path}; выполните python -m habr_search.retrieval.tantivy_index build')
        raw: object = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
        if not isinstance(raw, dict):
            raise ValueError('некорректный manifest индекса')
        metadata = cast(dict[str, object], raw)
        count, revision = metadata.get('articles'), metadata.get('revision')
        if (metadata.get('format') != FORMAT_VERSION or type(count) is not int or count < 1
                or not isinstance(revision, str) or re.fullmatch('[a-f0-9]{64}', revision) is None):
            raise ValueError('неподдерживаемый или повреждённый manifest индекса')
        self.info: IndexInfo = IndexInfo(count, revision)
        self.path: Path = path
        self.index: tantivy.Index = tantivy.Index.open(str(path))
        register_analyzer(self.index)
        self.searcher: tantivy.Searcher = self.index.searcher()
        if self.searcher.num_docs != count:
            raise ValueError('число документов не совпало с manifest; пересоберите индекс')

    def search(self, text: str, top_k: int = 5) -> list[LexicalResult]:
        """ищем по заголовкам и текстам; отрывок выбираем рядом с совпадением."""
        if not 1 <= top_k <= 20:
            raise ValueError('top_k должен быть от 1 до 20')
        query_text = literal_query(text)
        if not query_text:
            return []
        query = self.index.parse_query(query_text, ['title', 'body'],
                                       field_boosts={'title': 3.0, 'body': 1.0})
        hits = self.searcher.search(query, limit=top_k, count=False)
        snippets = tantivy.SnippetGenerator.create(self.searcher, query, self.index.schema, 'body')
        snippets.set_max_num_chars(350)
        results: list[LexicalResult] = []
        for raw_score, address in hits.hits:
            document = self.searcher.doc(address)
            article_id: object = document.get_first('article_id')
            title: object = document.get_first('title')
            url: object = document.get_first('url')
            score: object = raw_score
            if (type(article_id) is not int or not isinstance(title, str) or not isinstance(url, str)
                    or not isinstance(score, (float, int))):
                raise ValueError('повреждённый документ индекса')
            fragment = snippets.snippet_from_doc(document).fragment()
            results.append(LexicalResult(article_id, title, url, ' '.join(fragment.split()), float(score)))
        return results


class CliArgs(argparse.Namespace):
    command: str
    source: Path
    index_path: Path
    query: str | None
    top_k: int


def main() -> int:
    """отдельно строим индекс или проверяем выдачу без модели и сети."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['build', 'search'])
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--index-path', type=Path, default=DEFAULT_INDEX)
    parser.add_argument('--query')
    parser.add_argument('--top-k', type=int, default=5)
    args = parser.parse_args(namespace=CliArgs())
    try:
        if args.command == 'build':
            stats = build_index(args.source, args.index_path)
            print(f'статей: {stats.accepted}; отклонено: {sum(stats.rejected.values())}; дубликатов: {stats.duplicates}')
            print(f'индекс: {args.index_path}')
        else:
            if args.query is None:
                parser.error('для search нужен --query')
            for rank, result in enumerate(TantivySearch(args.index_path).search(args.query, args.top_k), 1):
                print(f'{rank}. {result.title}\n{result.url}\n{result.snippet}\n')
        return 0
    except (ValueError, OSError) as exc:
        parser.exit(1, f'tantivy: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
