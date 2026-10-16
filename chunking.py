"""делим статьи на фрагменты для будущего поискового индекса."""
from __future__ import annotations

import argparse
import hashlib
import sqlite3
from bisect import bisect_left
from collections.abc import Iterator
from contextlib import closing
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Protocol
from uuid import NAMESPACE_URL, uuid5

import requests
from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing

from prepare import DEFAULT_SOURCE, PreparationStats, PreparedArticle, iter_prepared_articles

# фиксируем версию, чтобы одинаковые настройки всегда давали одинаковые фрагменты
MODEL_ID: str = 'BAAI/bge-multilingual-gemma2'
MODEL_REVISION: str = '8365500e19bc804136f4ad12c2494430239a2d72'
MODEL_MAX_TOKENS: int = 8192  # предел модели; у удалённого деплоймента лимит может быть ниже
CHUNKING_VERSION: str = '2'
DEFAULT_TOKENIZER: Path = (
    Path(__file__).resolve().parent / 'models' / MODEL_ID / MODEL_REVISION / 'tokenizer.json'
)


# настройки относятся ко всему входу модели: заголовку, фрагменту и специальным токенам
@dataclass(frozen=True)
class ChunkingConfig:
    max_tokens: int = 1024  # верхняя граница с учётом специальных токенов
    overlap_tokens: int = 128  # повторяем конец предыдущего фрагмента для связности

    def __post_init__(self) -> None:
        if not 64 <= self.max_tokens <= MODEL_MAX_TOKENS:
            raise ValueError(f'max_tokens должен быть от 64 до {MODEL_MAX_TOKENS}')
        if not 0 <= self.overlap_tokens < self.max_tokens // 2:
            raise ValueError('overlap_tokens должен быть неотрицательным и меньше половины max_tokens')


# индексатор позже будет получать такие записи по одной
@dataclass(frozen=True)
class ArticleChunk:
    id: str  # стабильный uuid, который подходит для точки в qdrant
    article_id: int  # номер исходной статьи
    chunk_index: int  # порядковый номер фрагмента, начиная с нуля
    title: str  # полный заголовок для поисковой выдачи
    url: str  # ссылка на оригинал
    hubs: tuple[str, ...]  # тематики для фильтрации
    published_at: str | None  # дата публикации
    start_char: int  # начало фрагмента в подготовленном тексте
    end_char: int  # конец фрагмента, не включая этот символ
    text: str  # исходный отрывок с сохранением переносов и отступов
    embedding_text: str  # готовая строка для вычисления эмбеддинга
    token_count: int  # длина embedding_text со специальными токенами


# интерфейс позволяет проверять разбиение без скачивания модели
class TextTokenizer(Protocol):
    def offsets(self, text: str) -> list[tuple[int, int]]: ...
    def count(self, text: str) -> int: ...


class ModelTokenizer:
    def __init__(self, path: Path) -> None:
        self.tokenizer: Tokenizer = Tokenizer.from_file(str(path))
        # отклоняем старый кеш e5 и файлы с несовместимыми специальными токенами
        if self.tokenizer.token_to_id('<bos>') != 2 or self.tokenizer.token_to_id('<eos>') != 1:
            raise ValueError('нужен tokenizer.json модели bge-multilingual-gemma2')
        # настройки gemma требуют bos и eos; учитываем оба при подсчёте длины
        self.tokenizer.post_processor = TemplateProcessing(
            single='<bos> $A <eos>',
            special_tokens=[('<bos>', 2), ('<eos>', 1)],
        )
        # запрещаем скрытое обрезание входа и добавление пустых токенов
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()

    def offsets(self, text: str) -> list[tuple[int, int]]:
        return self.tokenizer.encode(text, add_special_tokens=False).offsets

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=True).ids)


def load_tokenizer(path: Path = DEFAULT_TOKENIZER) -> ModelTokenizer:
    """скачиваем только токенизатор при первом запуске и затем используем локальный файл."""
    if path.exists():
        return ModelTokenizer(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f'https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/tokenizer.json'
    temporary: Path | None = None
    try:
        # временный файл не оставит повреждённый кеш при обрыве загрузки
        with requests.get(url, stream=True, timeout=(10, 60)) as response:
            response.raise_for_status()
            with NamedTemporaryFile(dir=path.parent, suffix='.tmp', delete=False) as stream:
                temporary = Path(stream.name)
                for block in response.iter_content(chunk_size=1024 * 1024):
                    stream.write(block)
        tokenizer = ModelTokenizer(temporary)
        temporary.replace(path)
        return tokenizer
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def embedding_header(title: str, tokenizer: TextTokenizer, max_tokens: int) -> str:
    # длинный заголовок сокращаем только во входе модели; полный сохраняем в метаданных
    offsets = tokenizer.offsets(title)
    title_limit = min(64, max_tokens // 4)
    if len(offsets) > title_limit:
        title = title[:offsets[title_limit][0]].rstrip()
    # документы gemma получают обычный текст без префикса passage из e5
    return title + '\n\n'


def chunk_article(
    article: PreparedArticle, tokenizer: TextTokenizer, config: ChunkingConfig,
) -> Iterator[ArticleChunk]:
    """разбиваем текст по токенам, стараясь завершать фрагменты на переносах строк."""
    offsets = tokenizer.offsets(article.text)
    if not offsets:
        raise ValueError(f'статья {article.id}: текст не содержит токенов')
    starts = [start for start, _ in offsets]
    header = embedding_header(article.title, tokenizer, config.max_tokens)
    # оставляем запас на изменение токенизации на стыке заголовка и текста
    body_budget = config.max_tokens - tokenizer.count(header) - 8
    if body_budget <= config.overlap_tokens:
        raise ValueError('для текста мало места: увеличьте max_tokens или уменьшите overlap_tokens')
    start_token = 0
    chunk_index = 0
    while start_token < len(offsets):
        end_token = min(start_token + body_budget, len(offsets))
        if end_token < len(offsets):
            # ищем перенос во второй половине окна, чтобы не создавать слишком короткие куски
            middle = starts[start_token + (end_token - start_token) // 2]
            boundary = article.text.rfind('\n', middle, starts[end_token])
            if boundary >= 0:
                end_token = bisect_left(starts, boundary + 1, start_token + 1, end_token)
        start_char = 0 if start_token == 0 else starts[start_token]
        while True:
            end_char = len(article.text) if end_token == len(offsets) else starts[end_token]
            text = article.text[start_char:end_char]
            embedding_text = header + text
            token_count = tokenizer.count(embedding_text)
            if token_count <= config.max_tokens:
                break
            # фактическая длина важнее предварительной оценки: ничего молча не обрезаем
            end_token -= max(1, token_count - config.max_tokens)
            if end_token <= start_token:
                raise ValueError(f'статья {article.id}: не удалось вместить фрагмент в лимит')
        if not text.strip():
            raise ValueError(f'статья {article.id}: получился пустой фрагмент')
        # настройки и содержимое входят в id, поэтому повторный запуск не создаёт новые точки
        fingerprint = hashlib.sha256(embedding_text.encode('utf-8')).hexdigest()
        identity = (
            f'habr:{article.id}:{article.text_hash}:{CHUNKING_VERSION}:{MODEL_REVISION}:'
            f'{config.max_tokens}:{config.overlap_tokens}:{chunk_index}:{fingerprint}'
        )
        yield ArticleChunk(
            id=str(uuid5(NAMESPACE_URL, identity)), article_id=article.id,
            chunk_index=chunk_index, title=article.title, url=article.url,
            hubs=article.hubs, published_at=article.published_at,
            start_char=start_char, end_char=end_char, text=text,
            embedding_text=embedding_text, token_count=token_count,
        )
        if end_token == len(offsets):
            break
        # даже при коротком фрагменте обязательно продвигаемся вперёд
        start_token = max(start_token + 1, end_token - config.overlap_tokens)
        chunk_index += 1


# типы аргументов позволяют анализатору редактора проверять запуск
class CliArgs(argparse.Namespace):
    source: Path
    tokenizer_file: Path
    limit: int
    max_tokens: int
    overlap_tokens: int
    preview: int


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE, help='база исходных статей')
    parser.add_argument('--tokenizer-file', type=Path, default=DEFAULT_TOKENIZER, help='путь к локальному токенизатору')
    parser.add_argument('--limit', type=int, default=5, help='сколько статей проверить')
    parser.add_argument('--max-tokens', type=int, default=1024, help='лимит входа модели вместе с заголовком')
    parser.add_argument('--overlap-tokens', type=int, default=128, help='перекрытие соседних фрагментов')
    parser.add_argument('--preview', type=int, default=2, help='сколько примеров вывести')
    args: CliArgs = parser.parse_args(namespace=CliArgs())
    if args.limit < 1 or args.preview < 0:
        parser.error('limit должен быть положительным, а preview неотрицательным')
    if not args.source.is_file():
        parser.error(f'исходная база не найдена: {args.source}')
    try:
        config = ChunkingConfig(args.max_tokens, args.overlap_tokens)
        tokenizer = load_tokenizer(args.tokenizer_file)
        stats = PreparationStats()
        articles = 0
        chunks = 0
        max_length = 0
        # закрываем чтение базы и при достижении лимита, и при ошибке
        with closing(iter_prepared_articles(args.source, stats)) as prepared:
            for article in islice(prepared, args.limit):
                articles += 1
                for chunk in chunk_article(article, tokenizer, config):
                    chunks += 1
                    max_length = max(max_length, chunk.token_count)
                    if chunks <= args.preview:
                        print(f'\nстатья {chunk.article_id}, фрагмент {chunk.chunk_index}, токенов: {chunk.token_count}')
                        print(chunk.title)
                        print(chunk.text[:400])
        print(f'\nобработано статей: {articles}')
        print(f'создано фрагментов в памяти: {chunks}')
        print(f'максимальная длина входа: {max_length} / {config.max_tokens} токенов')
        print(f'отклонено записей: {sum(stats.rejected.values())}')
        return 0 if articles and not stats.rejected else 1
    except (OSError, ValueError, sqlite3.Error, requests.RequestException) as exc:
        parser.exit(1, f'не удалось подготовить фрагменты: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
