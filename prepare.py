"""проверяем корпус и подготавливаем статьи в памяти, не меняя исходную базу."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from collections import Counter
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit

# по умолчанию читаем базу скрапера из папки data
DEFAULT_SOURCE: Path = Path(__file__).resolve().parent / 'data' / 'corpus.sqlite3'
# удаляем управляющие символы, но оставляем табуляцию и переносы
CONTROL_CHARS: re.Pattern[str] = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
# принимаем обычные статьи и статьи из блогов компаний
ARTICLE_PATH: re.Pattern[str] = re.compile(
    r'^/ru/(?:articles|companies/[^/]+/articles)/(\d+)/$'
)


# единый формат статьи для последующего разбиения и индексации
@dataclass(frozen=True)
class PreparedArticle:
    id: int  # номер статьи на хабре
    url: str  # ссылка на оригинал
    title: str  # заголовок для выдачи и индексации
    text: str  # полный очищенный текст статьи
    author: str | None  # ник автора, если указан
    published_at: str | None  # дата публикации из страницы
    hubs: tuple[str, ...]  # тематические разделы хабра
    tags: tuple[str, ...]  # авторские теги
    text_hash: str  # отпечаток текста для поиска полных дублей


# счётчики заполняются по мере чтения корпуса
@dataclass
class PreparationStats:
    scanned: int = 0  # сколько записей прочитано из базы
    accepted: int = 0  # сколько уникальных статей прошло проверку
    duplicates: int = 0  # сколько повторов текста пропущено
    rejected: Counter[str] = field(default_factory=Counter[str])  # причина ошибки и её частота
    text_chars: int = 0  # сумма длин принятых текстов
    min_text_chars: int | None = None  # длина самого короткого текста
    max_text_chars: int = 0  # длина самого длинного текста


# сохраняем переносы, отступы кода, регистр и технические обозначения
def clean_text(value: str) -> str:
    value = value.replace('\r\n', '\n').replace('\r', '\n')
    return CONTROL_CHARS.sub('', value).strip()


def required_text(record: dict[str, object], key: str) -> str:
    """читаем обязательную строку и отклоняем пустое значение."""
    value = record.get(key)
    if not isinstance(value, str):
        raise ValueError(f'{key}: ожидается строка')
    cleaned = clean_text(value)
    if not cleaned:
        raise ValueError(f'{key}: пустое значение')
    return cleaned


def optional_text(record: dict[str, object], key: str) -> str | None:
    """читаем необязательную строку; отсутствие обозначаем через none."""
    value = record.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f'{key}: ожидается строка или null')
    return clean_text(value) or None


def string_list(record: dict[str, object], key: str) -> tuple[str, ...]:
    """проверяем список хабов или тегов и убираем повторяющиеся названия."""
    value = record.get(key, [])
    if not isinstance(value, list):
        raise ValueError(f'{key}: ожидается список строк')
    # cast уточняет тип для анализатора, сами элементы проверяем ниже
    items = cast(list[object], value)
    result: list[str] = []
    for item in items:
        if not isinstance(item, str):
            raise ValueError(f'{key}: ожидается список строк')
        cleaned = clean_text(item)
        if cleaned:
            result.append(cleaned)
    return tuple(dict.fromkeys(result))


def prepare_article(payload: str, stored_id: int) -> PreparedArticle:
    # проверяем типы до передачи данных в индексатор
    decoded: object = json.loads(payload)
    if not isinstance(decoded, dict):
        raise ValueError('документ: ожидается объект')
    # ключи объекта json строковые, тип каждого нужного значения проверяем отдельно
    record = cast(dict[str, object], decoded)
    # номер внутри документа должен совпадать с ключом записи в sqlite
    article_id = record.get('id')
    if type(article_id) is not int or article_id <= 0 or article_id != stored_id:
        raise ValueError('id: несовпадение с базой или неверный тип')
    url = required_text(record, 'url')
    # проверяем домен и номер статьи в ссылке
    parsed = urlsplit(url)
    match = ARTICLE_PATH.fullmatch(parsed.path)
    if (parsed.scheme != 'https' or parsed.netloc != 'habr.com'
            or match is None or int(match[1]) != article_id):
        raise ValueError('url: неверная ссылка на статью')
    text = required_text(record, 'text')
    return PreparedArticle(
        id=article_id,
        url=url,
        title=required_text(record, 'title'),
        text=text,
        author=optional_text(record, 'author'),
        published_at=optional_text(record, 'published_at'),
        hubs=string_list(record, 'hubs'),
        tags=string_list(record, 'tags'),
        text_hash=hashlib.sha256(text.encode('utf-8')).hexdigest(),
    )


def iter_prepared_articles(
    source: Path, stats: PreparationStats,
) -> Generator[PreparedArticle, None, None]:
    """читаем согласованный снимок базы и выдаём статьи без полных дублей текста."""
    source = source.resolve(strict=True)
    seen: set[str] = set()  # здесь храним только отпечатки уже принятых текстов
    # режим чтения не позволяет случайно изменить исходный корпус
    db = sqlite3.connect(source.as_uri() + '?mode=ro', uri=True)
    try:
        # все записи читаются из одного согласованного состояния базы
        db.execute('BEGIN')
        rows = db.execute('SELECT id, payload FROM articles ORDER BY id')
        for stored_id, payload in rows:
            stats.scanned += 1
            try:
                if type(stored_id) is not int or not isinstance(payload, str):
                    raise ValueError('запись базы: неверные типы полей')
                article = prepare_article(payload, stored_id)
            except ValueError as exc:
                # повреждённые записи учитываем в отчёте, остальные продолжаем читать
                stats.rejected[str(exc)] += 1
                continue
            # одинаковые тексты под разными id индексировать повторно не нужно
            if article.text_hash in seen:
                stats.duplicates += 1
                continue
            seen.add(article.text_hash)
            length = len(article.text)
            stats.accepted += 1
            stats.text_chars += length
            stats.min_text_chars = min(stats.min_text_chars, length) if stats.min_text_chars is not None else length
            stats.max_text_chars = max(stats.max_text_chars, length)
            # отдаём по одной статье, не загружая весь корпус в память
            yield article
    finally:
        db.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()
    source: Path = args.source
    stats = PreparationStats()
    try:
        # отдельный запуск только проверяет корпус, ничего не скачивает и не записывает
        for _ in iter_prepared_articles(source, stats):
            pass
    except (OSError, sqlite3.Error) as exc:
        parser.exit(1, f'не удалось прочитать корпус: {exc}\n')
    print(f'прочитано статей: {stats.scanned}')
    print(f'готово к обработке: {stats.accepted}')
    print(f'дубликатов текста: {stats.duplicates}')
    print(f'отклонено записей: {sum(stats.rejected.values())}')
    if stats.accepted:
        print(f'длина текста, символов: от {stats.min_text_chars} до {stats.max_text_chars}')
        print(f'средняя длина текста: {stats.text_chars / stats.accepted:.0f}')
    for reason, count in stats.rejected.items():
        print(f'  {reason}: {count}')
    return 0 if stats.accepted and not stats.rejected else 1


if __name__ == '__main__':
    raise SystemExit(main())
