"""подбираем исправления через символьные триграммы и расстояние редактирования."""
from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import tempfile
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from habr_search.paths import DATA_DIR

DEFAULT_DICTIONARY: Path = DATA_DIR / 'spelling.sqlite3'
DICTIONARY_VERSION: str = '1'
WORD_PATTERN: re.Pattern[str] = re.compile(r'[a-zA-Zа-яА-ЯёЁ]+')
MAX_WORD_LENGTH: int = 40
MAX_QUERY_CHARS: int = 4000


# сведения связывают словарь с коллекцией и конкретным набором частот
@dataclass(frozen=True)
class DictionaryInfo:
    collection: str
    points: int
    articles: int
    sha256: str
    created_at: str


# вариант написания с числом правок, частотой и близостью триграмм
@dataclass(frozen=True)
class SpellingCandidate:
    word: str
    distance: int
    article_frequency: int
    similarity: float


# координаты относятся к исходному запросу, включая его регистр и пробелы
@dataclass(frozen=True)
class WordSuggestion:
    original: str
    start: int
    end: int
    candidates: tuple[SpellingCandidate, ...]
    replacement: str | None


# результат содержит исходную фразу, исправленную фразу и все предложения
@dataclass(frozen=True)
class SpellingResult:
    original: str
    corrected: str
    suggestions: tuple[WordSuggestion, ...]


def normalize_word(word: str) -> str:
    """приводим регистр и ё к одной форме только для сопоставления слов."""
    return unicodedata.normalize('NFKC', word).casefold().replace('ё', 'е')


def protected_token(text: str, start: int, end: int) -> bool:
    """не исправляем части адресов, идентификаторов, смешанный регистр и алфавит."""
    word = text[start:end]
    adjacent = (text[start - 1] if start else '') + (text[end] if end < len(text) else '')
    if any(char.isalnum() or char in '_/@:#+\\-`' for char in adjacent):
        return True
    if ((start >= 2 and text[start - 1] == '.' and text[start - 2].isalnum())
            or (end + 1 < len(text) and text[end] == '.' and text[end + 1].isalnum())):
        return True
    latin = re.search(r'[a-zA-Z]', word) is not None
    cyrillic = re.search(r'[а-яА-ЯёЁ]', word) is not None
    return (latin and cyrillic) or word.isupper() or any(char.isupper() for char in word[1:])


def vocabulary_words(text: str) -> set[str]:
    """одно слово учитывается в статье один раз, даже если фрагменты перекрываются."""
    return {
        normalize_word(match[0]) for match in WORD_PATTERN.finditer(text)
        if len(match[0]) <= MAX_WORD_LENGTH
    }


def ngrams(word: str, size: int = 3) -> frozenset[str]:
    """добавляем границы слова, чтобы учитывать ошибки в начале и конце."""
    if size < 1:
        raise ValueError('размер n-граммы должен быть положительным')
    padded = '^' + normalize_word(word) + '$'
    return frozenset(padded[i:i + size] for i in range(max(0, len(padded) - size + 1)))


def edit_distance(left: str, right: str) -> int:
    """считаем вставки, удаления, замены и соседние перестановки по алгоритму osa."""
    previous = list(range(len(right) + 1))
    before_previous = previous
    for i, first in enumerate(left, 1):
        row = [i]
        for j, second in enumerate(right, 1):
            value = min(row[j - 1] + 1, previous[j] + 1, previous[j - 1] + (first != second))
            if i > 1 and j > 1 and first == right[j - 2] and left[i - 2] == second:
                value = min(value, before_previous[j - 2] + 1)
            row.append(value)
        before_previous, previous = previous, row
    return previous[-1]


def dictionary_hash(frequencies: Mapping[str, int]) -> str:
    """фиксируем слова и частоты независимо от порядка их чтения."""
    digest = hashlib.sha256()
    for word, frequency in sorted(frequencies.items()):
        digest.update(f'{word}\t{frequency}\n'.encode())
    return digest.hexdigest()


# держим частоты слов и обратный индекс триграмм для быстрого отбора вариантов
class SpellingDictionary:
    # проверяем словарь и связываем каждую триграмму со словами, где она встречается
    def __init__(self, frequencies: Mapping[str, int], info: DictionaryInfo) -> None:
        if (not info.collection or info.points < 1 or info.articles < 1 or info.articles > info.points
                or not frequencies):
            raise ValueError('нужен непустой словарь с метаданными коллекции')
        for word, frequency in frequencies.items():
            if (not WORD_PATTERN.fullmatch(word) or word != normalize_word(word)
                    or len(word) > MAX_WORD_LENGTH or type(frequency) is not int
                    or not 1 <= frequency <= info.articles):
                raise ValueError('в словаре некорректное слово или частота')
        if dictionary_hash(frequencies) != info.sha256:
            raise ValueError('содержимое словаря не совпадает с его отпечатком')
        self.info: DictionaryInfo = info
        self.frequencies: dict[str, int] = dict(frequencies)
        self.postings: dict[str, set[str]] = defaultdict(set)
        self.grams: dict[str, frozenset[str]] = {}
        for word, frequency in frequencies.items():
            # редкие слова остаются известными, но не используются для замены других слов
            if len(word) < 5 or frequency < 2:
                continue
            grams = ngrams(word)
            self.grams[word] = grams
            for gram in grams:
                self.postings[gram].add(word)

    # проверяем имя коллекции и число точек; изменения текста при том же числе не обнаруживаются
    def validate_context(self, collection: str, points: int) -> None:
        if self.info.collection != collection or self.info.points != points:
            raise ValueError('словарь относится к другой коллекции или числу фрагментов; пересоберите его')

    # отбираем слова по триграммам, затем уточняем число правок и порядок вариантов
    def candidates(self, word: str, limit: int = 3) -> tuple[SpellingCandidate, ...]:
        if limit < 1:
            raise ValueError('число кандидатов должно быть положительным')
        word = normalize_word(word)
        if word in self.frequencies or not 5 <= len(word) <= MAX_WORD_LENGTH:
            return ()
        max_distance = 2 if len(word) >= 8 else 1
        grams = ngrams(word)
        overlap: Counter[str] = Counter()
        for gram in grams:
            overlap.update(self.postings.get(gram, ()))
        ranked: list[SpellingCandidate] = []
        # проверяем все совпавшие по триграммам слова, чтобы обрезание не скрывало неоднозначность
        for candidate, common in overlap.items():
            if abs(len(candidate) - len(word)) > max_distance:
                continue
            distance = edit_distance(word, candidate)
            if distance > max_distance:
                continue
            similarity = common / (len(grams) + len(self.grams[candidate]) - common)
            ranked.append(SpellingCandidate(candidate, distance, self.frequencies[candidate], similarity))
        ranked.sort(key=lambda item: (item.distance, -item.article_frequency, -item.similarity, item.word))
        return tuple(ranked[:limit])

    # разбираем слова запроса и собираем новую фразу, сохраняя остальные символы
    def suggest(self, query: str) -> SpellingResult:
        if not query.strip() or len(query) > MAX_QUERY_CHARS:
            raise ValueError(f'нужен непустой запрос длиной до {MAX_QUERY_CHARS} символов')
        suggestions: list[WordSuggestion] = []
        corrected: list[str] = []
        previous_end = 0
        for match in WORD_PATTERN.finditer(query):
            original = match[0]
            if protected_token(query, match.start(), match.end()):
                continue
            candidates = self.candidates(original)
            if not candidates:
                continue
            best = candidates[0]
            # применяем только единственный ближайший вариант с одной правкой и частотой от трёх статей
            unambiguous = len(candidates) == 1 or candidates[1].distance > best.distance
            replacement: str | None = None
            # заглавное слово внутри фразы может быть названием неизвестной технологии
            allow_replacement = not original[0].isupper() or not query[:match.start()].strip()
            if unambiguous and allow_replacement and best.distance == 1 and best.article_frequency >= 3:
                replacement = best.word.capitalize() if original[0].isupper() else best.word
                corrected.extend((query[previous_end:match.start()], replacement))
                previous_end = match.end()
            suggestions.append(WordSuggestion(original, match.start(), match.end(), candidates, replacement))
        corrected.append(query[previous_end:])
        return SpellingResult(query, ''.join(corrected), tuple(suggestions))


def save_dictionary(path: Path, frequencies: Mapping[str, int], collection: str,
                    points: int, articles: int) -> DictionaryInfo:
    """записываем готовую sqlite-базу, не перезаписывая существующий файл."""
    info = DictionaryInfo(collection, points, articles, dictionary_hash(frequencies),
                          datetime.now(timezone.utc).isoformat())
    SpellingDictionary(frequencies, info)
    if path.exists():
        raise ValueError(f'словарь уже существует: {path}; укажите другой --dictionary')
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix='.spelling-', suffix='.sqlite3', dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        with closing(sqlite3.connect(temporary)) as connection, connection:
            connection.execute('CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
            connection.execute('CREATE TABLE words (word TEXT PRIMARY KEY, article_frequency INTEGER NOT NULL)')
            metadata = {
                'version': DICTIONARY_VERSION, 'collection': collection, 'points': str(points),
                'articles': str(articles), 'sha256': info.sha256, 'created_at': info.created_at,
            }
            connection.executemany('INSERT INTO metadata VALUES (?, ?)', metadata.items())
            connection.executemany('INSERT INTO words VALUES (?, ?)', sorted(frequencies.items()))
        # ссылка публикует готовый файл и не заменяет словарь, созданный параллельно
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return info


def load_dictionary(path: Path) -> SpellingDictionary:
    """читаем базу только для чтения и проверяем формат, частоты и отпечаток."""
    if not path.is_file():
        raise ValueError(f'словарь не найден: {path}; сначала выполните python -m habr_search.retrieval.spellcheck build')
    try:
        with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)) as connection:
            metadata = dict(connection.execute('SELECT key, value FROM metadata').fetchall())
            if metadata.get('version') != DICTIONARY_VERSION:
                raise ValueError('неподдерживаемая версия словаря')
            info = DictionaryInfo(str(metadata['collection']), int(metadata['points']), int(metadata['articles']),
                                  str(metadata['sha256']), str(metadata['created_at']))
            frequencies = dict(connection.execute('SELECT word, article_frequency FROM words').fetchall())
        # sqlite возвращает динамические значения, поэтому проверяем типы до передачи алгоритму
        checked: dict[str, int] = {}
        for word, frequency in frequencies.items():
            if not isinstance(word, str) or type(frequency) is not int:
                raise ValueError('в словаре некорректные типы полей')
            checked[word] = frequency
        return SpellingDictionary(checked, info)
    except (sqlite3.Error, KeyError, TypeError, OverflowError) as exc:
        raise ValueError('не удалось прочитать словарь; пересоберите его') from exc
