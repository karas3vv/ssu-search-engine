"""собираем статьи хабра с возможностью продолжить после остановки."""
# вход: ленты хабра; выход: база статей и очередь обхода в папке data
from __future__ import annotations

import argparse
import gzip
import json
import logging
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, TypedDict, cast
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup, Tag

from habr_search.paths import DATA_DIR

BASE: str = 'https://habr.com'
AGENT: str = 'HabrStudyCrawler/0.1'
ARTICLE: re.Pattern[str] = re.compile(r'^/ru/(?:articles|companies/[^/]+/articles)/(\d+)/$')
HUB: re.Pattern[str] = re.compile(r'^/ru/hubs/[^/]+/$')
MAX_REDIRECTS: int = 5


# различаем страницы лент и отдельные статьи в очереди обхода
TaskKind = Literal['article', 'listing']
# поля одной задачи: порядковый номер, url и тип страницы
TaskRow = tuple[int, str, TaskKind]


# структура сохраняемой статьи
class ArticleDocument(TypedDict):
    id: int  # номер статьи на хабре
    url: str  # ссылка на оригинал
    title: str  # заголовок статьи
    text: str  # полный текст без html-разметки
    author: str | None  # ник автора, если найден
    published_at: str | None  # время публикации из страницы
    hubs: list[str]  # тематические разделы статьи
    tags: list[str]  # теги из ссылок статьи
    fetched_at: str  # время загрузки в utc


# типы параметров командной строки
class CliArgs(argparse.Namespace):
    limit: int
    data_dir: Path
    delay: float
    feed: list[str] | None
    save_html: bool
    export_only: bool
    retry_redirects: bool


# сохраняем конечный адрес, чтобы статья компании не получила старую ссылку
@dataclass(frozen=True)
class FetchedPage:
    html: str
    url: str


def article_url(url: str) -> str | None:
    """принимаем ссылки на статьи хабра, сохраняя путь блога компании."""
    try:
        parsed = urlsplit(urljoin(BASE, url))
    except ValueError:
        return None
    if parsed.scheme != 'https' or parsed.netloc != 'habr.com' or not ARTICLE.fullmatch(parsed.path):
        return None
    return BASE + parsed.path


def string_attr(tag: Tag, name: str) -> str:
    """получаем строковый атрибут тега и проверяем его тип."""
    value = tag.get(name)
    if not isinstance(value, str):
        raise ValueError(f'Expected string attribute: {name}')
    return value


# извлекаем из страницы ленты ссылки на отдельные статьи
def article_links(html: str) -> tuple[list[str], BeautifulSoup]:
    # собираем ссылки на статьи из ленты без дублей
    soup = BeautifulSoup(html, 'html.parser')
    result: dict[str, str] = {}
    for a in soup.select('a.tm-title__link[href]'):
        url = article_url(string_attr(a, 'href'))
        if url is not None:
            match = ARTICLE.fullmatch(urlsplit(url).path)
            assert match is not None
            # если в ленте есть оба адреса одной статьи, предпочитаем ссылку компании
            if match[1] not in result or '/companies/' in url:
                result[match[1]] = url
    return list(result.values()), soup


# получаем текст и метаданные статьи, а также ссылки на её тематические ленты
def parse_article(html: str, url: str) -> tuple[ArticleDocument, list[str]]:
    # находим заголовок и основной текст статьи
    soup = BeautifulSoup(html, 'html.parser')
    article = soup.select_one('article')
    if article is None:
        raise ValueError('article element missing')
    title = article.select_one('h1')
    body = article.select_one('.article-formatted-body')
    if title is None or body is None:
        raise ValueError('title/body missing; check page layout')
    # убираем служебные элементы и извлекаем текст
    for node in body.select('script, style, noscript'):
        node.decompose()
    text = body.get_text('\n', strip=True)
    if not text:
        raise ValueError('empty article body')
    # извлекаем дату, автора, хабы и теги
    stamp = article.select_one('time[datetime]')
    author = article.select_one('.tm-user-info__username')
    hubs: list[str] = []
    feeds: list[str] = []
    for a in article.select('a.tm-publication-hub__link[href]'):
        path = urlsplit(string_attr(a, 'href')).path
        if HUB.fullmatch(path):
            hubs.append(a.get_text(' ', strip=True).removesuffix('*').strip())
            feeds.append(urljoin(BASE, path + 'articles/'))
    tags: list[str] = [a.get_text(' ', strip=True) for a in article.select('a[href*="/search/"]')
            if 'q=' in string_attr(a, 'href')]
    match = ARTICLE.fullmatch(urlsplit(url).path)
    if match is None:
        raise ValueError('unexpected article URL')
    # собираем данные для записи в базу
    document: ArticleDocument = {
        'id': int(match[1]),
        'url': url,
        'title': title.get_text(' ', strip=True),
        'text': text,
        'author': author.get_text(strip=True) if author else None,
        'published_at': string_attr(stamp, 'datetime') if stamp else None,
        'hubs': list(dict.fromkeys(hubs)),
        'tags': list(dict.fromkeys(tags)),
        'fetched_at': datetime.now(timezone.utc).isoformat(),
    }
    return document, feeds


# загрузка страниц с паузами и повторными попытками
class Fetcher:
    # создаём сессию и задаём паузу между обращениями к хабру
    def __init__(self, delay: float) -> None:
        self.session: requests.Session = requests.Session()
        self.session.headers['User-Agent'] = AGENT
        self.delay: float = max(10.0, delay)
        self.last: float = time.monotonic()  # выдерживаем паузу и после перезапуска
        self.robots: RobotFileParser | None = None

    # загружаем одну страницу с контролем адреса, паузами и обработкой сетевых ошибок
    def get(self, url: str) -> FetchedPage:
        # проверяем каждый переход до запроса, сохраняя id исходной статьи
        original_article = ARTICLE.fullmatch(urlsplit(url).path)
        visited: set[str] = set()
        while True:
            parsed = urlsplit(url)
            if parsed.scheme != 'https' or parsed.netloc != 'habr.com':
                raise ValueError('only https://habr.com is allowed')
            if self.robots and not self.robots.can_fetch(AGENT, url):
                raise ValueError('robots.txt disallows URL')
            if url in visited or len(visited) > MAX_REDIRECTS:
                raise ValueError('redirect loop or too many redirects')
            if original_article is not None:
                current_article = ARTICLE.fullmatch(parsed.path)
                if current_article is None or current_article[1] != original_article[1]:
                    raise ValueError('redirect changes article id or page type')
            visited.add(url)
            response = self.request(url)
            with response:
                if 300 <= response.status_code < 400:
                    location = response.headers.get('Location')
                    if not location or not location.strip():
                        raise ValueError('redirect has no location')
                    url = urljoin(url, location)
                    continue
                response.encoding = 'utf-8'
                return FetchedPage(response.text, url)

    # повторяем только временные сбои; каждый переход также выдерживает паузу
    def request(self, url: str) -> requests.Response:
        # делаем до трёх попыток с паузами между запросами
        for attempt in range(3):
            time.sleep(max(0, self.delay - (time.monotonic() - self.last)))
            self.last = time.monotonic()
            try:
                r = self.session.get(url, timeout=(10, 60), allow_redirects=False)
                # при ограничении доступа останавливаем сбор
                if r.status_code in (401, 403):
                    r.close()
                    raise PermissionError(f'HTTP {r.status_code}: stopping, see log')
                # при перегрузке или ошибке сервера увеличиваем паузу
                if r.status_code == 429 or r.status_code >= 500:
                    r.close()
                    if attempt == 2:
                        raise PermissionError(f'HTTP {r.status_code}: retry later')
                    retry = r.headers.get('Retry-After', '')
                    time.sleep(max(30 * (2 ** attempt), float(retry) if retry.isdigit() else 0))
                    continue
                try:
                    r.raise_for_status()
                except requests.HTTPError:
                    r.close()
                    raise
                return r
            except (requests.Timeout, requests.ConnectionError):
                if attempt == 2:
                    raise
                time.sleep(10 * (2 ** attempt))
        raise RuntimeError('request failed')

    # загружаем правила обхода и учитываем заданную сайтом задержку
    def load_robots(self) -> None:
        # читаем правила сайта и учитываем интервал запросов
        robots = RobotFileParser()
        robots.parse(self.get(BASE + '/robots.txt').html.splitlines())
        self.robots = robots
        self.delay = max(self.delay, float(robots.crawl_delay(AGENT) or 0))


def retry_redirect_tasks(db: sqlite3.Connection) -> int:
    """возвращаем в очередь только старые ошибки перехода на ту же статью хабра."""
    restored = 0
    prefix = 'unexpected redirect to '
    rows = db.execute("SELECT seq, url, error FROM queue WHERE kind='article' AND state='failed'")
    for seq, url, error in rows.fetchall():
        if not isinstance(url, str) or not isinstance(error, str) or not error.startswith(prefix):
            continue
        source = article_url(url)
        target = article_url(error.removeprefix(prefix))
        if source is None or target is None:
            continue
        source_match = ARTICLE.fullmatch(urlsplit(source).path)
        target_match = ARTICLE.fullmatch(urlsplit(target).path)
        assert source_match is not None and target_match is not None
        if source_match[1] != target_match[1]:
            continue
        # сохранённую статью повторно скачивать не нужно; адрес задачи сохраняем для уникальности
        saved = db.execute('SELECT 1 FROM articles WHERE id=?', (int(source_match[1]),)).fetchone()
        state = 'done' if saved else 'pending'
        db.execute('UPDATE queue SET state=?, error=NULL WHERE seq=?', (state, seq))
        restored += not bool(saved)
    db.commit()
    return restored


# собираем актуальную выгрузку из статей, уже сохранённых в базе
def export(db: sqlite3.Connection, output: Path) -> None:
    # выгружаем статьи: одна строка jsonl на документ
    temporary = output.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        for row in db.execute('SELECT payload FROM articles ORDER BY id'):
            stream.write(cast(str, row[0]) + '\n')
    # заменяем итоговый файл после завершения записи
    temporary.replace(output)


# связываем очередь ссылок, загрузку страниц, разбор статей и запись результатов
def main() -> int:
    # читаем настройки запуска
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=8000, help='target total, including saved articles')
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--delay', type=float, default=10)
    parser.add_argument('--feed', action='append', help='Habr article feed URL; repeatable')
    parser.add_argument('--save-html', action='store_true', help='also save gzip-compressed article HTML')
    parser.add_argument('--export-only', action='store_true',
                        help='выгрузить сохранённые статьи в jsonl без сбора новых')
    parser.add_argument('--retry-redirects', action='store_true',
                        help='вернуть в очередь статьи, пропущенные из-за старой ошибки перенаправления')
    args: CliArgs = parser.parse_args(namespace=CliArgs())
    if args.limit < 1 or args.delay < 0:
        parser.error('limit must be positive and delay nonnegative')
    # создаём папку данных и настраиваем журнал
    args.data_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(args.data_dir / 'crawl.log')])
    # открываем базу со статьями и очередью ссылок
    db: sqlite3.Connection = sqlite3.connect(args.data_dir / 'corpus.sqlite3')
    # articles хранит документы, queue хранит ссылки и состояние их обработки
    db.executescript('''
    CREATE TABLE IF NOT EXISTS articles(id INTEGER PRIMARY KEY, payload TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS queue(seq INTEGER PRIMARY KEY, url TEXT UNIQUE NOT NULL,
        kind TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'pending', error TEXT);
    ''')
    if args.export_only:
        # выгрузка создаётся только по явной команде и не нужна для продолжения сбора
        try:
            export(db, args.data_dir / 'articles.jsonl')
        finally:
            db.close()
        return 0

    if args.retry_redirects:
        logging.info('Restored redirect tasks: %s', retry_redirect_tasks(db))

    # добавляем новую задачу; уникальность url в базе защищает от дублей
    def enqueue(url: str, kind: TaskKind) -> None:
        # добавляем ссылку, если её ещё нет в очереди
        db.execute('INSERT OR IGNORE INTO queue(url,kind) VALUES (?,?)', (url, kind))

    # начинаем с указанных лент или общей ленты статей
    feeds: list[str] = args.feed or [BASE + '/ru/articles/']
    for feed in feeds:
        path = urlsplit(feed)
        if (path.scheme != 'https' or path.netloc != 'habr.com'
                or not re.fullmatch(r'/ru/(?:hubs/[^/]+/)?articles/', path.path) or path.query):
            parser.error('feed must be https://habr.com/ru/articles/ or a hub articles feed')
        enqueue(feed, 'listing')
    db.commit()
    total: int = cast(int, db.execute('SELECT count(*) FROM articles').fetchone()[0])
    fetcher = Fetcher(args.delay)
    try:
        if total < args.limit:
            fetcher.load_robots()
        # собираем статьи до общего лимита с учётом уже сохранённых
        while total < args.limit:
            # сначала обрабатываем статьи, затем страницы лент
            task = cast(TaskRow | None, db.execute("SELECT seq,url,kind FROM queue WHERE state='pending' ORDER BY CASE kind WHEN 'article' THEN 0 ELSE 1 END,seq LIMIT 1").fetchone())
            if not task:
                logging.warning('Queue exhausted at %s articles (target %s)', total, args.limit)
                break
            seq, url, kind = task
            try:
                page_data = fetcher.get(url)
                html = page_data.html
                if kind == 'listing':
                    # добавляем статьи из ленты и следующую страницу
                    links, soup = article_links(html)
                    if not links:
                        raise ValueError('no article links; check listing layout')
                    for link in links:
                        enqueue(link, 'article')
                    current = re.search(r'/page(\d+)/$', page_data.url)
                    page = int(current[1]) if current else 1
                    root = re.sub(r'page\d+/$', '', page_data.url)
                    next_url = root + f'page{page + 1}/'
                    if any(urljoin(BASE, string_attr(a, 'href')) == next_url for a in soup.select('a[href]')):
                        enqueue(next_url, 'listing')
                    logging.info('Listing: %s (%s links)', url, len(links))
                else:
                    # разбираем статью и проверяем, не сохранена ли она
                    doc, hubs = parse_article(html, page_data.url)
                    if not db.execute('SELECT 1 FROM articles WHERE id=?', (doc['id'],)).fetchone():
                        # при необходимости сохраняем сжатый исходный html
                        if args.save_html:
                            folder = args.data_dir / 'html'
                            folder.mkdir(exist_ok=True)
                            with gzip.open(folder / f"{doc['id']}.html.gz", 'wt', encoding='utf-8') as raw:
                                raw.write(html)
                        # сохраняем статью и подтверждаем запись перед обработкой следующей ссылки
                        payload = json.dumps(doc, ensure_ascii=False)
                        db.execute('INSERT INTO articles VALUES (?,?)', (doc['id'], payload))
                        db.commit()
                        total += 1
                        logging.info('Saved %s/%s: %s', total, args.limit, doc['title'])
                    # расширяем обход лентами найденных хабов
                    for hub in hubs:
                        enqueue(hub, 'listing')
                # помечаем успешно обработанную ссылку
                db.execute("UPDATE queue SET state='done',error=NULL WHERE seq=?", (seq,))
                db.commit()
            except PermissionError:
                raise
            except (requests.RequestException, ValueError) as exc:
                # при сбое сети оставляем задачу для следующего запуска
                if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
                    raise
                db.execute("UPDATE queue SET state='failed',error=? WHERE seq=?", (str(exc), seq))
                db.commit()
                logging.warning('%s: %s', url, exc)
    except KeyboardInterrupt:
        logging.info('Stopped; run the same command to resume')
    except (PermissionError, requests.RequestException) as exc:
        logging.error('Stopped; task remains pending: %s', exc)
        return 1
    finally:
        # закрываем базу даже при остановке; прогресс уже сохранён в sqlite
        db.close()
    logging.info('Corpus: %s articles, %s', total, args.data_dir / 'corpus.sqlite3')
    return 0 if total >= args.limit else 2


# при прямом запуске выполняем команду; при импорте этот блок пропускается
if __name__ == '__main__':
    raise SystemExit(main())
