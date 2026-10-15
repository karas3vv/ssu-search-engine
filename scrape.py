"""собираем статьи хабра с возможностью продолжить после остановки."""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, TypedDict, cast
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup, Tag

BASE: str = 'https://habr.com'
AGENT: str = 'HabrStudyCrawler/0.1'
ARTICLE: re.Pattern[str] = re.compile(r'^/ru/(?:articles|companies/[^/]+/articles)/(\d+)/$')
HUB: re.Pattern[str] = re.compile(r'^/ru/hubs/[^/]+/$')


TaskKind = Literal['article', 'listing']
TaskRow = tuple[int, str, TaskKind]


# структура сохраняемой статьи
class ArticleDocument(TypedDict):
    id: int
    url: str
    title: str
    text: str
    author: str | None
    published_at: str | None
    hubs: list[str]
    tags: list[str]
    fetched_at: str


# типы параметров командной строки
class CliArgs(argparse.Namespace):
    limit: int
    data_dir: Path
    delay: float
    feed: list[str] | None
    save_html: bool
    export_only: bool


def string_attr(tag: Tag, name: str) -> str:
    """получаем строковый атрибут тега и проверяем его тип."""
    value = tag.get(name)
    if not isinstance(value, str):
        raise ValueError(f'Expected string attribute: {name}')
    return value


def article_links(html: str) -> tuple[list[str], BeautifulSoup]:
    # собираем ссылки на статьи из ленты без дублей
    soup = BeautifulSoup(html, 'html.parser')
    result: dict[str, str] = {}
    for a in soup.select('a.tm-title__link[href]'):
        path = urlsplit(urljoin(BASE, string_attr(a, 'href'))).path
        match = ARTICLE.fullmatch(path)
        if match:
            result[match[1]] = f'{BASE}/ru/articles/{match[1]}/'
    return list(result.values()), soup


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
    # собираем данные для записи в базу и jsonl
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
    def __init__(self, delay: float) -> None:
        self.session: requests.Session = requests.Session()
        self.session.headers['User-Agent'] = AGENT
        self.delay: float = max(10.0, delay)
        self.last: float = time.monotonic()  # выдерживаем паузу и после перезапуска
        self.robots: RobotFileParser | None = None

    def get(self, url: str) -> str:
        # проверяем домен и разрешение в robots.txt
        if urlsplit(url).netloc != 'habr.com':
            raise ValueError('only habr.com is allowed')
        if self.robots and not self.robots.can_fetch(AGENT, url):
            raise ValueError('robots.txt disallows URL')
        # делаем до трёх попыток с паузами между запросами
        for attempt in range(3):
            time.sleep(max(0, self.delay - (time.monotonic() - self.last)))
            self.last = time.monotonic()
            try:
                r = self.session.get(url, timeout=(10, 60), allow_redirects=False)
                if 300 <= r.status_code < 400:
                    target = urljoin(url, r.headers.get('Location', ''))
                    if target == url:
                        raise ValueError('invalid redirect')
                    # неожиданный редирект считаем ошибкой
                    raise ValueError(f'unexpected redirect to {target}')
                # при ограничении доступа останавливаем сбор
                if r.status_code in (401, 403):
                    raise PermissionError(f'HTTP {r.status_code}: stopping, see log')
                # при перегрузке или ошибке сервера увеличиваем паузу
                if r.status_code == 429 or r.status_code >= 500:
                    if attempt == 2:
                        raise PermissionError(f'HTTP {r.status_code}: retry later')
                    retry = r.headers.get('Retry-After', '')
                    time.sleep(max(30 * (2 ** attempt), float(retry) if retry.isdigit() else 0))
                    continue
                r.raise_for_status()
                r.encoding = 'utf-8'
                return r.text
            except (requests.Timeout, requests.ConnectionError):
                if attempt == 2:
                    raise
                time.sleep(10 * (2 ** attempt))
        raise RuntimeError('request failed')

    def load_robots(self) -> None:
        # читаем правила сайта и учитываем интервал запросов
        robots = RobotFileParser()
        robots.parse(self.get(BASE + '/robots.txt').splitlines())
        self.robots = robots
        self.delay = max(self.delay, float(robots.crawl_delay(AGENT) or 0))


def export(db: sqlite3.Connection, output: Path) -> None:
    # выгружаем статьи: одна строка jsonl на документ
    temporary = output.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        for row in db.execute('SELECT payload FROM articles ORDER BY id'):
            stream.write(cast(str, row[0]) + '\n')
    # заменяем итоговый файл после завершения записи
    temporary.replace(output)


def main() -> int:
    # читаем настройки запуска
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=8000, help='target total, including saved articles')
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parent / 'data')
    parser.add_argument('--delay', type=float, default=10)
    parser.add_argument('--feed', action='append', help='Habr article feed URL; repeatable')
    parser.add_argument('--save-html', action='store_true', help='also save gzip-compressed article HTML')
    parser.add_argument('--export-only', action='store_true')
    args: CliArgs = parser.parse_args(namespace=CliArgs())
    if args.limit < 1 or args.delay < 0:
        parser.error('limit must be positive and delay nonnegative')
    # создаём папку данных и настраиваем журнал
    args.data_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(args.data_dir / 'crawl.log')])
    # открываем базу со статьями и очередью ссылок
    db: sqlite3.Connection = sqlite3.connect(args.data_dir / 'corpus.sqlite3')
    db.executescript('''
    CREATE TABLE IF NOT EXISTS articles(id INTEGER PRIMARY KEY, payload TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS queue(seq INTEGER PRIMARY KEY, url TEXT UNIQUE NOT NULL,
        kind TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'pending', error TEXT);
    ''')
    output = args.data_dir / 'articles.jsonl'
    export(db, output)  # восстанавливаем jsonl после возможного прерывания
    if args.export_only:
        db.close()
        return 0

    def enqueue(url: str, kind: TaskKind) -> None:
        # добавляем ссылку, если её ещё нет в очереди
        db.execute('INSERT OR IGNORE INTO queue(url,kind) VALUES (?,?)', (url, kind))

    # начинаем с указанных лент или общей ленты статей
    feeds: list[str] = args.feed or [BASE + '/ru/articles/']
    for feed in feeds:
        path = urlsplit(feed)
        if path.netloc != 'habr.com' or not re.fullmatch(r'/ru/(?:hubs/[^/]+/)?articles/', path.path) or path.query:
            parser.error('feed must be https://habr.com/ru/articles/ or a hub articles feed')
        enqueue(feed, 'listing')
    db.commit()
    total: int = cast(int, db.execute('SELECT count(*) FROM articles').fetchone()[0])
    fetcher = Fetcher(args.delay)
    try:
        if total < args.limit:
            fetcher.load_robots()
        with output.open('a', encoding='utf-8') as stream:
            # собираем статьи до общего лимита с учётом уже сохранённых
            while total < args.limit:
                # сначала обрабатываем статьи, затем страницы лент
                task = cast(TaskRow | None, db.execute("SELECT seq,url,kind FROM queue WHERE state='pending' ORDER BY CASE kind WHEN 'article' THEN 0 ELSE 1 END,seq LIMIT 1").fetchone())
                if not task:
                    logging.warning('Queue exhausted at %s articles (target %s)', total, args.limit)
                    break
                seq, url, kind = task
                try:
                    html = fetcher.get(url)
                    if kind == 'listing':
                        # добавляем статьи из ленты и следующую страницу
                        links, soup = article_links(html)
                        if not links:
                            raise ValueError('no article links; check listing layout')
                        for link in links:
                            enqueue(link, 'article')
                        current = re.search(r'/page(\d+)/$', url)
                        page = int(current[1]) if current else 1
                        root = re.sub(r'page\d+/$', '', url)
                        next_url = root + f'page{page + 1}/'
                        if any(urljoin(BASE, string_attr(a, 'href')) == next_url for a in soup.select('a[href]')):
                            enqueue(next_url, 'listing')
                        logging.info('Listing: %s (%s links)', url, len(links))
                    else:
                        # разбираем статью и проверяем, не сохранена ли она
                        doc, hubs = parse_article(html, url)
                        if not db.execute('SELECT 1 FROM articles WHERE id=?', (doc['id'],)).fetchone():
                            # при необходимости сохраняем сжатый исходный html
                            if args.save_html:
                                folder = args.data_dir / 'html'
                                folder.mkdir(exist_ok=True)
                                with gzip.open(folder / f"{doc['id']}.html.gz", 'wt', encoding='utf-8') as raw:
                                    raw.write(html)
                            # записываем документ в sqlite и добавляем строку в jsonl
                            payload = json.dumps(doc, ensure_ascii=False)
                            db.execute('INSERT INTO articles VALUES (?,?)', (doc['id'], payload))
                            db.commit()
                            stream.write(payload + '\n')
                            stream.flush()
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
        # обновляем выгрузку и закрываем базу даже при остановке
        export(db, output)
        db.close()
    logging.info('Corpus: %s articles, %s', total, output)
    return 0 if total >= args.limit else 2


if __name__ == '__main__':
    raise SystemExit(main())
