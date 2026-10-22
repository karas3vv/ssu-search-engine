"""проверяем ссылки, перенаправления и восстановление очереди без запросов к хабру."""
from __future__ import annotations

import json
import sqlite3
import unittest
from io import BytesIO
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from urllib.robotparser import RobotFileParser

import requests
from urllib3.response import HTTPResponse

from habr_search.ingestion.scrape import (
    BASE, MAX_REDIRECTS, Fetcher, FetchedPage, article_links, main, parse_article, retry_redirect_tasks,
)


# создаём ответ сайта в памяти; настоящая сеть в этих тестах не используется
def response(status: int, location: str | None = None, html: str = '') -> requests.Response:
    result = requests.Response()
    result.status_code = status
    result.raw = HTTPResponse(body=BytesIO(html.encode('utf-8')), preload_content=False)
    # чтение буфера повторяет получение полного тела requests и позволяет корректно закрыть ответ
    _ = result.content
    if location is not None:
        result.headers['Location'] = location
    return result


# готовим временный корпус и задачи, чтобы не обращаться к настоящим статьям
def create_corpus(data: Path, article_ids: tuple[int, ...] = ()) -> None:
    with closing(sqlite3.connect(data / 'corpus.sqlite3')) as db:
        db.executescript('''
            CREATE TABLE articles(id INTEGER PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE queue(seq INTEGER PRIMARY KEY, url TEXT UNIQUE NOT NULL,
                kind TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'pending', error TEXT);
        ''')
        db.executemany("INSERT INTO queue(url, kind) VALUES (?, 'article')",
                       [(BASE + f'/ru/articles/{article_id}/',) for article_id in article_ids])
        db.commit()


# проверяем полный переход от ссылки компании до сохранённого документа
class ScrapeTests(unittest.TestCase):
    # после остановки продолжаем по sqlite, не создавая jsonl ни в одном из запусков
    def test_crawl_resumes_after_interruption_without_jsonl(self) -> None:
        html = '<article><h1>пример</h1><div class="article-formatted-body">полный текст</div></article>'
        with TemporaryDirectory() as temporary:
            data = Path(temporary)
            create_corpus(data, (1, 2))
            arguments = ['scrape', '--data-dir', str(data), '--limit', '2']
            with (patch('sys.argv', arguments), patch('habr_search.ingestion.scrape.Fetcher') as factory,
                  patch('habr_search.ingestion.scrape.logging.basicConfig'),
                  patch('habr_search.ingestion.scrape.logging.FileHandler')):
                factory.return_value.get.side_effect = [FetchedPage(html, BASE + '/ru/articles/1/'), KeyboardInterrupt()]
                self.assertEqual(main(), 2)
                self.assertFalse((data / 'articles.jsonl').exists())
                with closing(sqlite3.connect(data / 'corpus.sqlite3')) as db:
                    self.assertEqual(db.execute('SELECT id FROM articles').fetchall(), [(1,)])
                    self.assertEqual(db.execute("SELECT state FROM queue WHERE kind='article' ORDER BY seq").fetchall(),
                                     [('done',), ('pending',)])
                factory.return_value.get.side_effect = None
                factory.return_value.get.return_value = FetchedPage(html, BASE + '/ru/articles/2/')
                self.assertEqual(main(), 0)
            self.assertFalse((data / 'articles.jsonl').exists())
            with closing(sqlite3.connect(data / 'corpus.sqlite3')) as db:
                self.assertEqual(db.execute('SELECT id FROM articles ORDER BY id').fetchall(), [(1,), (2,)])
                self.assertEqual(db.execute("SELECT state FROM queue WHERE kind='article' ORDER BY seq").fetchall(),
                                 [('done',), ('done',)])

    # обычный сбор не перезаписывает ранее сделанную пользовательскую выгрузку
    def test_crawl_leaves_existing_jsonl_unchanged(self) -> None:
        html = '<article><h1>пример</h1><div class="article-formatted-body">полный текст</div></article>'
        with TemporaryDirectory() as temporary:
            data = Path(temporary)
            create_corpus(data, (1,))
            output = data / 'articles.jsonl'
            output.write_text('сохранённая выгрузка\n', encoding='utf-8')
            with (patch('sys.argv', ['scrape', '--data-dir', str(data), '--limit', '1']),
                  patch('habr_search.ingestion.scrape.Fetcher') as factory,
                  patch('habr_search.ingestion.scrape.logging.basicConfig'),
                  patch('habr_search.ingestion.scrape.logging.FileHandler')):
                factory.return_value.get.return_value = FetchedPage(html, BASE + '/ru/articles/1/')
                self.assertEqual(main(), 0)
            self.assertEqual(output.read_text(encoding='utf-8'), 'сохранённая выгрузка\n')
            with closing(sqlite3.connect(data / 'corpus.sqlite3')) as db:
                self.assertEqual(db.execute('SELECT count(*) FROM articles').fetchone(), (1,))

    # явный экспорт восстанавливает выгрузку по порядку id без загрузки страниц
    def test_export_only_creates_jsonl_without_fetching(self) -> None:
        with TemporaryDirectory() as temporary:
            data = Path(temporary)
            create_corpus(data)
            documents = [{'id': 2, 'text': 'второй'}, {'id': 1, 'text': 'первый'}]
            with closing(sqlite3.connect(data / 'corpus.sqlite3')) as db:
                db.executemany('INSERT INTO articles VALUES (?, ?)',
                               [(item['id'], json.dumps(item, ensure_ascii=False)) for item in documents])
                db.commit()
            output = data / 'articles.jsonl'
            output.write_text('старая выгрузка\n', encoding='utf-8')
            with (patch('sys.argv', ['scrape', '--data-dir', str(data), '--export-only']),
                  patch('habr_search.ingestion.scrape.Fetcher', side_effect=AssertionError('сеть не нужна')),
                  patch('habr_search.ingestion.scrape.logging.basicConfig'),
                  patch('habr_search.ingestion.scrape.logging.FileHandler')):
                self.assertEqual(main(), 0)
            exported = [json.loads(line) for line in output.read_text(encoding='utf-8').splitlines()]
            self.assertEqual(exported, list(reversed(documents)))
            self.assertFalse((data / 'articles.tmp').exists())
            with closing(sqlite3.connect(data / 'corpus.sqlite3')) as db:
                self.assertEqual(db.execute('SELECT count(*) FROM articles').fetchone(), (2,))

    # пути компаний сохраняются, ссылки на чужие сайты отбрасываются, id не дублируются
    def test_listing_keeps_company_urls_and_rejects_foreign_links(self) -> None:
        html = ''.join(f'<a class="tm-title__link" href="{url}">статья</a>' for url in (
            '/ru/articles/123/', '/ru/companies/example/articles/123/?utm_source=feed',
            '/ru/articles/123/', 'https://other.example/ru/articles/456/',
            'http://habr.com/ru/articles/789/', 'https://[', '/ru/articles/999/',
        ))
        links, _ = article_links(html)
        self.assertEqual(links, [BASE + '/ru/companies/example/articles/123/', BASE + '/ru/articles/999/'])

    # после перехода документ получает конечную корпоративную ссылку с прежним id
    def test_redirect_preserves_final_url_and_waits_before_each_request(self) -> None:
        html = '<article><h1>пример</h1><div class="article-formatted-body">полный текст</div></article>'
        fetcher = Fetcher(10)
        with (closing(fetcher.session), patch('habr_search.ingestion.scrape.time.sleep') as sleep,
              patch.object(fetcher.session, 'get', side_effect=[
                  response(301, '/ru/companies/example/articles/123/'), response(200, html=html),
              ]) as get):
            page = fetcher.get(BASE + '/ru/articles/123/')
            document, _ = parse_article(page.html, page.url)
            self.assertEqual(document['id'], 123)
            self.assertEqual(document['url'], BASE + '/ru/companies/example/articles/123/')
            self.assertEqual(document['text'], 'полный текст')
            self.assertEqual(get.call_count, 2)
            self.assertEqual(sleep.call_count, 2)
            self.assertEqual(get.call_args.kwargs['allow_redirects'], False)

    # проверяем цель до второго запроса, чтобы не уйти с хабра или на другую статью
    def test_redirect_rejects_unsafe_targets_before_fetching_them(self) -> None:
        for target in ('https://other.example/ru/articles/123/', 'http://habr.com/ru/articles/123/',
                       '/ru/articles/456/', '/login/', '', None):
            fetcher = Fetcher(10)
            with (self.subTest(target=target), closing(fetcher.session),
                  patch('habr_search.ingestion.scrape.time.sleep'),
                  patch.object(fetcher.session, 'get', return_value=response(302, target)) as get):
                with self.assertRaises(ValueError):
                    fetcher.get(BASE + '/ru/articles/123/')
                self.assertEqual(get.call_count, 1)

    # правила сайта проверяются и для конечной корпоративной страницы
    def test_redirect_obeys_robots_for_destination(self) -> None:
        fetcher = Fetcher(10)
        fetcher.robots = RobotFileParser()
        fetcher.robots.parse(['User-agent: *', 'Disallow: /ru/companies/'])
        with (closing(fetcher.session), patch('habr_search.ingestion.scrape.time.sleep'),
              patch.object(fetcher.session, 'get', return_value=response(
                  301, '/ru/companies/example/articles/123/',
              )) as get):
            with self.assertRaisesRegex(ValueError, 'robots'):
                fetcher.get(BASE + '/ru/articles/123/')
            self.assertEqual(get.call_count, 1)

    # циклические и слишком длинные цепочки не создают бесконечный обход
    def test_redirect_loop_and_limit_are_bounded(self) -> None:
        chains = [
            [response(301, '/ru/companies/example/articles/123/'), response(301, '/ru/articles/123/')],
            [response(301, f'/ru/companies/company{i}/articles/123/') for i in range(MAX_REDIRECTS + 1)],
        ]
        for chain in chains:
            fetcher = Fetcher(10)
            with (self.subTest(length=len(chain)), closing(fetcher.session),
                  patch('habr_search.ingestion.scrape.time.sleep'),
                  patch.object(fetcher.session, 'get', side_effect=chain) as get):
                with self.assertRaisesRegex(ValueError, 'redirect'):
                    fetcher.get(BASE + '/ru/articles/123/')
                self.assertEqual(get.call_count, len(chain))

    # повтор временного сбоя не меняет адрес и не отменяет ограничение частоты
    def test_server_error_retries_same_page(self) -> None:
        fetcher = Fetcher(10)
        with (closing(fetcher.session), patch('habr_search.ingestion.scrape.time.sleep'),
              patch.object(fetcher.session, 'get', side_effect=[response(503), response(200, html='готово')]) as get):
            self.assertEqual(fetcher.get(BASE + '/ru/articles/123/').html, 'готово')
            self.assertEqual(get.call_count, 2)
            self.assertEqual(get.call_args_list[0], get.call_args_list[1])

    # восстанавливаем только подходящие ошибки, не меняя сохранённые документы и чужие задачи
    def test_retry_restores_only_same_article_redirects_and_is_idempotent(self) -> None:
        with closing(sqlite3.connect(':memory:')) as db:
            db.executescript('''
                CREATE TABLE articles(id INTEGER PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE queue(seq INTEGER PRIMARY KEY, url TEXT, kind TEXT, state TEXT, error TEXT);
                INSERT INTO articles VALUES (2, 'сохранённый документ');
            ''')
            rows = [
                (1, BASE + '/ru/articles/1/', 'article', 'failed',
                 'unexpected redirect to ' + BASE + '/ru/companies/example/articles/1/'),
                (2, BASE + '/ru/articles/2/', 'article', 'failed',
                 'unexpected redirect to ' + BASE + '/ru/companies/example/articles/2/'),
                (3, BASE + '/ru/articles/3/', 'article', 'failed', 'HTTP 404'),
                (4, BASE + '/ru/articles/4/', 'article', 'failed',
                 'unexpected redirect to https://other.example/ru/articles/4/'),
                (5, BASE + '/ru/articles/5/', 'article', 'failed',
                 'unexpected redirect to ' + BASE + '/ru/articles/6/'),
            ]
            db.executemany('INSERT INTO queue VALUES (?, ?, ?, ?, ?)', rows)
            self.assertEqual(retry_redirect_tasks(db), 1)
            self.assertEqual(db.execute('SELECT seq, state FROM queue ORDER BY seq').fetchall(),
                             [(1, 'pending'), (2, 'done'), (3, 'failed'), (4, 'failed'), (5, 'failed')])
            self.assertEqual(retry_redirect_tasks(db), 0)
            self.assertEqual(db.execute('SELECT * FROM articles').fetchall(), [(2, 'сохранённый документ')])


if __name__ == '__main__':
    unittest.main()
