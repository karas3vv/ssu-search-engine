"""проверяем метрики и запуск оценки на синтетическом индексе без api mws."""
from __future__ import annotations

import csv
import io
import tempfile
import unittest
from contextlib import closing, redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from qdrant_client import QdrantClient, models

from habr_search.ingestion.chunking import ChunkingConfig
from habr_search.embeddings import DEFAULT_MODEL, DEFAULT_URL, EmbeddingAPIError
from habr_search.evaluation.evaluate import (
    DEFAULT_QUERIES, QUERY_COLUMNS, REPORT_COLUMNS, PRE_SCORE_COLUMNS, EvaluationQuery, ReportContext, ReportEntry,
    evaluate_queries, load_queries, load_report, main, prepare_queries, query_set_hash,
    report_row, score_ranking, summarize, validate_expected_articles,
)
from habr_search.ingestion.index import VECTOR_SIZE, collection_name, ensure_collection
from habr_search.retrieval.search import build_query


# простой токенизатор для тестов считает символы вместо токенов настоящей модели
class CharacterTokenizer:
    # прибавляем два условных специальных токена к числу символов
    def count(self, text: str) -> int:
        return len(text) + 2

    # считаем каждый символ отдельным токеном с известными границами
    def offsets(self, text: str) -> list[tuple[int, int]]:
        return [(i, i + 1) for i in range(len(text))]


# подставной клиент хранит историю вызовов и позволяет имитировать сбой модели
class FakeEmbeddings:
    # сохраняем параметры подставного клиента и готовим историю вызовов
    def __init__(self, vector_size: int = 3, fail_on_call: int | None = None) -> None:
        self.calls: list[list[str]] = []
        self.vector_size: int = vector_size
        self.fail_on_call: int | None = fail_on_call
        self.closed: bool = False

    # возвращаем подставные векторы и при заданной настройке имитируем ошибку
    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        if len(self.calls) == self.fail_on_call:
            raise EmbeddingAPIError('синтетический сбой')
        return [[1.0] + [0.0] * (self.vector_size - 1) for _ in texts]

    # отмечаем освобождение клиента, чтобы проверить закрытие ресурсов
    def close(self) -> None:
        self.closed = True


# сохраняем временный csv с контрольными запросами для проверки команды
def write_queries(path: Path, rows: list[list[str]]) -> None:
    with path.open('w', encoding='utf-8', newline='') as target:
        writer = csv.writer(target)
        writer.writerow(QUERY_COLUMNS)
        writer.writerows(rows)


# добавляем синтетическую точку в локальную тестовую коллекцию
def add_point(store: QdrantClient, point_id: int, article_id: int, vector: list[float]) -> None:
    store.upsert('test', points=[models.PointStruct(id=point_id, vector=vector, payload={
        'article_id': article_id, 'title': f'статья {article_id}',
        'url': f'https://habr.com/ru/articles/{article_id}/', 'text': 'тестовый фрагмент',
    })])


# группа проверок модуля; каждый метод test_ описывает отдельный ожидаемый сценарий
class EvaluationTests(unittest.TestCase):
    # для продолжения нужен согласованный отчёт с теми же настройками и разметкой
    def test_report_resume_rejects_different_settings_and_corrupt_results(self) -> None:
        case = EvaluationQuery('one', 'exact', 'запрос', (1,))
        context = ReportContext('test', 1, query_set_hash([case]), False, 5)
        entry = ReportEntry('2026-10-19T09:00:00+00:00', score_ranking(case, [1], 5))
        row = report_row(entry, context)
        variants: list[tuple[str, str | int | float]] = [
            ('collection', 'other'), ('collection_points', 2), ('top_k', 3),
            ('query_set_sha256', 'other'), ('raw_query', 'True'), ('query_id', 'unknown'),
            ('query', 'другой запрос'), ('category', 'typo'), ('relevant_article_ids', '2'),
            ('first_relevant_rank', 2), ('reciprocal_rank', 0.5), ('reciprocal_rank', 'nan'),
            ('retrieved_article_ids', '1;1'), ('retrieved_article_ids', 'abc'),
            ('started_at', '2026-10-19T09:00:00'),
        ]
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'report.csv'
            for column, value in variants:
                changed = row.copy()
                changed[REPORT_COLUMNS.index(column)] = value
                with source.open('w', encoding='utf-8', newline='') as output:
                    writer = csv.writer(output)
                    writer.writerow(REPORT_COLUMNS)
                    writer.writerow(changed)
                before = source.read_bytes()
                with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                    load_report(source, [case], context)
                self.assertEqual(before, source.read_bytes())

    # допускаем отчёт без готовых строк и корректно сохраняем отсутствие попаданий
    def test_report_resume_accepts_empty_header_and_valid_misses(self) -> None:
        case = EvaluationQuery('one', 'exact', 'запрос', (1,))
        context = ReportContext('test', 1, query_set_hash([case]), False, 5)
        entry = ReportEntry('2026-10-19T09:00:00+00:00', replace(score_ranking(case, [], 5), scores=()))
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'report.csv'
            with source.open('w', encoding='utf-8', newline='') as output:
                csv.writer(output).writerow(REPORT_COLUMNS)
            self.assertEqual(load_report(source, [case], context), {})
            with source.open('a', encoding='utf-8', newline='') as output:
                csv.writer(output).writerow(report_row(entry, context))
            self.assertEqual(load_report(source, [case], context), {'one': entry})
            with source.open('a', encoding='utf-8', newline='') as output:
                csv.writer(output).writerow(report_row(entry, context))
            with self.assertRaises(ValueError):
                load_report(source, [case], context)

    # новые отчёты сохраняют оценки, старые читаются без выдумывания недостающих чисел
    def test_report_scores_roundtrip_legacy_and_corruption(self) -> None:
        case = EvaluationQuery('one', 'exact', 'запрос', (1,))
        context = ReportContext('test', 2, query_set_hash([case]), False, 5)
        entry = ReportEntry('2026-10-19T09:00:00+00:00',
                            replace(score_ranking(case, [1, 2], 5), scores=(0.8, 0.3)))
        row = report_row(entry, context)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'report.csv'
            for columns in [REPORT_COLUMNS, PRE_SCORE_COLUMNS]:
                with path.open('w', encoding='utf-8', newline='') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(columns)
                    writer.writerow(row[:len(columns)])
                result = load_report(path, [case], context)['one'].result
                self.assertEqual(result.scores, (0.8, 0.3) if columns == REPORT_COLUMNS else None)
                with self.assertRaises(ValueError):
                    load_report(path, [case], replace(context, min_score=0.5))
            for invalid in ['0.8', 'nan;0.3', '1.1;0.3']:
                changed = row.copy()
                changed[REPORT_COLUMNS.index('retrieved_scores')] = invalid
                with path.open('w', encoding='utf-8', newline='') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(REPORT_COLUMNS)
                    writer.writerow(changed)
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    load_report(path, [case], context)

    # оценка с порогом записывает пустой результат и не повторяет платный запрос
    def test_evaluation_threshold_records_scores_and_uses_one_call(self) -> None:
        case = EvaluationQuery('find', 'exact', 'запрос', (1,))
        backend = FakeEmbeddings()
        with closing(QdrantClient(':memory:')) as store:
            ensure_collection(store, 'test', 3)
            add_point(store, 1, 1, [0.0, 1.0, 0.0])
            results = list(evaluate_queries(prepare_queries([case], CharacterTokenizer()),
                                           store, 'test', backend, 5, 3, min_score=0.5))
            self.assertEqual(results[0].retrieved_ids, ())
            self.assertEqual(results[0].scores, ())
            self.assertEqual(len(backend.calls), 1)

    # проверяем метрики при разных местах, нескольких подходящих статьях и пропусках
    def test_metrics_handle_misses_cutoff_and_multiple_relevant_articles(self) -> None:
        case = EvaluationQuery('one', 'exact', 'запрос', (10, 20))
        first = score_ranking(case, [20, 30, 10], 3)
        second = score_ranking(case, [30, 10, 20], 3)
        missed = score_ranking(case, [30, 40, 10], 2)
        empty = score_ranking(case, [], 3)
        self.assertEqual(first.first_relevant_rank, 1)
        self.assertEqual(second.first_relevant_rank, 2)
        self.assertIsNone(missed.first_relevant_rank)
        self.assertEqual(missed.retrieved_ids, (30, 40))
        self.assertEqual(empty.reciprocal_rank, 0.0)
        summary = summarize([first, second, missed, empty])
        self.assertEqual(summary.hits, 2)
        self.assertEqual(summary.hit_rate, 0.5)
        self.assertEqual(summary.mrr, 0.375)
        with self.assertRaises(ValueError):
            score_ranking(case, [10, 10], 3)
        with self.assertRaises(ValueError):
            summarize([])

    # чтение csv сохраняет кавычки и несколько id, а отпечаток отражает изменения набора
    def test_csv_supports_quotes_and_multiple_relevant_ids(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'queries.csv'
            write_queries(path, [['case', 'paraphrase', 'как заполнить "шаблон", автоматически?', '10;20']])
            cases = load_queries(path)
            self.assertEqual(cases[0].relevant_ids, (10, 20))
            self.assertEqual(cases[0].query, 'как заполнить "шаблон", автоматически?')
            self.assertEqual(query_set_hash(cases), query_set_hash(load_queries(path)))
            different = [EvaluationQuery('case', 'paraphrase', 'другой запрос', (10, 20))]
            self.assertNotEqual(query_set_hash(cases), query_set_hash(different))

    # повторы запросов и ошибочные id отклоняются до оценки
    def test_csv_rejects_duplicate_queries_identifiers_and_invalid_ids(self) -> None:
        variants: list[list[list[str]]] = [
            [], [['case', 'exact', '', '1']], [['case', 'exact', 'запрос', '0']],
            [['case', 'exact', 'запрос', '1;1']], [['case', 'exact', 'запрос', 'True']],
            [['case', 'exact', 'один', '1'], ['case', 'exact', 'два', '2']],
            [['one', 'exact', 'Запрос', '1'], ['two', 'exact', 'запрос', '2']],
        ]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'queries.csv'
            for rows in variants:
                write_queries(path, rows)
                with self.subTest(rows=rows), self.assertRaises(ValueError):
                    load_queries(path)

    # инструкция добавляется в нужном режиме, а слишком длинный запрос отклоняется заранее
    def test_preflight_rejects_long_queries_and_checks_raw_prompt(self) -> None:
        case = EvaluationQuery('short', 'exact', 'запрос', (1,))
        prepared = prepare_queries([case], CharacterTokenizer())
        self.assertEqual(prepared[0][1], build_query(case.query))
        self.assertEqual(prepare_queries([case], CharacterTokenizer(), True)[0][1], case.query)
        long = EvaluationQuery('long', 'exact', 'я' * 1100, (1,))
        with self.assertRaisesRegex(ValueError, 'long'):
            prepare_queries([case, long], CharacterTokenizer())

    # оценка использует реальную группировку статей на подставных векторах
    def test_evaluation_reuses_grouped_search_and_does_not_modify_index(self) -> None:
        case = EvaluationQuery('find', 'paraphrase', 'запрос', (2,))
        backend = FakeEmbeddings()
        with closing(QdrantClient(location=':memory:')) as store:
            ensure_collection(store, 'test', 3)
            add_point(store, 1, 1, [1.0, 0.0, 0.0])
            add_point(store, 2, 1, [0.99, 0.01, 0.0])
            add_point(store, 3, 2, [0.8, 0.2, 0.0])
            validate_expected_articles(store, 'test', [case])
            results = list(evaluate_queries(prepare_queries([case], CharacterTokenizer()), store, 'test', backend, 2, 3))
            self.assertEqual(results[0].retrieved_ids, (1, 2))
            self.assertEqual(results[0].first_relevant_rank, 2)
            self.assertEqual(backend.calls, [[build_query(case.query)]])
            self.assertEqual(store.count('test', exact=True).count, 3)
            missing = EvaluationQuery('missing', 'exact', 'другой', (99,))
            with self.assertRaisesRegex(ValueError, '99'):
                validate_expected_articles(store, 'test', [case, missing])

    # пробная оценка проверяет только csv без подключения, скачиваний и записи отчёта
    def test_dry_run_does_not_connect_prompt_download_or_write_report(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'results.csv'
            argv = ['evaluate.py', '--queries', str(DEFAULT_QUERIES), '--output', str(output), '--dry-run']
            with (patch('sys.argv', argv), redirect_stdout(io.StringIO()) as stdout,
                  patch('habr_search.evaluation.evaluate.connect_store', side_effect=AssertionError('qdrant не нужен')),
                  patch('habr_search.evaluation.evaluate.load_tokenizer', side_effect=AssertionError('скачивание не нужно')),
                  patch('habr_search.evaluation.evaluate.read_api_key', side_effect=AssertionError('ключ не нужен')),
                  patch('habr_search.evaluation.evaluate.EmbeddingsClient', side_effect=AssertionError('mws не нужен'))):
                self.assertEqual(main(), 0)
            self.assertIn('контрольных запросов: 18', stdout.getvalue())
            self.assertFalse(output.exists())

    # недопустимое ожидание отклоняется ещё при проверке аргументов
    def test_invalid_timeout_is_rejected_before_loading_or_connecting(self) -> None:
        for value in ['0', '-1', 'nan', 'inf']:
            with (self.subTest(value=value), patch('sys.argv', ['evaluate.py', '--timeout', value]),
                  patch('habr_search.evaluation.evaluate.load_queries', side_effect=AssertionError('csv не нужен')),
                  patch('habr_search.evaluation.evaluate.connect_store', side_effect=AssertionError('qdrant не нужен')),
                  redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error):
                main()
            self.assertEqual(error.exception.code, 2)

    # готовый отчёт не перезаписывается при повторном выборе того же пути
    def test_existing_report_is_preserved_before_api_access(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'results.csv'
            output.write_text('предыдущий отчёт', encoding='utf-8')
            argv = ['evaluate.py', '--queries', str(DEFAULT_QUERIES), '--output', str(output)]
            with (patch('sys.argv', argv), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()),
                  patch('habr_search.evaluation.evaluate.connect_store', side_effect=AssertionError('подключение не нужно')),
                  patch('habr_search.evaluation.evaluate.read_api_key', side_effect=AssertionError('ключ не нужен')),
                  self.assertRaises(SystemExit) as error):
                main()
            self.assertEqual(error.exception.code, 1)
            self.assertEqual(output.read_text(encoding='utf-8'), 'предыдущий отчёт')

    # отсутствие ожидаемой статьи обнаруживается до ввода ключа и создания отчёта
    def test_missing_expected_article_stops_before_prompt_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as folder, closing(QdrantClient(location=':memory:')) as store:
            source = Path(folder) / 'queries.csv'
            output = Path(folder) / 'report.csv'
            write_queries(source, [['missing', 'exact', 'запрос', '99']])
            name = collection_name(DEFAULT_URL, DEFAULT_MODEL, ChunkingConfig())
            ensure_collection(store, name)
            store.upsert(name, points=[models.PointStruct(
                id=1, vector=[1.0] + [0.0] * (VECTOR_SIZE - 1), payload={'article_id': 1},
            )])
            argv = ['evaluate.py', '--queries', str(source), '--output', str(output),
                    '--qdrant-url', 'http://localhost:6333']
            with (patch('sys.argv', argv), patch.dict('os.environ', {}, clear=True),
                  patch('habr_search.evaluation.evaluate.connect_store', return_value=store), redirect_stdout(io.StringIO()),
                  redirect_stderr(io.StringIO()) as stderr,
                  patch('habr_search.evaluation.evaluate.read_api_key', side_effect=AssertionError('ключ не нужен')),
                  patch('habr_search.evaluation.evaluate.EmbeddingsClient', side_effect=AssertionError('mws не нужен')),
                  self.assertRaises(SystemExit) as error):
                main()
            self.assertEqual(error.exception.code, 1)
            self.assertIn('99', stderr.getvalue())
            self.assertFalse(output.exists())

    # сбой оставляет завершённые строки, а итоговая сводка появляется только после полного запуска
    def test_cli_preserves_completed_rows_on_failure_and_summarizes_only_success(self) -> None:
        for failure in [None, 2]:
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as folder:
                source = Path(folder) / 'queries.csv'
                output = Path(folder) / 'report.csv'
                write_queries(source, [
                    ['one', 'exact', 'первый запрос', '1'],
                    ['two', 'paraphrase', 'второй запрос', '1'],
                ])
                backend = FakeEmbeddings(VECTOR_SIZE, failure)
                with closing(QdrantClient(location=':memory:')) as store:
                    name = collection_name(DEFAULT_URL, DEFAULT_MODEL, ChunkingConfig())
                    ensure_collection(store, name)
                    store.upsert(name, points=[models.PointStruct(
                        id=1, vector=[1.0] + [0.0] * (VECTOR_SIZE - 1), payload={
                            'article_id': 1, 'title': 'тест', 'url': 'https://example.com/1', 'text': 'текст',
                        },
                    )])
                    argv = ['evaluate.py', '--queries', str(source), '--output', str(output),
                            '--qdrant-url', 'http://localhost:6333']
                    with (patch('sys.argv', argv), patch.dict('os.environ', {}, clear=True),
                          patch('habr_search.evaluation.evaluate.connect_store', return_value=store),
                          patch('habr_search.evaluation.evaluate.load_tokenizer', return_value=CharacterTokenizer()),
                          patch('habr_search.evaluation.evaluate.read_api_key', return_value='test-key'),
                          patch('habr_search.evaluation.evaluate.EmbeddingsClient', return_value=backend),
                          redirect_stdout(io.StringIO()) as stdout, redirect_stderr(io.StringIO())):
                        if failure is None:
                            self.assertEqual(main(), 0)
                        else:
                            with self.assertRaises(SystemExit) as error:
                                main()
                            self.assertEqual(error.exception.code, 1)
                with output.open(encoding='utf-8', newline='') as report:
                    rows = list(csv.DictReader(report))
                self.assertTrue(backend.closed)
                self.assertEqual(len(rows), 2 if failure is None else 1)
                self.assertEqual(rows[0]['first_relevant_rank'], '1')
                self.assertEqual(rows[0]['retrieved_article_ids'], '1')
                self.assertEqual(rows[0]['collection_points'], '1')
                self.assertNotIn('test-key', output.read_text(encoding='utf-8'))
                if failure is None:
                    self.assertIn('MRR@5: 1.0000', stdout.getvalue())
                else:
                    self.assertNotIn('MRR@5:', stdout.getvalue())

    # продолжение пропускает готовые запросы и сохраняет старые отчёты и время проверок
    def test_cli_resumes_without_reembedding_completed_queries(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'queries.csv'
            storage = Path(folder) / 'qdrant'
            first = Path(folder) / 'first.csv'
            second = Path(folder) / 'second.csv'
            third = Path(folder) / 'third.csv'
            write_queries(source, [
                ['one', 'exact', 'первый запрос', '1'],
                ['two', 'paraphrase', 'второй запрос', '1'],
            ])
            name = collection_name(DEFAULT_URL, DEFAULT_MODEL, ChunkingConfig())
            with closing(QdrantClient(path=str(storage))) as store:
                ensure_collection(store, name)
                store.upsert(name, points=[models.PointStruct(
                    id=1, vector=[1.0] + [0.0] * (VECTOR_SIZE - 1), payload={
                        'article_id': 1, 'title': 'тест', 'url': 'https://example.com/1', 'text': 'текст',
                    },
                )])
            failed_backend = FakeEmbeddings(VECTOR_SIZE, 2)
            resumed_backend = FakeEmbeddings(VECTOR_SIZE)
            phases: list[tuple[Path, Path | None, FakeEmbeddings | None, int]] = [
                (first, None, failed_backend, 1), (second, first, resumed_backend, 0),
                (third, second, None, 0),
            ]
            contents: dict[Path, bytes] = {}
            for output, resume, backend, expected_code in phases:
                argv = ['evaluate.py', '--queries', str(source), '--output', str(output),
                        '--qdrant-url', 'http://localhost:6333', '--timeout', '420']
                if resume is not None:
                    argv.extend(['--resume', str(resume)])
                client = QdrantClient(path=str(storage))
                with (patch('sys.argv', argv), patch.dict('os.environ', {}, clear=True),
                      patch('habr_search.evaluation.evaluate.connect_store', return_value=client),
                      patch('habr_search.evaluation.evaluate.load_tokenizer', return_value=CharacterTokenizer()) as tokenizer,
                      patch('habr_search.evaluation.evaluate.read_api_key', return_value='test-key') as key,
                      patch('habr_search.evaluation.evaluate.EmbeddingsClient', return_value=backend) as factory,
                      redirect_stdout(io.StringIO()) as stdout, redirect_stderr(io.StringIO()) as stderr):
                    if expected_code == 0:
                        self.assertEqual(main(), 0)
                        self.assertIn('MRR@5: 1.0000', stdout.getvalue())
                    else:
                        with self.assertRaises(SystemExit) as error:
                            main()
                        self.assertEqual(error.exception.code, expected_code)
                        self.assertIn('two: синтетический сбой', stderr.getvalue())
                    if backend is None:
                        key.assert_not_called()
                        factory.assert_not_called()
                        tokenizer.assert_not_called()
                    else:
                        self.assertEqual(factory.call_args.args[0].timeout_seconds, 420.0)
                for old, content in contents.items():
                    self.assertEqual(old.read_bytes(), content)
                contents[output] = output.read_bytes()
            self.assertEqual(resumed_backend.calls, [[build_query('второй запрос')]])
            with first.open(encoding='utf-8', newline='') as report:
                original_row = next(csv.DictReader(report))
            for output in [second, third]:
                with output.open(encoding='utf-8', newline='') as report:
                    rows = list(csv.DictReader(report))
                self.assertEqual(len(rows), 2)
                self.assertEqual(rows[0], original_row)


# при прямом запуске выполняем тесты этого файла
if __name__ == '__main__':
    unittest.main()
