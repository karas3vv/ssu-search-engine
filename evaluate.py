"""оцениваем поиск на контрольных запросах с известными подходящими статьями."""
# поток оценки: csv запросов → выдача поиска → места подходящих статей → csv и метрики
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import math
import os
import re
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from chunking import DEFAULT_TOKENIZER, ChunkingConfig, TextTokenizer, load_tokenizer
from embeddings import DEFAULT_MODEL, DEFAULT_URL, EmbeddingAPIError, EmbeddingConfig, EmbeddingsClient, read_api_key
from index import VECTOR_SIZE, EmbeddingProvider, collection_name
from search import MAX_QUERY_TOKENS, build_query, search_articles, validate_collection
from vector_store import connect_store, resolve_qdrant_config

ROOT: Path = Path(__file__).resolve().parent
DEFAULT_QUERIES: Path = ROOT / 'evaluation' / 'queries.csv'
QUERY_COLUMNS: tuple[str, ...] = ('query_id', 'category', 'query', 'relevant_article_ids')
# порядок полей csv фиксирован: по нему проверяется совместимость старого отчёта
REPORT_COLUMNS: tuple[str, ...] = (
    'started_at', 'collection', 'collection_points', 'query_set_sha256', 'raw_query', 'top_k',
    'query_id', 'category', 'query', 'relevant_article_ids', 'retrieved_article_ids',
    'first_relevant_rank', 'reciprocal_rank',
)


# контрольный запрос и список известных подходящих статей для него
@dataclass(frozen=True)
class EvaluationQuery:
    query_id: str  # устойчивое имя проверки для сравнения разных запусков
    category: str  # exact, paraphrase, alias, typo или своя группа
    query: str  # исходный текст, который проверяем поиском
    relevant_ids: tuple[int, ...]  # известные подходящие статьи; разметка может быть неполной


# найденные статьи и место первого подходящего результата для одного запроса
@dataclass(frozen=True)
class QueryEvaluation:
    case: EvaluationQuery  # исходный запрос вместе с его разметкой
    retrieved_ids: tuple[int, ...]  # статьи в порядке выдачи, ограниченные top-k
    first_relevant_rank: int | None  # место первой подходящей статьи или none при отсутствии

    # превращаем место в число для mrr: первое место даёт 1, отсутствие даёт 0
    @property
    def reciprocal_rank(self) -> float:
        return 1.0 / self.first_relevant_rank if self.first_relevant_rank is not None else 0.0


# итоговые метрики по всем запросам или по одной категории
@dataclass(frozen=True)
class EvaluationSummary:
    count: int  # число завершённых проверок
    hits: int  # сколько запросов нашли подходящую статью в top-k
    hit_rate: float  # доля запросов с подходящей статьёй в первых k результатах
    mrr: float  # среднее обратное место первой подходящей статьи в пределах k


# настройки, которые должны совпасть при продолжении сохранённого отчёта
@dataclass(frozen=True)
class ReportContext:
    collection: str  # имя коллекции с фрагментами
    collection_points: int  # число точек во время оценки
    query_set_sha256: str  # отпечаток выбранных запросов и разметки
    raw_query: bool  # передаём запрос без добавления инструкции на стороне клиента
    top_k: int  # сколько первых статей учитывается в метриках


# результат проверки вместе с временем начала исходного запуска
@dataclass(frozen=True)
class ReportEntry:
    started_at: str  # время начала запуска, в котором получен результат
    result: QueryEvaluation  # сохранённая выдача и рассчитанное место


# читаем csv с разметкой и проверяем обязательные поля, id и отсутствие повторов
def load_queries(path: Path) -> list[EvaluationQuery]:
    cases: list[EvaluationQuery] = []
    # отдельно отслеживаем повторы имён проверок и текстов запросов
    identifiers: set[str] = set()
    queries: set[str] = set()
    with path.open(encoding='utf-8-sig', newline='') as source:
        rows = csv.reader(source, strict=True)
        header = next(rows, None)
        if header is None or tuple(header) != QUERY_COLUMNS:
            raise ValueError('заголовок csv должен быть: ' + ','.join(QUERY_COLUMNS))
        for line_number, row in enumerate(rows, 2):
            if not any(value.strip() for value in row):
                continue
            if len(row) != len(QUERY_COLUMNS):
                raise ValueError(f'строка {line_number}: ожидалось {len(QUERY_COLUMNS)} поля')
            identifier, category, query, ids = (value.strip() for value in row)
            if (not re.fullmatch(r'[a-z0-9][a-z0-9_-]*', identifier)
                    or not re.fullmatch(r'[a-z][a-z0-9_-]*', category) or not query):
                raise ValueError(f'строка {line_number}: проверьте query_id, category и query')
            parts = [value.strip() for value in ids.split(';')]
            if any(not re.fullmatch(r'[1-9][0-9]*', value) for value in parts):
                raise ValueError(f'строка {line_number}: нужны положительные id статей через ;')
            relevant = tuple(int(value) for value in parts)
            if len(set(relevant)) != len(relevant):
                raise ValueError(f'строка {line_number}: id подходящей статьи повторяется')
            if identifier in identifiers or query.casefold() in queries:
                raise ValueError(f'строка {line_number}: повторяется query_id или текст запроса')
            identifiers.add(identifier)
            queries.add(query.casefold())
            cases.append(EvaluationQuery(identifier, category, query, relevant))
    if not cases:
        raise ValueError('набор контрольных запросов пуст')
    return cases


# определяем место первой подходящей статьи только в пределах top-k
def score_ranking(case: EvaluationQuery, article_ids: Sequence[int], top_k: int) -> QueryEvaluation:
    if top_k < 1:
        raise ValueError('top-k должен быть положительным')
    retrieved = tuple(article_ids[:top_k])
    if any(type(value) is not int or value < 1 for value in retrieved):
        raise ValueError('выдача должна содержать положительные id статей')
    if len(set(retrieved)) != len(retrieved):
        raise ValueError('выдача содержит повторные статьи')
    # места начинаются с единицы; первый подходящий id определяет оценку запроса
    rank = next((i for i, value in enumerate(retrieved, 1) if value in case.relevant_ids), None)
    return QueryEvaluation(case, retrieved, rank)


# усредняем попадания и обратные места завершённых проверок
def summarize(results: Sequence[QueryEvaluation]) -> EvaluationSummary:
    if not results:
        raise ValueError('нет завершённых проверок для расчёта метрик')
    hits = sum(result.first_relevant_rank is not None for result in results)
    return EvaluationSummary(
        len(results), hits, hits / len(results),
        sum(result.reciprocal_rank for result in results) / len(results),
    )


# добавляем инструкцию модели и проверяем длину всех ещё не выполненных запросов
def prepare_queries(
    cases: Sequence[EvaluationQuery], tokenizer: TextTokenizer, raw_query: bool = False,
) -> list[tuple[EvaluationQuery, str]]:
    prepared: list[tuple[EvaluationQuery, str]] = []
    for case in cases:
        text = build_query(case.query, raw_query)
        if tokenizer.count(text) > MAX_QUERY_TOKENS:
            raise ValueError(f'{case.query_id}: запрос вместе с инструкцией превышает {MAX_QUERY_TOKENS} токенов')
        prepared.append((case, text))
    return prepared


# убеждаемся, что ожидаемые статьи уже есть в поисковом индексе
def validate_expected_articles(store: QdrantClient, name: str, cases: Sequence[EvaluationQuery]) -> None:
    missing: list[int] = []
    for article_id in sorted({value for case in cases for value in case.relevant_ids}):
        count = store.count(name, exact=True, count_filter=models.Filter(must=[
            models.FieldCondition(key='article_id', match=models.MatchValue(value=article_id)),
        ])).count
        if count == 0:
            missing.append(article_id)
    if missing:
        raise ValueError('в индексе отсутствуют ожидаемые статьи: ' + ', '.join(map(str, missing)))


# последовательно выполняем запросы и выдаём оценки их результатов
def evaluate_queries(
    prepared: Sequence[tuple[EvaluationQuery, str]], store: QdrantClient, name: str,
    embeddings: EmbeddingProvider, top_k: int = 5, vector_size: int = VECTOR_SIZE,
) -> Iterator[QueryEvaluation]:
    if top_k < 1:
        raise ValueError('top-k должен быть положительным')
    for case, text in prepared:
        # каждый запрос векторизуется один раз, статьи повторно не обрабатываются
        try:
            vector = embeddings.embed([text])[0]
        except EmbeddingAPIError as exc:
            raise EmbeddingAPIError(f'{case.query_id}: {exc}') from None
        found = search_articles(store, name, vector, top_k, vector_size)
        yield score_ranking(case, [result.article_id for result in found], top_k)


# фиксируем состав, порядок запросов и разметку одним отпечатком
def query_set_hash(cases: Sequence[EvaluationQuery]) -> str:
    buffer = io.StringIO(newline='')
    writer = csv.writer(buffer)
    for case in cases:
        writer.writerow([case.query_id, case.category, case.query, ';'.join(map(str, case.relevant_ids))])
    return hashlib.sha256(buffer.getvalue().encode()).hexdigest()


# преобразуем результат проверки в строку csv с настройками запуска
def report_row(entry: ReportEntry, context: ReportContext) -> list[str | int | float]:
    result = entry.result
    rank = result.first_relevant_rank
    return [
        entry.started_at, context.collection, context.collection_points,
        context.query_set_sha256, str(context.raw_query), context.top_k,
        result.case.query_id, result.case.category, result.case.query,
        ';'.join(map(str, result.case.relevant_ids)), ';'.join(map(str, result.retrieved_ids)),
        rank if rank is not None else '', result.reciprocal_rank,
    ]


# читаем готовые строки отчёта и проверяем их совместимость и внутреннюю согласованность
def load_report(
    path: Path, cases: Sequence[EvaluationQuery], context: ReportContext,
) -> dict[str, ReportEntry]:
    # принимаем только результаты для того же набора, режима поиска и размера индекса
    by_id = {case.query_id: case for case in cases}
    saved: dict[str, ReportEntry] = {}
    with path.open(encoding='utf-8-sig', newline='') as source:
        rows = csv.reader(source, strict=True)
        header = next(rows, None)
        if header is None or tuple(header) != REPORT_COLUMNS:
            raise ValueError('resume: неверный или отсутствующий заголовок отчёта')
        for number, values in enumerate(rows, 2):
            if len(values) != len(REPORT_COLUMNS):
                raise ValueError(f'resume: строка {number} неполная; исходный отчёт не изменён')
            row = dict(zip(REPORT_COLUMNS, values))
            identifier = row['query_id']
            case = by_id.get(identifier)
            if case is None or identifier in saved:
                raise ValueError(f'resume: неизвестный или повторный query_id {identifier}')
            # готовые проверки можно повторно использовать только с совпадающими настройками
            expected_metadata = {
                'collection': context.collection, 'collection_points': str(context.collection_points),
                'query_set_sha256': context.query_set_sha256, 'raw_query': str(context.raw_query),
                'top_k': str(context.top_k), 'category': case.category, 'query': case.query,
                'relevant_article_ids': ';'.join(map(str, case.relevant_ids)),
            }
            if any(row[key] != value for key, value in expected_metadata.items()):
                raise ValueError(f'resume: {identifier}: настройки, индекс или набор запросов отличаются от отчёта')
            started = datetime.fromisoformat(row['started_at'])
            if started.tzinfo is None:
                raise ValueError(f'resume: {identifier}: время в отчёте должно содержать часовой пояс')
            parts = row['retrieved_article_ids'].split(';') if row['retrieved_article_ids'] else []
            if len(parts) > context.top_k or any(not re.fullmatch(r'[1-9][0-9]*', value) for value in parts):
                raise ValueError(f'resume: {identifier}: неверная выдача в отчёте')
            result = score_ranking(case, [int(value) for value in parts], context.top_k)
            rank = str(result.first_relevant_rank) if result.first_relevant_rank is not None else ''
            # сверяем сохранённые метрики с выдачей, чтобы не принять повреждённую строку
            if row['first_relevant_rank'] != rank or not math.isclose(
                float(row['reciprocal_rank']), result.reciprocal_rank, rel_tol=1e-12, abs_tol=1e-12,
            ):
                raise ValueError(f'resume: {identifier}: место или метрика не совпадают с выдачей')
            saved[identifier] = ReportEntry(row['started_at'], result)
    return saved


# настройки оценки, продолжения отчёта и ожидания api
class CliArgs(argparse.Namespace):
    queries: Path
    output: Path | None
    resume: Path | None
    timeout_seconds: float
    limit: int | None
    top_k: int
    qdrant_url: str | None
    qdrant_path: Path | None
    tokenizer_file: Path
    max_tokens: int
    overlap_tokens: int
    raw_query: bool
    dry_run: bool


# проверяем входные данные, продолжаем нужные запросы, записываем отчёт и считаем метрики
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queries', type=Path, default=DEFAULT_QUERIES)
    parser.add_argument('--output', type=Path, help='новый csv-файл отчёта; существующий не перезаписывается')
    parser.add_argument('--resume', type=Path, help='взять завершённые проверки из совместимого csv-отчёта')
    parser.add_argument('--timeout', '--timeout-seconds', dest='timeout_seconds', type=float, default=300.0,
                        help='ожидание ответа MWS в секундах, по умолчанию 300')
    parser.add_argument('--limit', type=int, help='сколько первых запросов проверить')
    parser.add_argument('--top-k', type=int, default=5)
    location = parser.add_mutually_exclusive_group()
    location.add_argument('--qdrant-url')
    location.add_argument('--qdrant-path', type=Path)
    parser.add_argument('--tokenizer-file', type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--overlap-tokens', type=int, default=128)
    parser.add_argument('--raw-query', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='проверить csv без qdrant, mws и записи отчёта')
    args: CliArgs = parser.parse_args(namespace=CliArgs())
    if args.top_k < 1 or (args.limit is not None and args.limit < 1):
        parser.error('top-k и limit должны быть положительными')
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        parser.error('timeout должен быть положительным конечным числом')
    try:
        cases = load_queries(args.queries)
        if args.limit is not None:
            cases = cases[:args.limit]
        chunking = ChunkingConfig(args.max_tokens, args.overlap_tokens)
        fingerprint = query_set_hash(cases)
        print(f'контрольных запросов: {len(cases)}')
        print(f'ожидаемых статей: {len({value for case in cases for value in case.relevant_ids})}')
        if args.dry_run:
            for case in cases:
                print(f'{case.query_id} [{case.category}]: {case.query}')
            print('проверен только набор csv; наличие статей в индексе и длина запросов не проверялись')
            return 0
        started = datetime.now(timezone.utc)
        output = args.output or ROOT / 'data' / 'evaluation' / f'results-{started:%Y%m%d-%H%M%S-%f}.csv'
        if output.exists():
            raise ValueError(f'отчёт уже существует: {output}; задайте новый --output')
        if args.resume is not None and output.resolve() == args.resume.resolve():
            raise ValueError('исходный отчёт resume и новый output должны отличаться')
        config = resolve_qdrant_config(args.qdrant_url, args.qdrant_path)
        if config.path is not None and not config.path.is_dir():
            raise ValueError('локальное хранилище не найдено; задайте --qdrant-url')
        url = os.environ.get('MWS_EMBEDDINGS_URL', DEFAULT_URL)
        deployment = os.environ.get('MWS_EMBEDDINGS_MODEL', DEFAULT_MODEL)
        name = collection_name(url, deployment, chunking)
        with closing(connect_store(config)) as store:
            # весь набор проверяется до первого платного запроса
            validate_collection(store, name)
            validate_expected_articles(store, name, cases)
            point_count = store.count(name, exact=True).count
            context = ReportContext(name, point_count, fingerprint, args.raw_query, args.top_k)
            saved = load_report(args.resume, cases, context) if args.resume is not None else {}
            # оставляем для модели только запросы без готового результата
            pending = [case for case in cases if case.query_id not in saved]
            prepared = prepare_queries(pending, load_tokenizer(args.tokenizer_file), args.raw_query) if pending else []
            print(f'из отчёта: {len(saved)}; осталось запросов: {len(pending)}')
            backend: EmbeddingsClient | None = None
            embedding_config: EmbeddingConfig | None = None
            if prepared:
                embedding_config = EmbeddingConfig(read_api_key(), url, deployment, args.timeout_seconds)
            output.parent.mkdir(parents=True, exist_ok=True)
            by_id: dict[str, QueryEvaluation] = {}
            # файл и клиент закроются даже при исключении или остановке пользователем
            with ExitStack() as stack:
                report = stack.enter_context(output.open('x', encoding='utf-8', newline=''))
                if embedding_config is not None:
                    backend = stack.enter_context(closing(EmbeddingsClient(embedding_config)))
                writer = csv.writer(report)
                writer.writerow(REPORT_COLUMNS)
                print(f'коллекция: {name}; фрагментов в индексе: {point_count}')
                print(f'отчёт: {output}')
                # копируем готовые результаты в новый файл, сохраняя время исходной проверки
                for case in cases:
                    entry = saved.get(case.query_id)
                    if entry is not None:
                        writer.writerow(report_row(entry, context))
                        by_id[case.query_id] = entry.result
                report.flush()
                # если все проверки взяты из отчёта, новые векторы вообще не запрашиваются
                if backend is not None:
                    for result in evaluate_queries(prepared, store, name, backend, args.top_k):
                        rank = result.first_relevant_rank
                        writer.writerow(report_row(ReportEntry(started.isoformat(), result), context))
                        report.flush()  # при ошибке уже завершённые проверки останутся в отчёте
                        by_id[result.case.query_id] = result
                        position = str(rank) if rank is not None else f'нет в первых {args.top_k}'
                        print(f'{len(by_id)}/{len(cases)} {result.case.query_id}: место {position}')
            # при изменении индекса во время проверки сводка уже не сопоставима
            if store.count(name, exact=True).count != point_count:
                raise ValueError('число фрагментов изменилось во время оценки; повторите на неизменном индексе')
        results = [by_id[case.query_id] for case in cases]
        summary = summarize(results)
        print(f'завершено запросов: {summary.count}')
        print(f'Hit@{args.top_k}: {summary.hit_rate:.2%} ({summary.hits}/{summary.count})')
        print(f'MRR@{args.top_k}: {summary.mrr:.4f}')
        for category in sorted({result.case.category for result in results}):
            group = summarize([result for result in results if result.case.category == category])
            print(f'{category}: запросов {group.count}; Hit@{args.top_k} {group.hit_rate:.2%}; MRR@{args.top_k} {group.mrr:.4f}')
        return 0
    except KeyboardInterrupt:
        print('\nоценка остановлена; завершённые строки отчёта сохранены, итоговые метрики не рассчитаны')
        return 130
    except UnexpectedResponse as exc:
        parser.exit(1, f'qdrant вернул http {exc.status_code}; проверьте настройки сервера\n')
    except ResponseHandlingException:
        parser.exit(1, 'не удалось связаться с qdrant; проверьте адрес и контейнер\n')
    except (ValueError, RuntimeError, OSError, csv.Error) as exc:
        parser.exit(1, f'оценка не завершена: {exc}\n')


# при прямом запуске выполняем команду; при импорте этот блок пропускается
if __name__ == '__main__':
    raise SystemExit(main())
