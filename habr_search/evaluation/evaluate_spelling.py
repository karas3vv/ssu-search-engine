"""оцениваем исправления локально и сравниваем сохранённые отчёты поиска."""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from habr_search.retrieval import spelling
from habr_search.evaluation.evaluate import (
    QUERY_COLUMNS, REPORT_COLUMNS, PRE_SCORE_COLUMNS, EvaluationQuery, QueryEvaluation, ReportContext,
    load_queries, load_report, query_set_hash, print_summary,
)
from habr_search.retrieval.spelling import DEFAULT_DICTIONARY, SpellingDictionary, load_dictionary
from habr_search.retrieval.search import validate_min_score

from habr_search.paths import PROJECT_ROOT as ROOT

DEFAULT_CASES: Path = ROOT / 'evaluation' / 'spelling_queries.csv'
CASE_COLUMNS: tuple[str, ...] = ('query_id', 'category', 'query', 'expected_query', 'relevant_article_ids')


# эталон объединяет поисковую разметку и ожидаемый текст после исправления
@dataclass(frozen=True)
class SpellingCase:
    case: EvaluationQuery
    expected_query: str


def load_cases(path: Path) -> list[SpellingCase]:
    """проверяем эталоны исправлений до чтения словаря и создания отчётов."""
    cases: list[SpellingCase] = []
    identifiers: set[str] = set()
    queries: set[str] = set()
    with path.open(encoding='utf-8-sig', newline='') as source:
        reader = csv.reader(source, strict=True)
        if tuple(next(reader, list[str]())) != CASE_COLUMNS:
            raise ValueError('неверные столбцы набора опечаток')
        for line, values in enumerate(reader, 2):
            if len(values) != len(CASE_COLUMNS):
                raise ValueError(f'строка {line}: неверное число полей')
            identifier, category, query, expected, ids = values
            parts = ids.split(';')
            if (not re.fullmatch(r'[A-Za-z0-9_-]+', identifier) or category not in ('typo', 'clean')
                    or not query.strip() or not expected.strip() or identifier in identifiers
                    or query.casefold() in queries
                    or any(not re.fullmatch(r'[1-9][0-9]*', part) for part in parts)
                    or len(set(parts)) != len(parts) or ((query == expected) != (category == 'clean'))):
                raise ValueError(f'строка {line}: неверная или повторная проверка')
            identifiers.add(identifier)
            queries.add(query.casefold())
            cases.append(SpellingCase(EvaluationQuery(identifier, category, query, tuple(map(int, parts))), expected))
    if not cases or not all(any(item.case.category == group for item in cases) for group in ('typo', 'clean')):
        raise ValueError('нужны непустые группы typo и clean')
    return cases


# сохраняем отдельный набор запросов, не перезаписывая предыдущий файл
def write_queries(path: Path, cases: Sequence[EvaluationQuery]) -> None:
    with path.open('x', encoding='utf-8', newline='') as target:
        writer = csv.writer(target)
        writer.writerow(QUERY_COLUMNS)
        for case in cases:
            writer.writerow([case.query_id, case.category, case.query, ';'.join(map(str, case.relevant_ids))])


def evaluate_local(dictionary: SpellingDictionary, cases: Sequence[SpellingCase], output: Path,
                   cases_hash: str) -> tuple[int, int, int, int]:
    """считаем полные исправления и порчу правильных запросов без обращений к модели."""
    original = [item.case for item in cases]
    corrected = [replace(item.case, query=dictionary.suggest(item.case.query).corrected) for item in cases]
    if len({case.query.casefold() for case in corrected}) != len(corrected):
        raise ValueError('исправленные запросы совпали; для сравнения нужен набор без повторов')
    typo_count = sum(item.case.category == 'typo' for item in cases)
    clean_count = len(cases) - typo_count
    fixed = sum(item.case.category == 'typo' and result.query == item.expected_query
                for item, result in zip(cases, corrected))
    damaged = sum(item.case.category == 'clean' and result.query != item.case.query
                  for item, result in zip(cases, corrected))
    # каталог нового запуска создаётся целиком отдельно от предыдущих результатов
    output.mkdir(parents=True, exist_ok=False)
    write_queries(output / 'original.csv', original)
    write_queries(output / 'corrected.csv', corrected)
    rules_hash = hashlib.sha256(Path(spelling.__file__).read_bytes()).hexdigest()
    with (output / 'local.csv').open('x', encoding='utf-8', newline='') as target:
        writer = csv.writer(target)
        writer.writerow(['query_id', 'category', 'query', 'expected_query', 'corrected_query',
                         'exact_match', 'changed', 'dictionary_sha256', 'rules_sha256', 'cases_sha256',
                         'collection', 'collection_points'])
        for item, result in zip(cases, corrected):
            writer.writerow([item.case.query_id, item.case.category, item.case.query, item.expected_query,
                             result.query, result.query == item.expected_query, result.query != item.case.query,
                             dictionary.info.sha256, rules_hash, cases_hash,
                             dictionary.info.collection, dictionary.info.points])
    return fixed, typo_count, damaged, clean_count


def read_complete_report(path: Path, cases: Sequence[EvaluationQuery]) -> tuple[ReportContext, list[QueryEvaluation]]:
    """повторно используем строгую проверку строк и метрик из evaluate.py."""
    with path.open(encoding='utf-8-sig', newline='') as source:
        reader = csv.reader(source, strict=True)
        header = tuple(next(reader, list[str]()))
        if header not in (REPORT_COLUMNS, PRE_SCORE_COLUMNS):
            raise ValueError('неверный формат поискового отчёта')
        first = next(reader, list[str]())
        if len(first) != len(header):
            raise ValueError('отчёт не содержит завершённых запросов')
        values = dict(zip(header, first))
    if values['raw_query'] not in ('True', 'False'):
        raise ValueError('неверный режим инструкции в отчёте')
    raw_score = values.get('min_score', '')
    min_score = float(raw_score) if raw_score else None
    validate_min_score(min_score)
    context = ReportContext(values['collection'], int(values['collection_points']), query_set_hash(cases),
                            values['raw_query'] == 'True', int(values['top_k']), values['index_revision'], min_score)
    if context.collection_points < 1 or context.top_k < 1 or not context.index_revision:
        raise ValueError('неверное число точек, top-k или версия индекса')
    entries = load_report(path, cases, context)
    if len(entries) != len(cases):
        raise ValueError('сравнение требует полностью завершённых отчётов')
    return context, [entries[case.query_id].result for case in cases]


def compare_reports(original_queries: Path, corrected_queries: Path,
                    original_report: Path, corrected_report: Path) -> None:
    """сравниваем места при одинаковых статьях, разметке, top-k и инструкции."""
    original = load_queries(original_queries)
    corrected = load_queries(corrected_queries)
    if ([(case.query_id, case.category, case.relevant_ids) for case in original]
            != [(case.query_id, case.category, case.relevant_ids) for case in corrected]):
        raise ValueError('наборы отличаются id, порядком, категориями или ожидаемыми статьями')
    before_context, before = read_complete_report(original_report, original)
    after_context, after = read_complete_report(corrected_report, corrected)
    if replace(before_context, query_set_sha256=after_context.query_set_sha256) != after_context:
        raise ValueError('коллекция, число точек, top-k или инструкция в отчётах отличаются')
    for label, results in [('исходные', before), ('с исправлениями', after)]:
        print_summary(results, before_context.top_k, label)
        for category in sorted({result.case.category for result in results}):
            print_summary([result for result in results if result.case.category == category],
                          before_context.top_k, f'  {category}')
    improved = worse = unchanged = 0
    for old, new in zip(before, after):
        # для no_answer улучшением считается пустая выдача, а не место ожидаемой статьи
        old_value = float(not old.retrieved_ids) if old.case.category == 'no_answer' else old.reciprocal_rank
        new_value = float(not new.retrieved_ids) if new.case.category == 'no_answer' else new.reciprocal_rank
        improved += new_value > old_value
        worse += new_value < old_value
        unchanged += new_value == old_value
        if new_value != old_value:
            if old.case.category == 'no_answer':
                print(f'{old.case.query_id}: пустая выдача {not old.retrieved_ids} → {not new.retrieved_ids}')
            else:
                print(f'{old.case.query_id}: место {old.first_relevant_rank} → {new.first_relevant_rank}')
    print(f'лучше: {improved}; хуже: {worse}; одинаково: {unchanged}')


# параметры локальной проверки и сравнения уже готовых отчётов
class CliArgs(argparse.Namespace):
    command: str
    cases: Path
    dictionary: Path
    output_dir: Path | None
    original_queries: Path
    corrected_queries: Path
    original_report: Path
    corrected_report: Path


# выбираем локальную проверку или сравнение отчётов без вызова модели
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    local = commands.add_parser('local', help='оценить исправления без mws и подготовить два набора поиска')
    local.add_argument('--cases', type=Path, default=DEFAULT_CASES)
    local.add_argument('--dictionary', type=Path, default=DEFAULT_DICTIONARY)
    local.add_argument('--output-dir', type=Path)
    compare = commands.add_parser('compare', help='сопоставить два готовых отчёта без mws')
    for name in ('original-queries', 'corrected-queries', 'original-report', 'corrected-report'):
        compare.add_argument('--' + name, type=Path, required=True)
    args: CliArgs = parser.parse_args(namespace=CliArgs())
    try:
        if args.command == 'local':
            cases = load_cases(args.cases)
            dictionary = load_dictionary(args.dictionary)
            output = args.output_dir or ROOT / 'data/evaluation' / f'spelling-{datetime.now(timezone.utc):%Y%m%d-%H%M%S-%f}'
            fixed, typos, damaged, clean = evaluate_local(
                dictionary, cases, output, hashlib.sha256(args.cases.read_bytes()).hexdigest(),
            )
            print(f'опечаток полностью исправлено: {fixed}/{typos}')
            print(f'правильных запросов изменено: {damaged}/{clean}')
            print(f'отчёт и наборы для поиска: {output}')
        else:
            compare_reports(args.original_queries, args.corrected_queries, args.original_report, args.corrected_report)
        print('обращений к mws: 0; поисковый индекс не изменялся')
        return 0
    except (ValueError, OSError, csv.Error) as exc:
        parser.exit(1, f'оценка опечаток не выполнена: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
