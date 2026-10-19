"""готовим таблицу ручной проверки выдачи по сохранённому отчёту без mws и qdrant."""
from __future__ import annotations

import argparse
import csv
import sqlite3
from contextlib import closing
from pathlib import Path

from habr_search.evaluation.evaluate import load_queries
from habr_search.evaluation.evaluate_spelling import read_complete_report
from habr_search.ingestion.prepare import DEFAULT_SOURCE, PreparationStats, PreparedArticle, iter_prepared_articles

REVIEW_COLUMNS: tuple[str, ...] = (
    'collection', 'index_revision', 'query_id', 'category', 'query', 'rank',
    'article_id', 'title', 'url', 'seed_relevant', 'text_sha256', 'relevance', 'comment',
)


def spreadsheet_text(value: str) -> str:
    """сохраняем текст статьи как текст, чтобы таблица не исполнила его как формулу."""
    return "'" + value if value.lstrip().startswith(('=', '+', '-', '@')) else value


def export_review(queries: Path, report: Path, source: Path, output: Path) -> int:
    """добавляем найденные и ожидаемые статьи, оставляя ручные оценки незаполненными."""
    if output.exists():
        raise ValueError('таблица уже существует; укажите новый --output')
    cases = load_queries(queries)
    context, results = read_complete_report(report, cases)
    needed = {value for result in results for value in (*result.case.relevant_ids, *result.retrieved_ids)}
    articles: dict[int, PreparedArticle] = {}
    # читаем корпус только для ссылок, заголовков и отпечатков; не меняем разметку автоматически
    with closing(iter_prepared_articles(source, PreparationStats())) as prepared:
        articles = {article.id: article for article in prepared if article.id in needed}
    missing = needed - articles.keys()
    if missing:
        raise ValueError('в корпусе отсутствуют статьи из отчёта: ' + ', '.join(map(str, sorted(missing))))
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open('x', encoding='utf-8', newline='') as target:
        writer = csv.writer(target)
        writer.writerow(REVIEW_COLUMNS)
        for result in results:
            ranks = {article_id: rank for rank, article_id in enumerate(result.retrieved_ids, 1)}
            ids: list[int | None] = list(dict.fromkeys((*result.retrieved_ids, *result.case.relevant_ids)))
            # пустая выдача тоже получает строку, чтобы отрицательная проверка не исчезла из таблицы
            for article_id in ids or [None]:
                article = articles[article_id] if article_id is not None else None
                writer.writerow([
                    context.collection, context.index_revision, result.case.query_id, result.case.category,
                    spreadsheet_text(result.case.query), ranks.get(article_id, '') if article_id else '',
                    article_id if article_id is not None else '',
                    spreadsheet_text(article.title) if article else '', article.url if article else '',
                    'yes' if article_id in result.case.relevant_ids else '',
                    article.text_hash if article else '', '', '',
                ])
                count += 1
    return count


# параметры чтения исходных файлов и создания отдельной таблицы проверки
class CliArgs(argparse.Namespace):
    queries: Path
    report: Path
    source: Path
    output: Path


def main() -> int:
    """читаем завершённый отчёт и создаём новую таблицу для проверяющего."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queries', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, required=True)
    args: CliArgs = parser.parse_args(namespace=CliArgs())
    try:
        count = export_review(args.queries, args.report, args.source, args.output)
        print(f'таблица проверки: {args.output}; строк: {count}')
        print('оценки relevance и comment не заполнены; обращений к mws и qdrant: 0')
        return 0
    except (ValueError, OSError, csv.Error, sqlite3.Error) as exc:
        parser.exit(1, f'не удалось подготовить проверку: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
