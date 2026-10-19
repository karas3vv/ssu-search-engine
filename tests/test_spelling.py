"""проверяем исправления, неоднозначность и словарь без удалённой модели."""
from __future__ import annotations

import csv
import io
import sqlite3
import tempfile
import unittest
from contextlib import closing, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from qdrant_client import QdrantClient, models

from habr_search.evaluation.evaluate import QUERY_COLUMNS, load_queries
from habr_search.ingestion.chunking import ChunkingConfig
from habr_search.embeddings import DEFAULT_MODEL, DEFAULT_URL
from habr_search.ingestion.index import VECTOR_SIZE, collection_name, ensure_collection
from habr_search.retrieval.search import build_query, main as search_main
from habr_search.retrieval.spellcheck import build_dictionary, prepare_evaluation
from habr_search.retrieval.spelling import (
    DictionaryInfo, SpellingDictionary, dictionary_hash, edit_distance, load_dictionary,
    ngrams, normalize_word, save_dictionary, vocabulary_words,
)


# создаём небольшой словарь с известными частотами для проверки правил исправления
def dictionary(frequencies: dict[str, int] | None = None) -> SpellingDictionary:
    words = frequencies if frequencies is not None else {
        'автоматизация': 10, 'документов': 8, 'python': 10, 'docker': 7, 'елка': 3,
    }
    return SpellingDictionary(words, DictionaryInfo('test', 20, 10, dictionary_hash(words),
                                                   '2026-10-19T00:00:00+00:00'))


# проверяем безопасные замены, неоднозначность и сохранение словаря
class SpellingTests(unittest.TestCase):
    # триграммы учитывают начало и конец, а нормализация не размножает формы ё и е
    def test_ngrams_and_normalization(self) -> None:
        self.assertEqual(ngrams('кот'), frozenset(('^ко', 'кот', 'от$')))
        self.assertEqual(normalize_word('ЁЛКА'), 'елка')
        self.assertEqual(vocabulary_words('Python PYTHON python Ёлка елка'), {'python', 'елка'})
        with self.assertRaises(ValueError):
            ngrams('кот', 0)

    # вставка, пропуск, замена и соседняя перестановка считаются одной правкой
    def test_edit_operations_and_empty_strings(self) -> None:
        for word in ['автоматизаця', 'автомаатизация', 'автоматизоция', 'автмоатизация']:
            with self.subTest(word=word):
                self.assertEqual(edit_distance(word, 'автоматизация'), 1)
                self.assertEqual(edit_distance('автоматизация', word), 1)
        self.assertEqual(edit_distance('', 'кот'), 3)
        self.assertEqual(edit_distance('кот', ''), 3)
        self.assertEqual(edit_distance('кот', 'кот'), 0)
        # osa не разрешает повторно редактировать одну и ту же подстроку
        self.assertEqual(edit_distance('ca', 'abc'), 3)

    # сохраняем регистр, знаки препинания и исходную фразу при замене слова
    def test_replacements_keep_case_punctuation_and_original_query(self) -> None:
        query = '  Автоматизаця,  документов! Pyhton. '
        result = dictionary().suggest(query)
        self.assertEqual(result.original, query)
        self.assertEqual(result.corrected, '  Автоматизация,  документов! Pyhton. ')
        self.assertEqual(dictionary().suggest('Pyhton.').corrected, 'Python.')
        for item in result.suggestions:
            self.assertEqual(query[item.start:item.end], item.original)
        for word in ['автоматизаця', 'автомаатизация', 'автоматизоция', 'автмоатизация']:
            self.assertEqual(dictionary().suggest(word).corrected, 'автоматизация')

    # частота не должна превращать неоднозначное исправление в автоматическую замену
    def test_ambiguous_and_two_edit_candidates_are_only_suggestions(self) -> None:
        ambiguous = dictionary({'серверы': 10, 'сервера': 3}).suggest('серверо')
        self.assertEqual(ambiguous.corrected, 'серверо')
        self.assertEqual([item.word for item in ambiguous.suggestions[0].candidates], ['серверы', 'сервера'])
        result = dictionary().suggest('афтаматизация')
        self.assertEqual(result.corrected, result.original)
        self.assertEqual(result.suggestions[0].candidates[0].distance, 2)

    # не исправляем известные слова, короткие обозначения и защищённые технические токены
    def test_known_rare_short_and_protected_terms_are_kept(self) -> None:
        spelling = dictionary({'автоматизация': 10, 'автоматизаця': 1, 'серверы': 2,
                               'python': 10, 'елка': 3})
        for query in ['автоматизаця', 'серверо', 'Ёлка', 'SQL C++ C#', 'ворд',
                      'PYHTON', 'PyHton', 'pyhtоn', 'my_pyhton', 'pyhton3',
                      'https://pyhton.example/path', 'pyhton.py', '`pyhton`']:
            with self.subTest(query=query):
                self.assertEqual(spelling.suggest(query).corrected, query)
        self.assertTrue(spelling.suggest('серверо').suggestions)
        self.assertEqual(dictionary({'стора': 10}).suggest('репозитории Ктора').corrected, 'репозитории Ктора')

    # при одинаковых частотах порядок вариантов устойчив и лимит проверяется
    def test_candidate_order_is_deterministic_and_limits_are_checked(self) -> None:
        first = dictionary({'серверы': 3, 'сервера': 3})
        second = dictionary({'сервера': 3, 'серверы': 3})
        self.assertEqual(first.candidates('серверо'), second.candidates('серверо'))
        with self.assertRaises(ValueError):
            first.candidates('серверо', 0)
        for query in ['', '  ', 'а' * 4001]:
            with self.assertRaises(ValueError):
                first.suggest(query)

    # частоты и отпечаток проверяются при чтении, а существующая база не заменяется
    def test_dictionary_round_trip_integrity_and_read_only_loading(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'spelling.sqlite3'
            original = dictionary()
            info = save_dictionary(path, original.frequencies, 'test', 20, 10)
            before = path.read_bytes()
            loaded = load_dictionary(path)
            self.assertEqual(loaded.info, info)
            self.assertEqual(loaded.suggest('pyhton').corrected, 'python')
            self.assertEqual(path.read_bytes(), before)
            with self.assertRaises(ValueError):
                save_dictionary(path, original.frequencies, 'test', 20, 10)
            self.assertEqual(path.read_bytes(), before)
            loaded.validate_context('test', 20)
            for name, points in [('other', 20), ('test', 21)]:
                with self.assertRaises(ValueError):
                    loaded.validate_context(name, points)
            with closing(sqlite3.connect(path)) as connection, connection:
                connection.execute("UPDATE words SET article_frequency=9 WHERE word='python'")
            with self.assertRaises(ValueError):
                load_dictionary(path)
            missing = Path(folder) / 'missing.sqlite3'
            with self.assertRaises(ValueError):
                load_dictionary(missing)
            self.assertFalse(missing.exists())

    # не загружаем словарь с неверными словами, частотами или метаданными
    def test_invalid_dictionary_content_is_rejected(self) -> None:
        for words in [{}, {'слово': 0}, {'слово': 11}, {'Слово': 2}, {'слово1': 2}]:
            with self.subTest(words=words), self.assertRaises(ValueError):
                dictionary(words)

    # повторяющиеся слова и перекрывающиеся фрагменты не увеличивают число статей со словом
    def test_build_counts_articles_and_does_not_change_qdrant(self) -> None:
        with tempfile.TemporaryDirectory() as folder, closing(QdrantClient(location=':memory:')) as store:
            ensure_collection(store, 'test', 3)
            store.upsert('test', [models.PointStruct(id=i, vector=[1.0, 0.0, 0.0], payload={
                'article_id': 1 if i < 3 else 2, 'title': 'Python',
                'text': 'автоматизация автоматизация документов',
            }) for i in range(1, 4)])
            before = store.scroll('test', limit=10, with_vectors=True)
            path = Path(folder) / 'dictionary.sqlite3'
            info = build_dictionary(store, 'test', path, batch_size=1)
            self.assertEqual((info.points, info.articles), (3, 2))
            self.assertEqual(load_dictionary(path).frequencies,
                             {'python': 2, 'автоматизация': 2, 'документов': 2})
            self.assertEqual(before, store.scroll('test', limit=10, with_vectors=True))

    # неполная коллекция не должна оставлять готовый словарь на диске
    def test_build_rejects_missing_payload_without_creating_dictionary(self) -> None:
        with tempfile.TemporaryDirectory() as folder, closing(QdrantClient(location=':memory:')) as store:
            ensure_collection(store, 'test', 3)
            path = Path(folder) / 'dictionary.sqlite3'
            for name in ['missing', 'test']:
                with self.assertRaises(ValueError):
                    build_dictionary(store, name, path)
            store.upsert('test', [models.PointStruct(id=1, vector=[1.0, 0.0, 0.0], payload={'text': 'текст'})])
            with self.assertRaises(ValueError):
                build_dictionary(store, 'test', path)
            self.assertFalse(path.exists())

    # исправленный набор сохраняет разметку и не перезаписывает исходные файлы
    def test_corrected_evaluation_preserves_labels_original_and_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'queries.csv'
            output = Path(folder) / 'corrected.csv'
            with source.open('w', encoding='utf-8', newline='') as target:
                writer = csv.writer(target)
                writer.writerow(QUERY_COLUMNS)
                writer.writerow(['one', 'typo', 'автоматизаця документов', '1;2'])
                writer.writerow(['two', 'exact', 'python', '3'])
            before = source.read_bytes()
            self.assertEqual(prepare_evaluation(dictionary(), source, output), 1)
            cases = load_queries(output)
            self.assertEqual(cases[0].query, 'автоматизация документов')
            self.assertEqual((cases[0].query_id, cases[0].category, cases[0].relevant_ids), ('one', 'typo', (1, 2)))
            self.assertEqual(source.read_bytes(), before)
            saved = output.read_bytes()
            with self.assertRaises(FileExistsError):
                prepare_evaluation(dictionary(), source, output)
            self.assertEqual(output.read_bytes(), saved)

    # не создаём набор, в котором исправления превратили разные запросы в одинаковые
    def test_evaluation_collisions_are_rejected_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'queries.csv'
            output = Path(folder) / 'corrected.csv'
            with source.open('w', encoding='utf-8', newline='') as target:
                writer = csv.writer(target)
                writer.writerow(QUERY_COLUMNS)
                writer.writerow(['one', 'typo', 'pyhton', '1'])
                writer.writerow(['two', 'exact', 'python', '1'])
            with self.assertRaises(ValueError):
                prepare_evaluation(dictionary(), source, output)
            self.assertFalse(output.exists())

    # проверяем настоящий путь команды: режим предложения сохраняет исходный вход модели
    def test_search_modes_send_one_query_and_preserve_index(self) -> None:
        from tests.test_evaluate import CharacterTokenizer, FakeEmbeddings

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'dictionary.sqlite3'
            name = collection_name(DEFAULT_URL, DEFAULT_MODEL, ChunkingConfig())
            save_dictionary(path, {'python': 3}, name, 3, 3)
            for mode, expected in [('off', 'pyhton'), ('suggest', 'pyhton'), ('apply', 'python')]:
                with self.subTest(mode=mode), closing(QdrantClient(location=':memory:')) as store:
                    ensure_collection(store, name)
                    vector = [1.0] + [0.0] * (VECTOR_SIZE - 1)
                    store.upsert(name, [models.PointStruct(id=i, vector=vector, payload={
                        'article_id': i, 'title': 'Python', 'url': f'https://example.org/{i}', 'text': 'python',
                    }) for i in range(1, 4)])
                    before = store.scroll(name, limit=10, with_vectors=True)
                    embeddings = FakeEmbeddings(VECTOR_SIZE)
                    argv = ['search.py', '--query', 'pyhton', '--qdrant-url', 'http://localhost:6333',
                            '--spelling-mode', mode, '--spelling-db', str(path)]
                    with (patch.dict('os.environ', {'MWS_EMBEDDINGS_URL': DEFAULT_URL,
                                                   'MWS_EMBEDDINGS_MODEL': DEFAULT_MODEL}),
                          patch('sys.argv', argv), patch('habr_search.retrieval.search.connect_store', return_value=store),
                          patch.object(store, 'close'),
                          patch('habr_search.retrieval.search.read_api_key', return_value='synthetic-key'),
                          patch('habr_search.retrieval.search.load_tokenizer', return_value=CharacterTokenizer()),
                          patch('habr_search.retrieval.search.EmbeddingsClient', return_value=embeddings),
                          redirect_stdout(io.StringIO())):
                        self.assertEqual(search_main(), 0)
                    self.assertEqual(embeddings.calls, [[build_query(expected)]])
                    self.assertEqual(store.scroll(name, limit=10, with_vectors=True), before)

    # несовместимый словарь останавливает поиск до обращения к платной модели
    def test_wrong_dictionary_stops_search_before_model_access(self) -> None:
        with tempfile.TemporaryDirectory() as folder, closing(QdrantClient(location=':memory:')) as store:
            path = Path(folder) / 'dictionary.sqlite3'
            save_dictionary(path, {'python': 3}, 'other', 3, 3)
            argv = ['search.py', '--query', 'pyhton', '--qdrant-url', 'http://localhost:6333',
                    '--spelling-mode', 'apply', '--spelling-db', str(path)]
            with (patch('sys.argv', argv), patch('habr_search.retrieval.search.connect_store', return_value=store),
                  patch('habr_search.retrieval.search.validate_collection'), patch('habr_search.retrieval.search.read_api_key') as key,
                  patch('habr_search.retrieval.search.load_tokenizer') as tokenizer, patch('habr_search.retrieval.search.EmbeddingsClient') as embeddings,
                  redirect_stderr(io.StringIO()), self.assertRaises(SystemExit)):
                search_main()
            key.assert_not_called()
            tokenizer.assert_not_called()
            embeddings.assert_not_called()


if __name__ == '__main__':
    unittest.main()
