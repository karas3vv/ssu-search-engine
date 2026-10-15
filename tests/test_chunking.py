"""проверяем разбиение без интернета и без весов модели."""
from __future__ import annotations

import hashlib
import unittest

from chunking import ChunkingConfig, chunk_article
from prepare import PreparedArticle


# вместо модели используем прозрачный счётчик: один символ равен одному токену
class CharacterTokenizer:
    def offsets(self, text: str) -> list[tuple[int, int]]:
        return [(i, i + 1) for i in range(len(text))]

    def count(self, text: str) -> int:
        return len(text) + 2


def article(text: str, title: str = 'заголовок', article_id: int = 1) -> PreparedArticle:
    return PreparedArticle(
        id=article_id, url=f'https://habr.com/ru/articles/{article_id}/',
        title=title, text=text, author=None, published_at=None,
        hubs=('Python',), tags=(),
        text_hash=hashlib.sha256(text.encode('utf-8')).hexdigest(),
    )


class ChunkingTests(unittest.TestCase):
    def test_long_paragraph_is_covered_without_gaps(self) -> None:
        doc = article('проверяем длинный абзац 🧪 ' * 80)
        config = ChunkingConfig(max_tokens=128, overlap_tokens=12)
        chunks = list(chunk_article(doc, CharacterTokenizer(), config))
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0].start_char, 0)
        self.assertEqual(chunks[-1].end_char, len(doc.text))
        for index, chunk in enumerate(chunks):
            self.assertEqual(chunk.text, doc.text[chunk.start_char:chunk.end_char])
            self.assertLessEqual(chunk.token_count, config.max_tokens)
            self.assertEqual(chunk.chunk_index, index)
            if index:
                previous = chunks[index - 1]
                self.assertGreater(chunk.start_char, previous.start_char)
                self.assertLess(chunk.start_char, previous.end_char)

    def test_short_article_keeps_code_indentation(self) -> None:
        doc = article('пример:\n    print("привет")\n\treturn 1')
        chunks = list(chunk_article(doc, CharacterTokenizer(), ChunkingConfig()))
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, doc.text)
        self.assertTrue(chunks[0].embedding_text.startswith('passage: заголовок\n\n'))

    def test_prefers_line_boundary(self) -> None:
        doc = article('а' * 45 + '\n' + 'б' * 30 + '\n' + 'в' * 200)
        chunks = list(chunk_article(doc, CharacterTokenizer(), ChunkingConfig(128, 12)))
        self.assertTrue(chunks[0].text.endswith('\n'))

    def test_ids_are_stable_and_depend_on_settings(self) -> None:
        doc = article('текст ' * 100)
        tokenizer = CharacterTokenizer()
        original = list(chunk_article(doc, tokenizer, ChunkingConfig(128, 12)))
        repeated = list(chunk_article(doc, tokenizer, ChunkingConfig(128, 12)))
        changed = list(chunk_article(doc, tokenizer, ChunkingConfig(128, 8)))
        self.assertEqual(original, repeated)
        self.assertNotEqual(original[0].id, changed[0].id)
        self.assertEqual(len({chunk.id for chunk in original}), len(original))

    def test_long_title_does_not_displace_body_or_change_metadata(self) -> None:
        doc = article('короткий текст', title='длинный заголовок ' * 100)
        chunks = list(chunk_article(doc, CharacterTokenizer(), ChunkingConfig(128, 12)))
        self.assertEqual(chunks[0].title, doc.title)
        self.assertEqual(chunks[0].text, doc.text)
        self.assertLessEqual(chunks[0].token_count, 128)

    def test_zero_overlap_does_not_lose_characters(self) -> None:
        doc = article('слово ' * 100)
        chunks = list(chunk_article(doc, CharacterTokenizer(), ChunkingConfig(128, 0)))
        self.assertEqual(''.join(chunk.text for chunk in chunks), doc.text)

    def test_invalid_config_is_rejected(self) -> None:
        for max_tokens, overlap in [(513, 10), (63, 10), (128, -1), (128, 64)]:
            with self.subTest(max_tokens=max_tokens, overlap=overlap):
                with self.assertRaises(ValueError):
                    ChunkingConfig(max_tokens, overlap)


if __name__ == '__main__':
    unittest.main()
