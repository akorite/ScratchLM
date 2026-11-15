import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('tokenizer_tables', ROOT / 'tools/tokenizer_tables.py')
tables_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tables_module)


class BpeTableTests(unittest.TestCase):
    def test_byte_alphabet_is_injective(self):
        alphabet = tables_module.bytes_to_unicode()
        self.assertEqual(len(alphabet), 256)
        self.assertEqual(len(set(alphabet.values())), 256)
        self.assertEqual(alphabet[0x20], 'Ġ')

    @unittest.skipUnless(os.environ.get('BONSAI_MODEL_PACKAGE'), 'set BONSAI_MODEL_PACKAGE to a package')
    def test_bpe_reproduces_the_pinned_tokenizer(self):
        tables = tables_module.build(tables_module.read_package(Path(os.environ['BONSAI_MODEL_PACKAGE'])))
        encode = lambda text: tables_module.bpe(tables, [tables['byte_token'][b] for b in text.encode('utf-8')])
        # Token ids captured from the pinned build's own tokenizer, one chunk each.
        for text, expected in (('hello', [14556]), (' world', [1814]), ('Hello', [9419]),
                               ('你好', [109266]), ('🙂', [169171])):
            with self.subTest(text=text):
                self.assertEqual(encode(text), expected)

    @unittest.skipUnless(os.environ.get('BONSAI_MODEL_PACKAGE'), 'set BONSAI_MODEL_PACKAGE to a package')
    def test_merge_order_follows_rank_not_key(self):
        tables = tables_module.build(tables_module.read_package(Path(os.environ['BONSAI_MODEL_PACKAGE'])))
        self.assertEqual(tables['merge_keys'], sorted(tables['merge_keys']))
        self.assertEqual(len(set(tables['merge_rank'])), len(tables['merge_rank']))


if __name__ == '__main__':
    unittest.main()
