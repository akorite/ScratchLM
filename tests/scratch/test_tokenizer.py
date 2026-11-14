"""Check the Scratch tokenizer against ids captured from the pinned tokenizer."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]

CASES = [
    ('hello world', [14556, 1814]),
    (' Hello', [21251]),
    ('line1\nline2', [1021, 16, 198, 1021, 17]),
    ('CASE case', [39034, 1105]),
    ("it's fine", [275, 579, 6699]),
    ('你好,世界', [109266, 11, 96748]),
    ('🙂 emoji', [169171, 40958]),
    ('  two  spaces  ', [220, 1330, 220, 12258, 256]),
    ('tab\there', [5999, 197, 6527]),
    ('123 4567', [16, 17, 18, 220, 19, 20, 21, 22]),
]


@unittest.skipUnless(os.environ.get('BONSAI_TURBOWARP_VM') and os.environ.get('BONSAI_MODEL_PACKAGE'),
                     'set BONSAI_TURBOWARP_VM and BONSAI_MODEL_PACKAGE')
class ScratchTokenizerTests(unittest.TestCase):
    def test_fixture_classes_match_the_pinned_tokenizer(self):
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp) / 'tokenizer.sb3'
            generated = subprocess.run([sys.executable, str(ROOT / 'tools/generate_project.py'), str(project),
                                        '--text', '--benchmark'], capture_output=True, text=True)
            self.assertEqual(generated.returncode, 0, generated.stderr)
            env = dict(os.environ, BONSAI_PROMPTS=json.dumps([text for text, _ in CASES]),
                       BONSAI_TIMEOUT_MS='180000')
            run = subprocess.run(['node', str(ROOT / 'tests/scratch/run-vm.js'), str(project), '--tokenize', '22'],
                                 env=env, capture_output=True, text=True, timeout=600)
            self.assertEqual(run.returncode, 0, run.stderr)
            actual = json.loads(run.stdout)['prompts']
            self.assertEqual([row['prompt'] for row in actual], [text for text, _ in CASES])
            for row, (text, expected) in zip(actual, CASES):
                with self.subTest(text=text):
                    self.assertEqual(row['tokens'], expected)


    def test_detokenizer_round_trips_the_fixture_text(self):
        tokens = [424, 579, 220, 109266, 26594]
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp) / 'detok.sb3'
            generated = subprocess.run([sys.executable, str(ROOT / 'tools/generate_project.py'), str(project),
                                        '--text', '--benchmark'], capture_output=True, text=True)
            self.assertEqual(generated.returncode, 0, generated.stderr)
            env = dict(os.environ, BONSAI_OUT_TOKENS=json.dumps(tokens), BONSAI_TIMEOUT_MS='240000',
                       BONSAI_DEBUG_VARS='BS text')
            run = subprocess.run(['node', str(ROOT / 'tests/scratch/run-vm.js'), str(project), '--detokenize', '23'],
                                 env=env, capture_output=True, text=True, timeout=600)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(json.loads(run.stdout)['BS text'], " it's 你好 🙂")


if __name__ == '__main__':
    unittest.main()
