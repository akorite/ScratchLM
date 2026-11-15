import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ArtifactTests(unittest.TestCase):
    def test_shipped_projects_regenerate_to_their_recorded_hashes(self):
        recorded = json.loads((ROOT / 'fixtures/artifact-checksums.json').read_text())['artifacts']
        self.assertEqual(len(recorded), 2)
        for relative, expected in recorded.items():
            with self.subTest(artifact=relative):
                shipped = ROOT / relative
                self.assertTrue(shipped.exists(), f'{relative} is missing')
                self.assertEqual(hashlib.sha256(shipped.read_bytes()).hexdigest(), expected,
                                 f'{relative} does not match its recorded hash')
                flags = ['--text', '--benchmark'] if 'prompt-to-text' in relative else []
                with tempfile.TemporaryDirectory() as temp:
                    regenerated = Path(temp) / shipped.name
                    run = subprocess.run([sys.executable, str(ROOT / 'tools/generate_project.py'),
                                          str(regenerated), *flags], capture_output=True, text=True)
                    self.assertEqual(run.returncode, 0, run.stderr)
                    self.assertEqual(hashlib.sha256(regenerated.read_bytes()).hexdigest(), expected)


if __name__ == '__main__':
    unittest.main()
