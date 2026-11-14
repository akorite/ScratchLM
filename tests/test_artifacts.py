import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ArtifactTests(unittest.TestCase):
    def test_shipped_project_regenerates_to_the_recorded_hash(self):
        recorded = json.loads((ROOT / 'fixtures/artifact-checksums.json').read_text())['artifacts']
        relative, expected = next(iter(recorded.items()))
        shipped = ROOT / relative
        self.assertTrue(shipped.exists(), f'{relative} is missing')
        self.assertEqual(hashlib.sha256(shipped.read_bytes()).hexdigest(), expected,
                         f'{relative} does not match its recorded hash')
        with tempfile.TemporaryDirectory() as temp:
            regenerated = Path(temp) / shipped.name
            run = subprocess.run([sys.executable, str(ROOT / 'tools/generate_project.py'), str(regenerated)],
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(hashlib.sha256(regenerated.read_bytes()).hexdigest(), expected)


if __name__ == '__main__':
    unittest.main()
