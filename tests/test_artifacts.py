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

    def test_visible_monitors_have_writers(self):
        """Every visible variable monitor must be written somewhere in the project."""
        import zipfile
        for relative, flags in (('dist/bonsai-prompt-to-text.sb3', ['--text', '--benchmark']),):
            project = json.loads(zipfile.ZipFile(ROOT / relative).read('project.json'))
            monitored = {m['params']['VARIABLE'] for m in project['monitors'] if m.get('visible')}
            writers = set()
            for target in project['targets']:
                for block in (target.get('blocks') or {}).values():
                    if isinstance(block, dict) and block.get('opcode', '').startswith('data_'):
                        variable = block.get('fields', {}).get('VARIABLE')
                        if variable:
                            writers.add(variable[0])
            self.assertTrue(monitored <= writers,
                            f'monitors with no writer: {sorted(monitored - writers)}')


if __name__ == '__main__':
    unittest.main()
