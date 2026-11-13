import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(os.environ.get('BONSAI_TURBOWARP_VM') and os.environ.get('BONSAI_MODEL_PACKAGE'),
                     'set BONSAI_TURBOWARP_VM and BONSAI_MODEL_PACKAGE')
class StreamedHeadTests(unittest.TestCase):
    def test_split_tiles_match_real_native_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp) / 'head.sb3'
            generated = subprocess.run([sys.executable, str(ROOT / 'tools/generate_project.py'), str(project), '--benchmark'],
                                       capture_output=True, text=True)
            self.assertEqual(generated.returncode, 0, generated.stderr)
            run = subprocess.run(['node', str(ROOT / 'tests/scratch/run-vm.js'), str(project), '--head', '5',
                                  'blk.3.attn_q.weight', '2'], capture_output=True, text=True, timeout=60)
            self.assertEqual(run.returncode, 0, run.stderr)
            actual = json.loads(run.stdout)
            expected = struct.unpack('<5f', (ROOT / 'fixtures/native/layer3-q-first5.f32').read_bytes())
            self.assertEqual(len(actual), 5)
            for got, want in zip(actual, expected):
                self.assertLess(abs(got - want), 1e-6)

    def test_wide_transformed_rows_match_native(self):
        env = dict(os.environ, BONSAI_ACTIVATION_F32=str(ROOT / 'fixtures/native/layer0-ffn-down-input.f32'))
        expected = struct.unpack('<2048f', (ROOT / 'fixtures/native/layer0-ffn-down-output.f32').read_bytes())
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp) / 'wide.sb3'
            generated = subprocess.run([sys.executable, str(ROOT / 'tools/generate_project.py'), str(project), '--benchmark'],
                                       capture_output=True, text=True)
            self.assertEqual(generated.returncode, 0, generated.stderr)
            run = subprocess.run(['node', str(ROOT / 'tests/scratch/run-vm.js'), str(project), '--head', '5',
                                  'blk.0.ffn_down.weight', '2'], env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(run.returncode, 0, run.stderr)
            actual = json.loads(run.stdout)
            self.assertEqual(len(actual), 5)
            for got, want in zip(actual, expected[:5]):
                self.assertLess(abs(got - want), 1e-6)

    def test_stateful_holdout_projection_tolerances(self):
        cases = [('q3', 4096, 'blk.3.attn_q.weight', 2e-6, 4e-7),
                 ('ffn0', 2048, 'blk.0.ffn_down.weight', 2e-7, 4e-8)]
        fixtures = ROOT / 'fixtures/native/operators'
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp) / 'holdout.sb3'
            generated = subprocess.run([sys.executable, str(ROOT / 'tools/generate_project.py'), str(project), '--benchmark'],
                                       capture_output=True, text=True)
            self.assertEqual(generated.returncode, 0, generated.stderr)
            for name, rows, tensor, max_abs, max_rms in cases:
                with self.subTest(tensor=tensor):
                    env = dict(os.environ, BONSAI_ACTIVATION_F32=str(fixtures / f'{name}-1-input.f32'))
                    run = subprocess.run(['node', str(ROOT / 'tests/scratch/run-vm.js'), str(project), '--head',
                                          str(rows), tensor, '120'], env=env, capture_output=True, text=True, timeout=60)
                    self.assertEqual(run.returncode, 0, run.stderr)
                    actual = json.loads(run.stdout)
                    native = struct.unpack(f'<{rows}f', (fixtures / f'{name}-1-output.f32').read_bytes())
                    self.assertEqual(len(actual), rows)
                    errors = [a - b for a, b in zip(actual, native)]
                    self.assertLessEqual(max(map(abs, errors)), max_abs)
                    self.assertLessEqual(math.sqrt(math.fsum(x*x for x in errors) / rows), max_rms)


if __name__ == '__main__':
    unittest.main()
