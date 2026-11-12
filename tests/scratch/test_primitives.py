import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / 'tools' / 'generate_project.py'
RUNNER = ROOT / 'tests' / 'scratch' / 'run-vm.js'


class PrimitiveProjectTests(unittest.TestCase):
    def test_reproducible_project(self):
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp) / 'a.sb3', Path(temp) / 'b.sb3'
            for path in (a, b):
                result = subprocess.run([sys.executable, str(GENERATOR), str(path)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(a.read_bytes(), b.read_bytes())

    @unittest.skipUnless(os.environ.get('BONSAI_TURBOWARP_VM'), 'set BONSAI_TURBOWARP_VM to the pinned VM checkout')
    def test_compiled_fp16_edges(self):
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp) / 'primitives.sb3'
            generated = subprocess.run([sys.executable, str(GENERATOR), str(project)], capture_output=True, text=True)
            self.assertEqual(generated.returncode, 0, generated.stderr)
            words = [0, 32768, 1, 1023, 1024, 15360, 48128, 31743, 31744, 32256]
            result = subprocess.run(['node', str(RUNNER), str(project), json.dumps(words)], capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), [0, 0, 2**-24, 1023 * 2**-24, 2**-14, 1, -1, 65504, 'nonfinite', 'nonfinite'])

    @unittest.skipUnless(os.environ.get('BONSAI_TURBOWARP_VM'), 'set BONSAI_TURBOWARP_VM to the pinned VM checkout')
    def test_compiled_q8_matches_native_group_and_zero(self):
        source = (ROOT / 'fixtures/native/layer3-q-input.f32').read_bytes()
        native = (ROOT / 'fixtures/native/layer3-q-input.q8_k').read_bytes()
        inputs = list(struct.unpack('<2048f', source))
        expected_native = []
        for offset in range(0, len(native), 292):
            expected_native.append(struct.unpack('<f', native[offset:offset + 4])[0])
            expected_native.extend(struct.unpack('<256b', native[offset + 4:offset + 260]))
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp) / 'primitives.sb3'
            generated = subprocess.run([sys.executable, str(GENERATOR), str(project)], capture_output=True, text=True)
            self.assertEqual(generated.returncode, 0, generated.stderr)
            ties = [127, -127, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5] + [0] * 248
            tied_values = [-127, 127, 0, -2, -2, 0, 2, 2] + [0] * 248
            negative_first = [-127, 127, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5] + [0] * 248
            negative_values = [-127, 127, 0, 2, 2, 0, -2, -2] + [0] * 248
            cases = ((inputs, expected_native), ([0] * 256, [0] * 257),
                     (ties, [-1] + tied_values), (negative_first, [1] + negative_values))
            for data, expected in cases:
                result = subprocess.run(['node', str(RUNNER), str(project), json.dumps(data), '1'], capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr)
                actual = json.loads(result.stdout)
                self.assertEqual(len(actual), len(expected))
                self.assertEqual(struct.pack('<f', actual[0]), struct.pack('<f', expected[0]))
                self.assertEqual(actual[1:], expected[1:])

    @unittest.skipUnless(os.environ.get('BONSAI_TURBOWARP_VM'), 'set BONSAI_TURBOWARP_VM to the pinned VM checkout')
    def test_compiled_f32_rounds_ties_and_subnormals(self):
        inputs = [0, 1 + 2**-24, 1 + 3 * 2**-24, -(1 + 2**-24),
                  2**-150, 3 * 2**-150, 2**-126 - 2**-150, 2**127]
        expected = [struct.unpack('<f', struct.pack('<f', value))[0] for value in inputs]
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp) / 'primitives.sb3'
            generated = subprocess.run([sys.executable, str(GENERATOR), str(project)], capture_output=True, text=True)
            self.assertEqual(generated.returncode, 0, generated.stderr)
            result = subprocess.run(['node', str(RUNNER), str(project), json.dumps(inputs), '3'], capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), expected)


if __name__ == '__main__':
    unittest.main()
