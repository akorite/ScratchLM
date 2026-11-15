import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(os.environ.get('BONSAI_TURBOWARP_VM'), 'set BONSAI_TURBOWARP_VM to the pinned VM checkout')
class LayerPrimitiveTests(unittest.TestCase):
    def test_partial_rope_matches_native_position_two(self):
        fixture = ROOT / 'fixtures/native/m5/attention'
        unpack = lambda path: struct.unpack('<2048f', path.read_bytes())
        inputs = unpack(fixture / 'Qcur_normed-3-pos2.f32')
        expected = unpack(fixture / 'Qcur-3-pos2.f32')
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / 'layers.sb3'
            build = subprocess.run([sys.executable, str(ROOT / 'tools/generate_project.py'), str(project)],
                                   capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stderr)
            run = subprocess.run(['node', str(ROOT / 'tests/scratch/run-vm.js'), str(project),
                                  json.dumps([1, *inputs]), '30'], capture_output=True, text=True, timeout=60)
            self.assertEqual(run.returncode, 0, run.stderr)
            actual = json.loads(run.stdout)
        self.assertEqual(len(actual), len(expected))
        self.assertLessEqual(max(abs(a - b) for a, b in zip(actual, expected)), 1e-6)
        for head in range(8):
            self.assertEqual(actual[head * 256 + 64:head * 256 + 256], list(inputs[head * 256 + 64:head * 256 + 256]))

    def test_partial_rope_matches_later_native_positions(self):
        fixture = ROOT / 'fixtures/native/m5/attention'
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / 'layers.sb3'
            build = subprocess.run([sys.executable, str(ROOT / 'tools/generate_project.py'), str(project)],
                                   capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stderr)
            for position in (16, 64):
                with self.subTest(position=position):
                    inputs = struct.unpack('<2048f', (fixture / f'Qcur_normed-3-pos{position}.f32').read_bytes())
                    expected = struct.unpack('<2048f', (fixture / f'Qcur-3-pos{position}.f32').read_bytes())
                    run = subprocess.run(['node', str(ROOT / 'tests/scratch/run-vm.js'), str(project),
                                          json.dumps([position - 1, *inputs]), '30'], capture_output=True, text=True, timeout=60)
                    self.assertEqual(run.returncode, 0, run.stderr)
                    actual = json.loads(run.stdout)
                    self.assertEqual(len(actual), len(expected))
                    self.assertLessEqual(max(abs(a - b) for a, b in zip(actual, expected)), 1e-6)

    def test_partial_rope_rejects_bad_shapes_and_preserves_position_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / 'layers.sb3'
            build = subprocess.run([sys.executable, str(ROOT / 'tools/generate_project.py'), str(project)],
                                   capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stderr)
            values = [float(i - 128) for i in range(256)]
            for payload, expected in (([0, *values], values), ([0, 1], ['invalid dimensions']),
                                      ([-1, *values], ['invalid dimensions'])):
                with self.subTest(length=len(payload), position=payload[0]):
                    run = subprocess.run(['node', str(ROOT / 'tests/scratch/run-vm.js'), str(project),
                                          json.dumps(payload), '30'], capture_output=True, text=True, timeout=60)
                    self.assertEqual(run.returncode, 0, run.stderr)
                    self.assertEqual(json.loads(run.stdout), expected)

    def test_causal_softmax_ignores_future_positions(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / 'layers.sb3'
            build = subprocess.run([sys.executable, str(ROOT / 'tools/generate_project.py'), str(project)],
                                   capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stderr)
            for payload, expected in (([1, 8, 999], [1.0]),
                                      ([3, 1, 2, 3, 999], [0.09003057, 0.24472847, 0.66524096]),
                                      ([3, 1001, 1002, 1003, 9999], [0.09003057, 0.24472847, 0.66524096]),
                                      ([4, 1, 2, 3], ['invalid dimensions'])):
                with self.subTest(payload=payload):
                    run = subprocess.run(['node', str(ROOT / 'tests/scratch/run-vm.js'), str(project),
                                          json.dumps(payload), '31'], capture_output=True, text=True, timeout=60)
                    self.assertEqual(run.returncode, 0, run.stderr)
                    actual = json.loads(run.stdout)
                    self.assertEqual(len(actual), len(expected))
                    for got, want in zip(actual, expected):
                        if isinstance(want, str):
                            self.assertEqual(got, want)
                        else:
                            self.assertAlmostEqual(got, want, delta=1e-6)

    def test_gating_and_residual_vector_operations(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / 'layers.sb3'
            build = subprocess.run([sys.executable, str(ROOT / 'tools/generate_project.py'), str(project)],
                                   capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stderr)
            cases = [(32, [3, -1, 0, 1, 2, 3, -4], [-0.53788284, 0, -2.92423431]),
                     (33, [3, 1, -2, 3, 4, 5, -6], [5, 3, -3]),
                     (32, [2, 1, 2, 3], ['invalid dimensions']),
                     (33, [0], ['invalid dimensions'])]
            for mode, payload, expected in cases:
                with self.subTest(mode=mode, payload=payload):
                    run = subprocess.run(['node', str(ROOT / 'tests/scratch/run-vm.js'), str(project),
                                          json.dumps(payload), str(mode)], capture_output=True, text=True, timeout=60)
                    self.assertEqual(run.returncode, 0, run.stderr)
                    actual = json.loads(run.stdout)
                    self.assertEqual(len(actual), len(expected))
                    for got, want in zip(actual, expected):
                        if isinstance(want, str):
                            self.assertEqual(got, want)
                        else:
                            self.assertAlmostEqual(got, want, delta=1e-6)

    def test_per_head_weighted_gated_normalization(self):
        import math
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / 'layers.sb3'
            build = subprocess.run([sys.executable, str(ROOT / 'tools/generate_project.py'), str(project)],
                                   capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stderr)
            payload = [2, 2, 3, 4, 0, 0, 1, -1, 0, 0, 2, 3]
            scale = math.sqrt(12.5 + 1e-6)
            expected = [3 / scale * 2 / (1 + math.exp(-1)),
                        4 / scale * 3 * (-1 / (1 + math.exp(1))), 0, 0]
            for values, want in ((payload, expected), (payload[:-1], ['invalid dimensions'])):
                with self.subTest(size=len(values)):
                    run = subprocess.run(['node', str(ROOT / 'tests/scratch/run-vm.js'), str(project),
                                          json.dumps(values), '34'], capture_output=True, text=True, timeout=60)
                    self.assertEqual(run.returncode, 0, run.stderr)
                    actual = json.loads(run.stdout)
                    self.assertEqual(len(actual), len(want))
                    for got, expected_value in zip(actual, want):
                        if isinstance(expected_value, str):
                            self.assertEqual(got, expected_value)
                        else:
                            self.assertAlmostEqual(got, expected_value, delta=1e-6)

    def test_per_head_weighted_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / 'layers.sb3'
            build = subprocess.run([sys.executable, str(ROOT / 'tools/generate_project.py'), str(project)],
                                   capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stderr)
            run = subprocess.run(['node', str(ROOT / 'tests/scratch/run-vm.js'), str(project),
                                  json.dumps([2, 2, 3, 4, 0, 0, 2, 3]), '35'],
                                 capture_output=True, text=True, timeout=60)
            self.assertEqual(run.returncode, 0, run.stderr)
            actual = json.loads(run.stdout)
            self.assertEqual(len(actual), 4)
            for got, want in zip(actual, [1.6970562, 3.3941124, 0, 0]):
                self.assertAlmostEqual(got, want, delta=1e-6)

    @unittest.skipUnless(os.environ.get('BONSAI_MODEL_PATH'), 'set BONSAI_MODEL_PATH to the locked model')
    def test_per_head_norms_match_native_layer_fixtures(self):
        tensors = {t['name']: t for t in json.loads((ROOT / 'fixtures/tensor-manifest.json').read_text())['tensors']}
        def read_f32(path):
            data = path.read_bytes()
            return list(struct.unpack(f'<{len(data) // 4}f', data))
        def weight(name):
            tensor = tensors[name]
            with open(os.environ['BONSAI_MODEL_PATH'], 'rb') as source:
                source.seek(tensor['offset'])
                return list(struct.unpack(f'<{tensor["size_bytes"] // 4}f', source.read(tensor['size_bytes'])))
        fixture = ROOT / 'fixtures/native/m5'
        query = read_f32(fixture / 'attention/Qcur_full-3-pos2.f32')
        query = [value for head in range(8) for value in query[head * 512:head * 512 + 256]]
        cases = [
            (35, [256, 8, *query, *weight('blk.3.attn_q_norm.weight')],
             read_f32(fixture / 'attention/Qcur_normed-3-pos2.f32')),
            (34, [128, 16, *read_f32(fixture / 'deltanet/attn_output-0-pos2.f32'),
                  *read_f32(fixture / 'deltanet/z-0-pos2.f32'), *weight('blk.0.ssm_norm.weight')],
             read_f32(fixture / 'deltanet/final_output-0-pos2.f32')),
        ]
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / 'layers.sb3'
            build = subprocess.run([sys.executable, str(ROOT / 'tools/generate_project.py'), str(project)],
                                   capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stderr)
            for mode, payload, expected in cases:
                with self.subTest(mode=mode):
                    run = subprocess.run(['node', str(ROOT / 'tests/scratch/run-vm.js'), str(project),
                                          json.dumps(payload), str(mode)], capture_output=True, text=True, timeout=60)
                    self.assertEqual(run.returncode, 0, run.stderr)
                    actual = json.loads(run.stdout)
                    self.assertEqual(len(actual), len(expected))
                    self.assertLessEqual(max(abs(a - b) for a, b in zip(actual, expected)), 1e-6)

    def test_vector_primitives_match_native_layer_fixtures(self):
        def read_f32(path):
            data = path.read_bytes()
            return list(struct.unpack(f'<{len(data) // 4}f', data))
        fixture = ROOT / 'fixtures/native/m5'
        read = lambda family, name: read_f32(fixture / family / f'{name}-pos2.f32')
        gate = read('deltanet', 'ffn_gate-0')
        up = read('deltanet', 'ffn_up-0')
        residual = read('deltanet', 'input')
        projected = read('deltanet', 'linear_attn_out-0')
        scores = read('attention', 'kq-3')
        cases = [
            (32, [len(gate), *gate, *up], read('deltanet', 'ffn_swiglu-0')),
            (33, [len(residual), *residual, *projected], read('deltanet', 'attn_residual-0')),
            (31, [2, scores[0] / 16, scores[1] / 16], read('attention', 'kq_soft_max-3')[:2]),
        ]
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / 'layers.sb3'
            build = subprocess.run([sys.executable, str(ROOT / 'tools/generate_project.py'), str(project)],
                                   capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stderr)
            for mode, payload, expected in cases:
                with self.subTest(mode=mode):
                    inputs = Path(directory) / 'input.f32'
                    inputs.write_bytes(struct.pack(f'<{len(payload)}f', *payload))
                    env = dict(os.environ, BONSAI_ACTIVATION_F32=str(inputs))
                    run = subprocess.run(['node', str(ROOT / 'tests/scratch/run-vm.js'), str(project),
                                          '[]', str(mode)], capture_output=True, text=True, timeout=60, env=env)
                    self.assertEqual(run.returncode, 0, run.stderr)
                    actual = json.loads(run.stdout)
                    self.assertEqual(len(actual), len(expected))
                    self.assertLessEqual(max(abs(a - b) for a, b in zip(actual, expected)), 1e-6)


if __name__ == '__main__':
    unittest.main()
