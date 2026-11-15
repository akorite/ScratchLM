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
GENERATOR = ROOT / 'tools' / 'generate_project.py'
RUNNER = ROOT / 'tests' / 'scratch' / 'run-vm.js'
M4_TRANS = ROOT / 'fixtures' / 'native' / 'm4' / 'transforms'


def read_f32(path):
    data = path.read_bytes()
    return list(struct.unpack(f'<{len(data) // 4}f', data))


@unittest.skipUnless(os.environ.get('BONSAI_TURBOWARP_VM'), 'set BONSAI_TURBOWARP_VM to the pinned VM checkout')
class TransformTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.project = Path(self.temp.name) / 'transform.sb3'
        generated = subprocess.run([sys.executable, str(GENERATOR), str(self.project)], capture_output=True, text=True)
        self.assertEqual(generated.returncode, 0, generated.stderr)

    def tearDown(self):
        self.temp.cleanup()

    def run_transform(self, values, mode=12, signs_path=None):
        f32_path = Path(self.temp.name) / 'in.f32'
        f32_path.write_bytes(struct.pack(f'<{len(values)}f', *values))
        env = dict(os.environ, BONSAI_ACTIVATION_F32=str(f32_path))
        if signs_path is not None:
            env['BONSAI_SIGNS_F32'] = str(signs_path)
        result = subprocess.run(['node', str(RUNNER), str(self.project), '[]', str(mode)],
                                env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_width_2048_and_6144_match_native_goldens(self):
        manifest = json.loads((M4_TRANS / 'manifest.json').read_text())
        pair_2048 = next(p for p in manifest['pairs']
                         if p['activation_width'] == 2048 and p['pre_sign']
                         and p['pre_sign']['sha256'] != p['input']['sha256'])
        pair_6144 = next(p for p in manifest['pairs'] if p['activation_width'] == 6144 and p['signs'])
        # Note: 6144-wide ffn_swiglu sign-mul runs in-place in ggml, so undo ±1 signs once to test signed transform path.
        signs_6144 = M4_TRANS / 'prism.hadamard.signs.6144.f32'
        s6 = read_f32(signs_6144)
        pre_6144 = [x * s for x, s in zip(read_f32(M4_TRANS / pair_6144['input']['file']), s6)]
        cases = [
            (2048, pair_2048['node'], read_f32(M4_TRANS / pair_2048['pre_sign']['file']),
             M4_TRANS / 'prism.hadamard.signs.2048.f32', read_f32(M4_TRANS / pair_2048['rotation_out']['file'])),
            (6144, pair_6144['node'], pre_6144, signs_6144, read_f32(M4_TRANS / pair_6144['rotation_out']['file'])),
        ]
        for width, node, pre, signs, expected in cases:
            with self.subTest(width=width, node=node):
                actual = self.run_transform(pre, mode=12, signs_path=signs)
                self.assertEqual(len(actual), width)
                errors = [a - b for a, b in zip(actual, expected)]
                self.assertLessEqual(max(abs(e) for e in errors), 1e-6)
                self.assertLessEqual(math.sqrt(math.fsum(e * e for e in errors) / width), 3e-7)

    def test_inplace_6144_pair_matches_without_double_signing(self):
        manifest = json.loads((M4_TRANS / 'manifest.json').read_text())
        pair = next(p for p in manifest['pairs'] if p['activation_width'] == 6144)
        signed_in = read_f32(M4_TRANS / pair['input']['file'])
        expected = read_f32(M4_TRANS / pair['rotation_out']['file'])
        actual = self.run_transform(signed_in, mode=12)
        errors = [a - b for a, b in zip(actual, expected)]
        self.assertLessEqual(max(abs(e) for e in errors), 1e-6)
        self.assertLessEqual(math.sqrt(math.fsum(e * e for e in errors) / 6144), 3e-7)

    def test_disabling_or_flipping_signs_changes_output(self):
        manifest = json.loads((M4_TRANS / 'manifest.json').read_text())
        pair = next(p for p in manifest['pairs']
                    if p['activation_width'] == 2048 and p['pre_sign']
                    and p['pre_sign']['sha256'] != p['input']['sha256'])
        pre = read_f32(M4_TRANS / pair['pre_sign']['file'])
        signs = M4_TRANS / 'prism.hadamard.signs.2048.f32'
        with_signs = self.run_transform(pre, mode=12, signs_path=signs)
        without_signs = self.run_transform(pre, mode=12)
        self.assertGreater(max(abs(a - b) for a, b in zip(with_signs, without_signs)), 0.1)
        flipped_path = Path(self.temp.name) / 'flipped.f32'
        flipped = [-v for v in read_f32(signs)]
        flipped_path.write_bytes(struct.pack('<2048f', *flipped))
        wrong_signs = self.run_transform(pre, mode=12, signs_path=flipped_path)
        self.assertGreater(max(abs(a - b) for a, b in zip(with_signs, wrong_signs)), 0.1)

    def test_normalized_hadamard_is_self_inverse(self):
        vec = [math.sin(i * 0.13) for i in range(1024)]
        once = self.run_transform(vec, mode=12)
        twice = self.run_transform(once, mode=12)
        self.assertLessEqual(max(abs(a - b) for a, b in zip(twice, vec)), 2e-6)

    def test_partial_blocks_and_mismatched_signs_rejected(self):
        self.assertEqual(self.run_transform([1.0] * 500, mode=12), ['invalid dimensions'])
        self.assertEqual(self.run_transform([], mode=12), ['invalid dimensions'])
        bad_signs = Path(self.temp.name) / 'bad-signs.f32'
        bad_signs.write_bytes(struct.pack('<512f', *([1.0] * 512)))
        self.assertEqual(self.run_transform([1.0] * 1024, mode=12, signs_path=bad_signs), ['invalid dimensions'])


if __name__ == '__main__':
    unittest.main()
