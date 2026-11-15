import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
STATE = os.environ.get('BONSAI_LAYER_STATE')
TOLERANCE = {'max_abs': 1e-4, 'rms': 2e-5}


def read_f32(path):
    data = path.read_bytes()
    return list(struct.unpack(f'<{len(data) // 4}f', data))


@unittest.skipUnless(os.environ.get('BONSAI_TURBOWARP_VM') and os.environ.get('BONSAI_MODEL_PACKAGE') and STATE,
                     'set BONSAI_TURBOWARP_VM, BONSAI_MODEL_PACKAGE and BONSAI_LAYER_STATE')
class AttentionLayerTests(unittest.TestCase):
    """The generated full-attention layer against the captured native boundaries."""

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.project = Path(cls.temp.name) / 'layer.sb3'
        built = subprocess.run([sys.executable, str(ROOT / 'tools/generate_project.py'), str(cls.project)],
                               capture_output=True, text=True)
        if built.returncode:
            raise RuntimeError(built.stderr)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def run_layer(self, position, activation, cache=None, tokens=0, dump='BS output'):
        output = Path(self.temp.name) / f'{dump.replace(" ", "-")}-{position}.json'
        env = dict(os.environ, BONSAI_ACTIVATION_F32=str(activation), BONSAI_LAYER_POSITION=str(position - 1),
                   BONSAI_DUMP=f'{dump}:{output}')
        if cache:
            env.update(BONSAI_KV_TOKENS=str(tokens), BONSAI_KV_KEY=str(cache[0]), BONSAI_KV_VALUE=str(cache[1]))
        run = subprocess.run(['node', str(ROOT / 'tests/scratch/run-vm.js'), str(self.project), '[]', '40'],
                             capture_output=True, text=True, timeout=300, env=env)
        self.assertEqual(run.returncode, 0, run.stderr)
        return [float(v) for v in json.loads(output.read_text())]

    def compare(self, actual, expected, label):
        self.assertEqual(len(actual), len(expected), label)
        errors = [abs(a - b) for a, b in zip(actual, expected)]
        rms = (sum(e * e for e in errors) / len(errors)) ** 0.5
        self.assertLessEqual(max(errors), TOLERANCE['max_abs'], label)
        self.assertLessEqual(rms, TOLERANCE['rms'], label)

    def test_layer_matches_native_layer_boundaries(self):
        fixture = ROOT / 'fixtures/native/m5/attention'
        state = Path(STATE)
        cases = [
            (1, None, 0),
            (2, (state / 'cache-key-3-pos1.f32', state / 'cache-value-3-pos1.f32'), 1),
            (64, (state / 'cache-key-3-pos64.f32', state / 'cache-value-3-pos64.f32'), 63),
        ]
        for position, cache, tokens in cases:
            with self.subTest(position=position):
                actual = self.run_layer(position, fixture / f'input-pos{position}.f32', cache, tokens)
                self.compare(actual, read_f32(fixture / f'l_out-3-pos{position}.f32'), f'layer output at {position}')

    def test_attention_stage_matches_single_position_fixture(self):
        fixture = ROOT / 'fixtures/native/m5/attention'
        actual = self.run_layer(1, fixture / 'input-pos1.f32', dump='BS attn')
        expected = read_f32(fixture / 'attn_pregate-3-pos1.f32')
        errors = [abs(a - b) for a, b in zip(actual, expected)]
        self.assertEqual(len(actual), len(expected))
        self.assertLessEqual(max(errors), 1e-6)


if __name__ == '__main__':
    unittest.main()
