"""Two activations must project through one weight pass with independent results."""
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / 'fixtures/native'


@unittest.skipUnless(os.environ.get('BONSAI_TURBOWARP_VM') and os.environ.get('BONSAI_MODEL_PACKAGE'),
                     'set BONSAI_TURBOWARP_VM and BONSAI_MODEL_PACKAGE')
class PairProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.project = Path(cls.temp.name) / 'pair.sb3'
        built = subprocess.run([sys.executable, str(ROOT / 'tools/generate_project.py'), str(cls.project),
                                '--benchmark'], capture_output=True, text=True)
        assert built.returncode == 0, built.stderr
        cls.base = {key: value for key, value in os.environ.items()
                    if key in ('BONSAI_TURBOWARP_VM', 'BONSAI_MODEL_PACKAGE', 'NODE_PATH')}
        cls.base.update(BONSAI_TIMEOUT_MS='600000', BONSAI_DEBUG_FULL='1',
                        BONSAI_DEBUG_VARS='BS pair a,BS pair b')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def run_pair(self, input_a, input_b):
        env = dict(self.base, BONSAI_ACTIVATION_F32=str(input_a), BONSAI_ACTIVATION_B_F32=str(input_b))
        run = subprocess.run(['node', str(ROOT / 'tests/scratch/run-vm.js'), str(self.project), '[]', '43'],
                             env=env, capture_output=True, text=True, timeout=600)
        self.assertEqual(run.returncode, 0, run.stderr[-2000:])
        state = json.loads(run.stdout.strip().splitlines()[-1])
        return state['BS pair a'], state['BS pair b']

    def test_equal_inputs_give_equal_projections(self):
        same = FIXTURES / 'layer3-q-input.f32'
        a, b = self.run_pair(same, same)
        self.assertEqual(len(a), 4096)
        self.assertEqual(len(b), 4096)
        self.assertEqual(a, b, 'both accumulators see the same weights and bytes')

    def test_the_second_input_is_independent(self):
        first = FIXTURES / 'layer3-q-input.f32'
        second = FIXTURES / 'layer0-ffn-down-input.f32'
        a_same, b_same = self.run_pair(first, first)
        a_mixed, b_mixed = self.run_pair(first, second)
        self.assertEqual(a_same, a_mixed, 'the first projection does not depend on the second input')
        self.assertNotEqual(b_same, b_mixed, 'the second projection follows its own input')
        self.assertLessEqual(max(abs(x - y) for x, y in zip(b_mixed, b_same)), 1e6)


if __name__ == '__main__':
    unittest.main()
