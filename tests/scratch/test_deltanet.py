import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / 'fixtures/native/m5/deltanet'


def read_f32(path):
    data = path.read_bytes()
    return list(struct.unpack(f'<{len(data) // 4}f', data))


@unittest.skipUnless(os.environ.get('BONSAI_TURBOWARP_VM') and os.environ.get('BONSAI_MODEL_PACKAGE'),
                     'set BONSAI_TURBOWARP_VM and BONSAI_MODEL_PACKAGE')
class DeltaNetStageTests(unittest.TestCase):
    """Stage-by-stage checks of the Gated DeltaNet layer against captured boundaries."""

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.project = Path(cls.temp.name) / 'deltanet.sb3'
        built = subprocess.run([sys.executable, str(ROOT / 'tools/generate_project.py'), str(cls.project)],
                               capture_output=True, text=True)
        if built.returncode:
            raise RuntimeError(built.stderr)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def run_stage(self, position, activation, dump):
        output = Path(self.temp.name) / f'{dump.replace(" ", "-")}-{position}.json'
        env = dict(os.environ, BONSAI_ACTIVATION_F32=str(activation), BONSAI_LAYER_POSITION=str(position - 1),
                   BONSAI_DUMP=f'{dump}:{output}')
        run = subprocess.run(['node', str(ROOT / 'tests/scratch/run-vm.js'), str(self.project), '[]', '41'],
                             capture_output=True, text=True, timeout=300, env=env)
        self.assertEqual(run.returncode, 0, run.stderr)
        return [float(v) for v in json.loads(output.read_text())]

    def test_convolution_and_silu_match_native(self):
        actual = self.run_stage(1, FIXTURE / 'input-pos1.f32', 'BS conv silu')
        expected = read_f32(FIXTURE / 'conv_output_silu-0-pos1.f32')
        self.assertEqual(len(actual), len(expected))
        errors = [abs(a - b) for a, b in zip(actual, expected)]
        self.assertLessEqual(max(errors), 1e-6)
        self.assertLessEqual((sum(e * e for e in errors) / len(errors)) ** 0.5, 1e-7)


if __name__ == '__main__':
    unittest.main()
