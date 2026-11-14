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

    def compare(self, dump, fixture, tolerance):
        actual = self.run_stage(1, FIXTURE / 'input-pos1.f32', dump)
        expected = read_f32(FIXTURE / fixture)
        self.assertEqual(len(actual), len(expected), fixture)
        errors = [abs(a - b) for a, b in zip(actual, expected)]
        rms = (sum(e * e for e in errors) / len(errors)) ** 0.5
        self.assertLessEqual(max(errors), tolerance, f'{fixture} max abs')
        self.assertLessEqual(rms, tolerance / 3, f'{fixture} rms')

    def test_convolution_and_silu_match_native(self):
        self.compare('BS conv silu', 'conv_output_silu-0-pos1.f32', 1e-6)

    def test_joint_qk_l2_matches_native(self):
        self.compare('BS qk l2', 'qk_conv_l2-0-pos1.f32', 1e-6)

    def test_gate_projection_matches_native(self):
        self.compare('BS z', 'z-0-pos1.f32', 2e-6)

    def test_gate_scalars_match_native(self):
        self.compare('BS beta', 'beta-0-pos1.f32', 5e-6)
        self.compare('BS alpha', 'alpha-0-pos1.f32', 5e-6)


if __name__ == '__main__':
    unittest.main()
