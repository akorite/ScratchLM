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
STATE = os.environ.get('BONSAI_LAYER_STATE')


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
                   BONSAI_LAYER='0', BONSAI_DUMP=f'{dump}:{output}')
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

    def test_layers_dispatch_by_index_through_the_chain(self):
        cases = ((0, 'deltanet/input-pos1.f32', 'chain/l_out-0-pos1.f32', '41'),
                 (1, 'chain/l_out-0-pos1.f32', 'chain/l_out-1-pos1.f32', '41'),
                 (3, 'chain/l_out-2-pos1.f32', 'chain/l_out-3-pos1.f32', '40'))
        for layer, activation, expected, mode in cases:
            with self.subTest(layer=layer):
                output = Path(self.temp.name) / f'chain-{layer}.json'
                env = dict(os.environ, BONSAI_ACTIVATION_F32=str(ROOT / 'fixtures/native/m5' / activation),
                           BONSAI_LAYER_POSITION='0', BONSAI_LAYER=str(layer),
                           BONSAI_DUMP=f'BS output:{output}')
                run = subprocess.run(['node', str(ROOT / 'tests/scratch/run-vm.js'), str(self.project), '[]', mode],
                                     capture_output=True, text=True, timeout=300, env=env)
                self.assertEqual(run.returncode, 0, run.stderr)
                got = [float(v) for v in json.loads(output.read_text())]
                want = read_f32(ROOT / 'fixtures/native/m5' / expected)
                errors = [abs(a - b) for a, b in zip(got, want)]
                self.assertLessEqual(max(errors), 1e-6, expected)

    def test_gated_norm_and_output_projection_match_native(self):
        self.compare('BS final out', 'final_output-0-pos1.f32', 1e-6)
        self.compare('BS attn out', 'linear_attn_out-0-pos1.f32', 1e-6)

    def test_layer_output_matches_native_at_first_position(self):
        self.compare('BS output', 'l_out-0-pos1.f32', 1e-6)

    def seeded_run(self, position, seed_position):
        """Run one position with the previous token's recurrent state and convolution taps."""
        state = Path(self.temp.name) / f'conv-{position}.f32'
        native = read_f32(FIXTURE / f'conv_input-0-pos{position}.f32')
        taps = [0.0] * (3 * 6144)
        for channel in range(6144):
            for tap in range(3):
                taps[tap * 6144 + channel] = native[channel * 4 + tap]
        state.write_bytes(struct.pack(f'<{len(taps)}f', *taps))
        output = Path(self.temp.name) / f'seeded-{position}.json'
        env = dict(os.environ, BONSAI_ACTIVATION_F32=str(FIXTURE / f'input-pos{position}.f32'),
                   BONSAI_LAYER_POSITION=str(position - 1), BONSAI_LAYER='0',
                   BONSAI_DUMP=f'BS output:{output}',
                   BONSAI_DN_STATE=str(Path(STATE) / f'new_state-0-pos{seed_position}.f32'),
                   BONSAI_CONV_STATE=str(state))
        run = subprocess.run(['node', str(ROOT / 'tests/scratch/run-vm.js'), str(self.project), '[]', '41'],
                             capture_output=True, text=True, timeout=300, env=env)
        self.assertEqual(run.returncode, 0, run.stderr)
        return [float(v) for v in json.loads(output.read_text())]

    def check_output(self, actual, fixture):
        expected = read_f32(FIXTURE / fixture)
        self.assertEqual(len(actual), len(expected), fixture)
        errors = [abs(a - b) for a, b in zip(actual, expected)]
        self.assertLessEqual(max(errors), 1e-6, f'{fixture} max abs')
        self.assertLessEqual((sum(e * e for e in errors) / len(errors)) ** 0.5, 1e-7, f'{fixture} rms')

    @unittest.skipUnless(STATE, 'set BONSAI_LAYER_STATE to the captured state directory')
    def test_layer_output_matches_native_with_seeded_state(self):
        for position, seed in ((2, 1), (16, 15)):
            with self.subTest(position=position):
                self.check_output(self.seeded_run(position, seed), f'l_out-0-pos{position}.f32')


if __name__ == '__main__':
    unittest.main()
