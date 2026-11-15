"""The paired DeltaNet stage must reproduce the single-token stage exactly."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / 'fixtures/native'


@unittest.skipUnless(os.environ.get('BONSAI_TURBOWARP_VM') and os.environ.get('BONSAI_MODEL_PACKAGE'),
                     'set BONSAI_TURBOWARP_VM and BONSAI_MODEL_PACKAGE')
class StagePairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.project = Path(cls.temp.name) / 'stage-pair.sb3'
        built = subprocess.run([sys.executable, str(ROOT / 'tools/generate_project.py'), str(cls.project)],
                               capture_output=True, text=True)
        assert built.returncode == 0, built.stderr
        cls.env = {key: value for key, value in os.environ.items()
                   if key in ('BONSAI_TURBOWARP_VM', 'BONSAI_MODEL_PACKAGE', 'NODE_PATH')}
        cls.env.update(BONSAI_LAYER='0', BONSAI_LAYER_POSITION='0', BONSAI_DEBUG_FULL='1',
                       BONSAI_TIMEOUT_MS='900000')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def invoke(self, mode, **env):
        result = subprocess.run(['node', str(ROOT / 'tests/scratch/run-vm.js'), str(self.project), '[]', mode],
                                env=dict(self.env, **env), capture_output=True, text=True, timeout=1800)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        return json.loads([line for line in result.stdout.strip().splitlines() if line.startswith('{')][-1])

    def single(self, path):
        return self.invoke('41', BONSAI_ACTIVATION_F32=str(path), BONSAI_DEBUG_VARS='BS dn layer out')['BS dn layer out']

    def pair(self, first, second):
        state = self.invoke('46', BONSAI_ACTIVATION_F32=str(first), BONSAI_ACTIVATION_B_F32=str(second),
                         BONSAI_DEBUG_VARS='BS pair out a,BS pair out b')
        return state['BS pair out a'], state['BS pair out b']

    def test_equal_inputs_match_the_single_stage(self):
        first = FIXTURES / 'layer3-q-input.f32'
        a, b = self.pair(first, first)
        want = self.single(first)
        self.assertEqual(len(a), len(want))
        self.assertEqual(a, want, 'one weight pass must reproduce the single-token stage exactly')
        # the second token runs after the first has advanced the recurrent state,
        # so it differs from a fresh single run while staying a full-width vector
        self.assertEqual(len(b), len(want))
        self.assertNotEqual(b, want, 'the second token continues the recurrence')

    def control(self, first, second):
        state = self.invoke('47', BONSAI_ACTIVATION_F32=str(first), BONSAI_ACTIVATION_B_F32=str(second),
                            BONSAI_DEBUG_VARS='BS ctl out a,BS ctl out b,BS state')
        return state['BS ctl out a'], state['BS ctl out b'], state['BS state']

    def test_both_tokens_match_a_sequential_run_and_leave_the_same_state(self):
        first = FIXTURES / 'layer3-q-input.f32'
        second = FIXTURES / 'layer0-attn-norm.f32'
        a, b = self.pair(first, second)
        ctl_a, ctl_b, ctl_state = self.control(first, second)
        self.assertEqual(a, ctl_a, 'the first token must match a sequential run')
        self.assertEqual(b, ctl_b, 'the second token must match a sequential run after the first')
        final = self.invoke('46', BONSAI_ACTIVATION_F32=str(first), BONSAI_ACTIVATION_B_F32=str(second),
                            BONSAI_DEBUG_VARS='BS state')['BS state']
        self.assertEqual(final, ctl_state, 'the recurrent state must advance identically')

    def test_the_second_token_does_not_change_the_first(self):
        first = FIXTURES / 'layer3-q-input.f32'
        second = FIXTURES / 'layer0-attn-norm.f32'
        a_same, b_same = self.pair(first, first)
        a_mixed, b_mixed = self.pair(first, second)
        self.assertEqual(a_same, a_mixed, 'the first token must not depend on the second')
        self.assertNotEqual(b_same, b_mixed, 'the second token must follow its own input')


if __name__ == '__main__':
    unittest.main()
