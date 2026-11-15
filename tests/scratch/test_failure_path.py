"""A rejected weight tile must stop the step instead of decoding stale state."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(os.environ.get('BONSAI_TURBOWARP_VM') and os.environ.get('BONSAI_MODEL_PACKAGE'),
                     'set BONSAI_TURBOWARP_VM and BONSAI_MODEL_PACKAGE')
class FailurePathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.project = Path(cls.temp.name) / 'failure.sb3'
        built = subprocess.run([sys.executable, str(ROOT / 'tools/generate_project.py'), str(cls.project),
                                '--text', '--benchmark'], capture_output=True, text=True)
        assert built.returncode == 0, built.stderr
        cls.env = {key: value for key, value in os.environ.items()
                   if key in ('BONSAI_TURBOWARP_VM', 'BONSAI_MODEL_PACKAGE', 'NODE_PATH')}
        cls.env.update(BONSAI_TOKEN_ID='1237', BONSAI_LAYER_POSITION='0', BONSAI_TIMEOUT_MS='600000',
                       BONSAI_DEBUG_VARS='BS step valid,BS proj valid,BS greedy token,BS scanned,BS stop reason,BS text',
                       BONSAI_DEBUG_FULL='1')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def run_mode(self, args, extra):
        env = dict(self.env, **extra)
        run = subprocess.run(['node', str(ROOT / 'tests/scratch/run-vm.js'), str(self.project), *args],
                             env=env, capture_output=True, text=True, timeout=900)
        self.assertEqual(run.returncode, 0, run.stderr[-3000:])
        return json.loads(run.stdout.strip().splitlines()[-1])

    def test_rejected_tile_invalidates_the_step_and_produces_no_token(self):
        failed = self.run_mode(['[]', '42'], {'BONSAI_FAIL_TILE': '1'})
        self.assertEqual(failed['BS step valid'], 0, 'the step reports failure')
        self.assertEqual(failed['BS proj valid'], 0)
        self.assertEqual(failed['BS scanned'], 0, 'the head scan never runs on a failed step')
        self.assertEqual(failed['BS greedy token'], 0, 'no token is produced')

    def test_generation_loop_stops_on_a_failed_step(self):
        state = self.run_mode(['--generate', '24'], {'BONSAI_FAIL_TILE': '1', 'BONSAI_PROMPT': 'hi',
                                                     'BONSAI_MAX_TOKENS': '1', 'BONSAI_STEPS': '3'})
        self.assertEqual(state['BS step valid'], 0)
        self.assertEqual(state['BS stop reason'], 'failure')
        self.assertEqual(state['BS greedy token'], 0)
        self.assertFalse(state['BS text'], 'a failed step renders no text')

    def test_a_good_step_still_produces_the_native_token(self):
        good = self.run_mode(['[]', '42'], {})
        self.assertEqual(good['BS step valid'], 1)
        self.assertEqual(good['BS greedy token'], 11)
        self.assertEqual(good['BS scanned'], 248320)


if __name__ == '__main__':
    unittest.main()
