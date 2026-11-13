import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(os.environ.get('BONSAI_TURBOWARP_VM'), 'set BONSAI_TURBOWARP_VM to the pinned VM checkout')
class NumericExtraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.project = Path(cls.temp.name) / 'numeric.sb3'
        result = subprocess.run([sys.executable, str(ROOT / 'tools/generate_project.py'), str(cls.project)], capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stderr)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def check_values(self, mode, values, expected):
        result = subprocess.run(['node', str(ROOT / 'tests/scratch/run-vm.js'), str(self.project), json.dumps(values), str(mode)], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        actual = json.loads(result.stdout)
        self.assertEqual(len(actual), len(expected))
        for got, want in zip(actual, expected):
            if isinstance(want, str):
                self.assertEqual(got, want)
            else:
                self.assertAlmostEqual(got, want, delta=1e-9 * max(1, abs(want)))

    def test_dense_dot_and_invalid_dimensions(self):
        for values, expected in [([3, 1, -2, 3, 4, 5, -6], [-24]), ([0], [0]),
                                 ([], ['invalid dimensions']), ([2, 1, 2, 3], ['invalid dimensions']),
                                 ([-1], ['invalid dimensions']), ([0.5, 1], ['invalid dimensions'])]:
            with self.subTest(values=values):
                self.check_values(4, values, expected)

    def test_rms_normalization(self):
        self.check_values(5, [3, 4], [0.8485281034827337, 1.1313708046436448])
        self.check_values(5, [0, 0], [0, 0])
        self.check_values(5, [], [])
        self.check_values(5, [1e200, -1e200], [1, -1])

    def test_l2_normalization(self):
        self.check_values(6, [3, 4], [0.6, 0.8])
        self.check_values(6, [0, 0], [0, 0])
        self.check_values(6, [1e-8, 0], [0.010000000025247573, 0])
        self.check_values(6, [], [])
        self.check_values(6, [1e200, -1e200], [0.7071067811865475, -0.7071067811865475])

    def test_silu_handles_both_extremes(self):
        self.check_values(7, [-1000, -1, 0, 1, 1000], [0, -0.2689414213699951, 0, 0.7310585786300049, 1000])
        self.check_values(7, [], [])

    def test_softmax_is_shift_invariant(self):
        self.check_values(8, [1000, 1001, 1002], [0.09003057317038046, 0.24472847105479764, 0.6652409557748218])
        self.check_values(8, [-1000, -1000], [0.5, 0.5])
        self.check_values(8, [], [])

    def test_radian_trig(self):
        self.check_values(10, [0, 1.5707963267948966, -3.141592653589793, 0.5],
                          [0, 1, 1, 0, 0, -1, 0.4794255386, 0.8775825619])
