import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from test_transport_integration import add_loader_block

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(os.environ.get('BONSAI_TURBOWARP_VM') and os.environ.get('BONSAI_MODEL_PACKAGE'),
                     'set BONSAI_TURBOWARP_VM and BONSAI_MODEL_PACKAGE')
class LutKernelTests(unittest.TestCase):
    def test_lut_matches_scalar_with_real_resident_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            for row in range(5):
                with self.subTest(row=row):
                    project = Path(temp) / f'lut-{row}.sb3'
                    result = subprocess.run([sys.executable, str(ROOT / 'tools/generate_project.py'), str(project)], capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    add_loader_block(project, 16, 'l_scales', 16 * row)
                    outputs = []
                    cases = ((2, 1), (9, 1), (9, 3)) if row == 0 else ((2, 1), (9, 1))
                    for mode, iterations in cases:
                        result = subprocess.run(['node', str(ROOT / 'tests/scratch/run-vm.js'), str(project), '--dot', str(mode), str(iterations)],
                                                capture_output=True, text=True, timeout=60)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        outputs.append(json.loads(result.stdout))
                    self.assertEqual(outputs, [outputs[0]] * len(outputs))


if __name__ == '__main__':
    unittest.main()
