import json
import os
from pathlib import Path
import hashlib
import subprocess
import struct
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

    def test_bslut5_kernel_matches_with_real_resident_rows(self):
        package5 = os.environ.get('BONSAI_MODEL_PACKAGE5')
        if not package5:
            self.skipTest('set BONSAI_MODEL_PACKAGE5 to a bslut5 package')
        env = dict(os.environ, BONSAI_MODEL_PACKAGE=package5)
        with tempfile.TemporaryDirectory() as temp:
            for row in range(3):
                with self.subTest(row=row):
                    project = Path(temp) / f'lut5-{row}.sb3'
                    result = subprocess.run([sys.executable, str(ROOT / 'tools/generate_project.py'), str(project)], capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    add_loader_block(project, 16, 'l_scales', 16 * row)
                    result = subprocess.run(['node', str(ROOT / 'tests/scratch/run-vm.js'), str(project), '--dot', '9', '1'],
                                            env=env, capture_output=True, text=True, timeout=60)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    outputs5 = json.loads(result.stdout)
                    result = subprocess.run(['node', str(ROOT / 'tests/scratch/run-vm.js'), str(project), '--dot', '9', '1'],
                                            capture_output=True, text=True, timeout=60)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(outputs5, json.loads(result.stdout))

    def test_bslut5_tail_only_record_matches_raw(self):
        import struct as st
        with tempfile.TemporaryDirectory() as temp:
            tmp = Path(temp)
            name = b'blk.3.attn_q.weight'
            tensor = st.pack('<Q', len(name)) + name + st.pack('<IQQIQ', 2, 256, 1, 142, 0)
            header = b'GGUF' + st.pack('<IQQ', 3, 1, 0) + tensor
            header += bytes((-len(header)) % 32)
            # Group 0: all trits 1 (zero weight) except positions 125,126,127 = (0, 2, 0) -> (-1, +1, -1).
            g0_trits = [1] * 125 + [0, 2, 0]
            g0_bytes = bytes(g0_trits[4 * j] | (g0_trits[4 * j + 1] << 2) | (g0_trits[4 * j + 2] << 4) | (g0_trits[4 * j + 3] << 6)
                             for j in range(32))
            g0 = b'\x00\x3c' + g0_bytes
            g1 = b'\x00\x3c' + bytes([0x55] * 32)
            model = tmp / 'tail.gguf'
            model.write_bytes(header + g0 + g1)
            lock = tmp / 'tail.lock.json'
            lock.write_text(json.dumps({'size_bytes': model.stat().st_size,
                                        'sha256': hashlib.sha256(model.read_bytes()).hexdigest()}))
            raw_pkg, b5_pkg = tmp / 'raw.bsp', tmp / 'b5.bsp'
            for codec, pkg in (('raw', raw_pkg), ('bslut5', b5_pkg)):
                res = subprocess.run([sys.executable, str(ROOT / 'tools/package_model.py'),
                                      str(model), str(pkg), '--codec', codec, '--lock', str(lock)],
                                     capture_output=True, text=True)
                self.assertEqual(res.returncode, 0, res.stderr)
            act = tmp / 'act.f32'
            act.write_bytes(st.pack('<256f', *[0.01 * (i + 1) for i in range(256)]))
            project = tmp / 'tail.sb3'
            res = subprocess.run([sys.executable, str(ROOT / 'tools/generate_project.py'), str(project)],
                                 capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, res.stderr)
            add_loader_block(project, 2, 'l_scales', 0)
            outs = []
            for mode, pkg in (('2', raw_pkg), ('9', b5_pkg)):
                env = dict(os.environ, BONSAI_MODEL_PACKAGE=str(pkg), BONSAI_ACTIVATION_F32=str(act))
                run = subprocess.run(['node', str(ROOT / 'tests/scratch/run-vm.js'), str(project), '--dot', mode, '1'],
                                     env=env, capture_output=True, text=True, timeout=60)
                self.assertEqual(run.returncode, 0, run.stderr)
                outs.append(json.loads(run.stdout))
            self.assertEqual(outs[0], outs[1])
            self.assertNotEqual(outs[0][0], 0.0)


if __name__ == '__main__':
    unittest.main()
