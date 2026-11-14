import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

from test_transport_integration import add_loader_block

ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / 'tools' / 'generate_project.py'
RUNNER = ROOT / 'tests' / 'scratch' / 'run-vm.js'
M4 = ROOT / 'fixtures' / 'native' / 'm4'


def read_f32(path):
    data = path.read_bytes()
    return list(struct.unpack(f'<{len(data) // 4}f', data))


def extract_tensor_f32(package_path, tensor_name):
    with open(package_path, 'rb') as f:
        assert f.read(8) == b'BNSPKG1\0'
        mlen = struct.unpack('<Q', f.read(8))[0]
        manifest = json.loads(f.read(mlen))
        pstart = f.tell()
        t = next(item for item in manifest['tensors'] if item['name'] == tensor_name)
        f.seek(pstart + t['offset'])
        return list(struct.unpack(f'<{t["length"] // 4}f', f.read(t['length'])))


@unittest.skipUnless(os.environ.get('BONSAI_TURBOWARP_VM') and os.environ.get('BONSAI_MODEL_PACKAGE'),
                     'set BONSAI_TURBOWARP_VM and BONSAI_MODEL_PACKAGE')
class EmbeddingAndNormTests(unittest.TestCase):
    def test_loader_embedding_rows_match_native_and_share_tied_tensor(self):
        pkg = Path(os.environ['BONSAI_MODEL_PACKAGE'])
        with open(pkg, 'rb') as f:
            assert f.read(8) == b'BNSPKG1\0'
            mlen = struct.unpack('<Q', f.read(8))[0]
            manifest = json.loads(f.read(mlen))
        names = [t['name'] for t in manifest['tensors']]
        self.assertEqual(names.count('token_embd.weight'), 1)
        self.assertNotIn('output.weight', names)

        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp) / 'emb.sb3'
            res = subprocess.run([sys.executable, str(GENERATOR), str(project), '--benchmark'],
                                 capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, res.stderr)
            for tid in (1, 321, 248046):
                with self.subTest(token_id=tid):
                    env = dict(os.environ, BONSAI_TOKEN_ID=str(tid))
                    run = subprocess.run(['node', str(RUNNER), str(project), '--stream', '15'],
                                         env=env, capture_output=True, text=True, timeout=60)
                    self.assertEqual(run.returncode, 0, run.stderr)
                    actual = json.loads(run.stdout)
                    expected = read_f32(M4 / f'embedding/token-{tid}-row.f32')
                    self.assertEqual(len(actual), 2048)
                    errors = [a - b for a, b in zip(actual, expected)]
                    self.assertLessEqual(max(abs(e) for e in errors), 1e-6)
                    self.assertLessEqual(math.sqrt(math.fsum(e * e for e in errors) / 2048), 3e-7)

    def test_resident_embedding_runs_with_loader_disabled(self):
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp) / 'res-emb.sb3'
            res = subprocess.run([sys.executable, str(GENERATOR), str(project)], capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, res.stderr)
            add_loader_block(project, 16, 'l_scales', 16, tensor='token_embd.weight')
            env = dict(os.environ, BONSAI_SIGNS_F32=str(M4 / 'transforms/prism.hadamard.signs.2048.f32'))
            run = subprocess.run(['node', str(RUNNER), str(project), '--dot', '14'],
                                 env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(run.returncode, 0, run.stderr)
            actual = json.loads(run.stdout)
            expected = read_f32(M4 / 'embedding/token-1-row.f32')
            errors = [a - b for a, b in zip(actual, expected)]
            self.assertLessEqual(max(abs(e) for e in errors), 1e-6)

    def test_output_norm_matches_native_and_handles_near_zero_edges(self):
        pkg = Path(os.environ['BONSAI_MODEL_PACKAGE'])
        weights = extract_tensor_f32(pkg, 'output_norm.weight')
        eps = 9.999999974752427e-07
        with tempfile.TemporaryDirectory() as temp:
            tmp = Path(temp)
            w_file = tmp / 'norm-w.f32'
            w_file.write_bytes(struct.pack('<2048f', *weights))
            # Resident mode 16 against native l_out-23 -> result_norm golden
            project = tmp / 'norm.sb3'
            res = subprocess.run([sys.executable, str(GENERATOR), str(project)], capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, res.stderr)
            env = dict(os.environ,
                       BONSAI_ACTIVATION_F32=str(M4 / 'head/output-norm-input.f32'),
                       BONSAI_NORM_WEIGHTS_F32=str(w_file))
            run = subprocess.run(['node', str(RUNNER), str(project), '[]', '16'],
                                 env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(run.returncode, 0, run.stderr)
            actual = json.loads(run.stdout)
            expected = read_f32(M4 / 'head/output-norm-output.f32')
            errors = [a - b for a, b in zip(actual, expected)]
            self.assertLessEqual(max(abs(e) for e in errors), 1e-6)
            self.assertLessEqual(math.sqrt(math.fsum(e * e for e in errors) / 2048), 2e-7)

            # Loader-backed mode 16 (--benchmark) loads output_norm.weight directly from package
            bproj = tmp / 'norm-bench.sb3'
            res = subprocess.run([sys.executable, str(GENERATOR), str(bproj), '--benchmark'],
                                 capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, res.stderr)
            env_b = dict(os.environ, BONSAI_ACTIVATION_F32=str(M4 / 'head/output-norm-input.f32'))
            run_b = subprocess.run(['node', str(RUNNER), str(bproj), '--stream', '16'],
                                   env=env_b, capture_output=True, text=True, timeout=60)
            self.assertEqual(run_b.returncode, 0, run_b.stderr)
            self.assertEqual(json.loads(run_b.stdout), actual)

            # Near-zero mean-square and exact-zero edge cases
            for vec in ([0.0] * 2048, [1e-6] * 2048):
                vf = tmp / 'edge.f32'
                vf.write_bytes(struct.pack('<2048f', *vec))
                env_e = dict(os.environ, BONSAI_ACTIVATION_F32=str(vf), BONSAI_NORM_WEIGHTS_F32=str(w_file))
                run_e = subprocess.run(['node', str(RUNNER), str(project), '[]', '16'],
                                       env=env_e, capture_output=True, text=True, timeout=60)
                self.assertEqual(run_e.returncode, 0, run_e.stderr)
                got_e = json.loads(run_e.stdout)
                ms = sum(x * x for x in vec) / 2048
                scale = 1.0 / math.sqrt(ms + eps) if any(vec) else 0.0
                want_e = [struct.unpack('<f', struct.pack('<f', x * scale * w))[0] for x, w in zip(vec, weights)]
                for a, b in zip(got_e, want_e):
                    self.assertLessEqual(abs(a - b), 1e-9)


if __name__ == '__main__':
    unittest.main()
