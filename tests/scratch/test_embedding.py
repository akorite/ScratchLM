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

    def test_loader_modes_replace_stale_dense_lists_and_follow_package_replacement(self):
        script = r"""
const fs = require('fs'), path = require('path');
const VM = require(path.join(process.env.BONSAI_TURBOWARP_VM, 'src/index.js'));
const Thread = require(path.join(process.env.BONSAI_TURBOWARP_VM, 'src/engine/thread.js'));
const Storage = require('scratch-storage');
let attempts = [];
const compile = Thread.prototype.tryCompile;
Thread.prototype.tryCompile = function () { compile.call(this); attempts.push(Boolean(this.isCompiled)); };
(async () => {
    const vm = new VM(); vm.attachStorage(new Storage());
    vm.setCompatibilityMode(false); vm.setTurboMode(true); vm.setCompilerOptions({enabled: true});
    let loader, compileError;
    vm.on('COMPILE_ERROR', (target, error) => { compileError = error; });
    const Scratch = {vm, extensions: {unsandboxed: true, register: x => { loader = x; }},
        BlockType: {COMMAND: 'command', REPORTER: 'reporter'}, ArgumentType: {STRING: 'string', NUMBER: 'number'}};
    require('node:vm').runInNewContext(fs.readFileSync(process.argv[2], 'utf8'), {Scratch, TextDecoder});
    const pkg = await fs.openAsBlob(process.env.BONSAI_MODEL_PACKAGE);
    await loader.openFile(pkg);
    if (loader.fileStatus() !== 'ok') throw Error(loader.fileStatus());
    vm.extensionManager.addBuiltinExtension('bonsailoader', class { constructor() { return loader; } });
    vm.start(); await vm.loadProject(fs.readFileSync(process.argv[1]));
    const stage = vm.runtime.getTargetForStage();
    const scalar = name => stage.lookupVariableByNameAndType('BS ' + name, '');
    const list = name => stage.lookupVariableByNameAndType('BS ' + name, 'list');
    async function run(mode) {
        scalar('mode').value = mode; scalar('done').value = 0;
        await vm.greenFlag();
        while (Number(scalar('done').value) !== 1) {
            if (compileError) throw compileError;
            await new Promise(r => setTimeout(r, 5));
        }
        return list('output').value.slice();
    }
    scalar('token id').value = 1;
    const emb = await run(15);
    list('signs').value = Array(6144).fill(0);
    const embShort = await run(15);
    list('signs').value = Array(2048).fill(0);
    const embSame = await run(15);
    list('input').value = Array(2048).fill(1);
    const norm = await run(16);
    list('norm weights').value = Array(512).fill(0);
    const normShort = await run(16);
    list('norm weights').value = Array(2048).fill(0);
    const normSame = await run(16);
    const header = Buffer.from(await pkg.slice(0, 16).arrayBuffer());
    const mlen = Number(header.readBigUInt64LE(8));
    const manifest = JSON.parse(await pkg.slice(16, 16 + mlen).text());
    const changes = [];
    for (const [name, factor] of [['prism.hadamard.signs.2048', -1], ['output_norm.weight', 2]]) {
        const tensor = manifest.tensors.find(t => t.name === name);
        const start = 16 + mlen + tensor.offset;
        const data = Buffer.from(await pkg.slice(start, start + tensor.length).arrayBuffer());
        for (let i = 0; i < data.length; i += 4) data.writeFloatLE(data.readFloatLE(i) * factor, i);
        changes.push({start, data});
    }
    changes.sort((a, b) => a.start - b.start);
    const parts = []; let offset = 0;
    for (const change of changes) {
        parts.push(pkg.slice(offset, change.start), change.data);
        offset = change.start + change.data.length;
    }
    parts.push(pkg.slice(offset));
    await loader.openFile(new Blob(parts));
    if (loader.fileStatus() !== 'ok') throw Error(loader.fileStatus());
    const embReplaced = await run(15), normReplaced = await run(16);
    if (!attempts.length || attempts.some(x => !x)) throw Error('Scratch thread did not compile');
    process.stdout.write(JSON.stringify({emb, embShort, embSame, embReplaced, norm, normShort, normSame, normReplaced}));
    vm.quit();
})().catch(e => { console.error(e); process.exit(1); });
"""
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp) / 'reload.sb3'
            generated = subprocess.run([sys.executable, str(GENERATOR), str(project), '--benchmark'],
                                       capture_output=True, text=True)
            self.assertEqual(generated.returncode, 0, generated.stderr)
            run = subprocess.run(['node', '-e', script, str(project), str(ROOT / 'extension/loader.js')],
                                 capture_output=True, text=True, timeout=60)
            self.assertEqual(run.returncode, 0, run.stderr)
            result = json.loads(run.stdout)
            for prefix in ('emb', 'norm'):
                self.assertEqual(len(result[prefix]), 2048)
                self.assertEqual(result[prefix + 'Short'], result[prefix])
                self.assertEqual(result[prefix + 'Same'], result[prefix])
                factor = -1 if prefix == 'emb' else 2
                for actual, original in zip(result[prefix + 'Replaced'], result[prefix]):
                    self.assertAlmostEqual(actual, factor * original, delta=1e-6)


if __name__ == '__main__':
    unittest.main()
