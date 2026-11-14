import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(os.environ.get('BONSAI_TURBOWARP_VM') and os.environ.get('BONSAI_MODEL_PACKAGE'),
                     'set BONSAI_TURBOWARP_VM and BONSAI_MODEL_PACKAGE')
class StreamedHeadTests(unittest.TestCase):
    def test_split_tiles_match_real_native_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp) / 'head.sb3'
            generated = subprocess.run([sys.executable, str(ROOT / 'tools/generate_project.py'), str(project), '--benchmark'],
                                       capture_output=True, text=True)
            self.assertEqual(generated.returncode, 0, generated.stderr)
            run = subprocess.run(['node', str(ROOT / 'tests/scratch/run-vm.js'), str(project), '--head', '5',
                                  'blk.3.attn_q.weight', '2'], capture_output=True, text=True, timeout=60)
            self.assertEqual(run.returncode, 0, run.stderr)
            actual = json.loads(run.stdout)
            expected = struct.unpack('<5f', (ROOT / 'fixtures/native/layer3-q-first5.f32').read_bytes())
            self.assertEqual(len(actual), 5)
            for got, want in zip(actual, expected):
                self.assertLess(abs(got - want), 1e-6)

    def test_wide_transformed_rows_match_native(self):
        env = dict(os.environ, BONSAI_ACTIVATION_F32=str(ROOT / 'fixtures/native/layer0-ffn-down-input.f32'))
        expected = struct.unpack('<2048f', (ROOT / 'fixtures/native/layer0-ffn-down-output.f32').read_bytes())
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp) / 'wide.sb3'
            generated = subprocess.run([sys.executable, str(ROOT / 'tools/generate_project.py'), str(project), '--benchmark'],
                                       capture_output=True, text=True)
            self.assertEqual(generated.returncode, 0, generated.stderr)
            run = subprocess.run(['node', str(ROOT / 'tests/scratch/run-vm.js'), str(project), '--head', '5',
                                  'blk.0.ffn_down.weight', '2'], env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(run.returncode, 0, run.stderr)
            actual = json.loads(run.stdout)
            self.assertEqual(len(actual), 5)
            for got, want in zip(actual, expected[:5]):
                self.assertLess(abs(got - want), 1e-6)

    def test_stateful_holdout_projection_tolerances(self):
        cases = [('q3', 4096, 'blk.3.attn_q.weight', 2e-6, 4e-7),
                 ('ffn0', 2048, 'blk.0.ffn_down.weight', 2e-7, 4e-8)]
        fixtures = ROOT / 'fixtures/native/operators'
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp) / 'holdout.sb3'
            generated = subprocess.run([sys.executable, str(ROOT / 'tools/generate_project.py'), str(project), '--benchmark'],
                                       capture_output=True, text=True)
            self.assertEqual(generated.returncode, 0, generated.stderr)
            for name, rows, tensor, max_abs, max_rms in cases:
                with self.subTest(tensor=tensor):
                    env = dict(os.environ, BONSAI_ACTIVATION_F32=str(fixtures / f'{name}-1-input.f32'))
                    run = subprocess.run(['node', str(ROOT / 'tests/scratch/run-vm.js'), str(project), '--head',
                                          str(rows), tensor, '120'], env=env, capture_output=True, text=True, timeout=60)
                    self.assertEqual(run.returncode, 0, run.stderr)
                    actual = json.loads(run.stdout)
                    native = struct.unpack(f'<{rows}f', (fixtures / f'{name}-1-output.f32').read_bytes())
                    self.assertEqual(len(actual), rows)
                    errors = [a - b for a, b in zip(actual, native)]
                    self.assertLessEqual(max(map(abs, errors)), max_abs)
                    self.assertLessEqual(math.sqrt(math.fsum(x*x for x in errors) / rows), max_rms)


    def test_split_tiles_match_single_largest_tile(self):
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp) / 'head.sb3'
            generated = subprocess.run([sys.executable, str(ROOT / 'tools/generate_project.py'), str(project), '--benchmark'],
                                       capture_output=True, text=True)
            self.assertEqual(generated.returncode, 0, generated.stderr)
            outputs = []
            for tile in (1920, 4096):
                run = subprocess.run(['node', str(ROOT / 'tests/scratch/run-vm.js'), str(project), '--head', '4096',
                                      'blk.3.attn_q.weight', str(tile)], capture_output=True, text=True, timeout=60)
                self.assertEqual(run.returncode, 0, run.stderr)
                outputs.append(json.loads(run.stdout))
            self.assertEqual(outputs[0], outputs[1])

    def test_cancellation_aborts_at_tile_boundary(self):
        script = r"""
const fs = require('fs'), path = require('path');
const {performance} = require('node:perf_hooks');
const VM = require(path.join(process.env.BONSAI_TURBOWARP_VM, 'src/index.js'));
const Storage = require('scratch-storage');
(async () => {
    const vm = new VM();
    vm.attachStorage(new Storage());
    vm.setCompatibilityMode(false);
    vm.setTurboMode(true);
    vm.setCompilerOptions({enabled: true});
    let loader;
    const Scratch = {vm, extensions: {unsandboxed: true, register: x => { loader = x; }},
        BlockType: {COMMAND: 'command', REPORTER: 'reporter'}, ArgumentType: {STRING: 'string', NUMBER: 'number'}};
    require('node:vm').runInNewContext(fs.readFileSync(process.argv[2], 'utf8'), {Scratch, TextDecoder});
    await loader.openFile(await fs.openAsBlob(process.env.BONSAI_MODEL_PACKAGE));
    vm.extensionManager.addBuiltinExtension('bonsailoader', class { constructor() { return loader; } });
    vm.start();
    await vm.loadProject(fs.readFileSync(process.argv[1]));
    const stage = vm.runtime.getTargetForStage();
    stage.lookupVariableByNameAndType('BS codec', '').value = loader.codec() === 'raw' ? 0 : loader.codec() === 'bslut4' ? 1 : 2;
    stage.lookupVariableByNameAndType('BS head rows', '').value = 248320;
    stage.lookupVariableByNameAndType('BS tile size', '').value = 1280;
    const done = stage.lookupVariableByNameAndType('BS done', '');
    const cancel = stage.lookupVariableByNameAndType('BS cancel', '');
    const orig = loader.loadGroups.bind(loader);
    let requests = 0, cancelAt = 0;
    loader.loadGroups = (...args) => {
        requests++;
        const res = orig(...args);
        if (requests === 4) { cancel.value = 1; cancelAt = performance.now(); }
        return res;
    };
    await vm.greenFlag();
    while (Number(done.value) !== 1) await new Promise(r => setTimeout(r, 5));
    const elapsedMs = performance.now() - cancelAt;
    const scanned = Number(stage.lookupVariableByNameAndType('BS scanned', '').value);
    const output = stage.lookupVariableByNameAndType('BS output', 'list').value;
    process.stdout.write(JSON.stringify({requests, scanned, elapsedMs, tail: output[output.length - 1]}) + '\n');
    vm.quit();
})().catch(e => { console.error(e); process.exit(1); });
"""
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp) / 'cancel.sb3'
            generated = subprocess.run([sys.executable, str(ROOT / 'tools/generate_project.py'), str(project), '--benchmark'],
                                       capture_output=True, text=True)
            self.assertEqual(generated.returncode, 0, generated.stderr)
            run = subprocess.run(['node', '-e', script, str(project), str(ROOT / 'extension/loader.js')],
                                 capture_output=True, text=True, timeout=60)
            self.assertEqual(run.returncode, 0, run.stderr)
            info = json.loads(run.stdout)
            self.assertEqual(info['requests'], 4)
            self.assertEqual(info['scanned'], 4 * 1280)
            self.assertEqual(info['tail'], 'error')
            self.assertLess(info['elapsedMs'], 250.0)

if __name__ == '__main__':
    unittest.main()
