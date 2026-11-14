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
    stage.lookupVariableByNameAndType('BS tile size', '').value = 640;
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
            self.assertEqual(info['scanned'], 4 * 640)
            self.assertEqual(info['tail'], 'error')
            self.assertLess(info['elapsedMs'], 250.0)


@unittest.skipUnless(os.environ.get('BONSAI_TURBOWARP_VM'), 'set BONSAI_TURBOWARP_VM')
class TopKTests(unittest.TestCase):
    def test_topk_ties_cutoff_and_short_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp) / 'topk.sb3'
            generated = subprocess.run([sys.executable, str(ROOT / 'tools/generate_project.py'), str(project)],
                                       capture_output=True, text=True)
            self.assertEqual(generated.returncode, 0, generated.stderr)
            cases = [
                ([5.0, 5.0, 4.0], [(0, 5.0), (1, 5.0), (2, 4.0)]),
                ([4.0, 5.000001, 5.0], [(1, 5.000001), (2, 5.0), (0, 4.0)]),
                ([2, 5, 5, 1, 4, 3, 5, 0, 4, 5, 5, -1],
                 [(1, 5), (2, 5), (6, 5), (9, 5), (10, 5), (4, 4), (8, 4), (5, 3)]),
                ([1] * 10, [(i, 1) for i in range(8)]),
                ([-3e38, -3.1e38], [(0, -3e38), (1, -3.1e38)]),
                ([], []),
            ]
            for logits, expected in cases:
                with self.subTest(logits=logits):
                    run = subprocess.run(['node', str(ROOT / 'tests/scratch/run-vm.js'),
                                          str(project), json.dumps(logits), '18'],
                                         capture_output=True, text=True, timeout=30)
                    self.assertEqual(run.returncode, 0, run.stderr)
                    result = json.loads(run.stdout)
                    self.assertEqual(list(zip(result['top_k_tokens'], result['top_k_logits'])), expected)
                    if expected:
                        self.assertEqual(result['output'][0:2], list(expected[0]))
                        if len(expected) > 1:
                            self.assertEqual(result['output'][2], expected[1][1])


@unittest.skipUnless(os.environ.get('BONSAI_TURBOWARP_VM') and os.environ.get('BONSAI_MODEL_PACKAGE'),
                     'set BONSAI_TURBOWARP_VM and BONSAI_MODEL_PACKAGE')
class PipelineHeadTests(unittest.TestCase):
    def test_m4_end_to_end_head_matches_native_greedy_and_topk(self):
        m4_head = ROOT / 'fixtures/native/m4/head'
        manifest = json.loads((m4_head / 'manifest.json').read_text())
        native_manifest = os.environ.get('BONSAI_NATIVE_MANIFEST')
        native_logits = os.environ.get('BONSAI_NATIVE_LOGITS')
        self.assertEqual(bool(native_manifest), bool(native_logits),
                         'set both BONSAI_NATIVE_MANIFEST and BONSAI_NATIVE_LOGITS')
        native = None
        if native_manifest:
            from benchmarks.head import load_m4_native
            native, _ = load_m4_native(manifest, json.loads(Path(native_manifest).read_text()), Path(native_logits))
        env = dict(os.environ, BONSAI_ACTIVATION_F32=str(m4_head / manifest['pre_norm']['file']),
                   BONSAI_TIMEOUT_MS='600000', BONSAI_BENCHMARK='1')
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp) / 'm4-head.sb3'
            generated = subprocess.run([sys.executable, str(ROOT / 'tools/generate_project.py'), str(project), '--benchmark'],
                                       capture_output=True, text=True)
            self.assertEqual(generated.returncode, 0, generated.stderr)
            run = subprocess.run(['node', str(ROOT / 'tests/scratch/run-vm.js'), str(project),
                                  '--m4', '248320', 'token_embd.weight', '640'],
                                 env=env, capture_output=True, text=True, timeout=620)
            self.assertEqual(run.returncode, 0, run.stderr)
            result = json.loads(run.stdout)
            logits = result['output']
            self.assertEqual(result['scanned'], 248320)
            self.assertEqual(len(logits), 248320)
            self.assertEqual(result['greedy_token'], manifest['greedy_token'])
            self.assertEqual(result['top_k_tokens'], [entry['token'] for entry in manifest['top_k']])
            self.assertEqual(result['top_k_logits'], [logits[token] for token in result['top_k_tokens']])
            max_err = max(abs(got - want['logit'])
                          for got, want in zip(result['top_k_logits'], manifest['top_k']))
            margin = result['greedy_logit'] - result['second_logit']
            self.assertLessEqual(max_err, 1e-5)
            self.assertGreater(margin, 2 * max_err)
            if native is not None:
                full_max_err = max(abs(got - want) for got, want in zip(logits, native))
                self.assertGreater(manifest['margin_top1_top2'], 2 * full_max_err)

    def test_m4_cancellation_aborts_at_tile_boundary(self):
        script = r"""
const fs = require('fs'), path = require('path');
const {performance} = require('node:perf_hooks');
const VM = require(path.join(process.env.BONSAI_TURBOWARP_VM, 'src/index.js'));
const Storage = require('scratch-storage');
(async () => {
    const vm = new VM(); vm.attachStorage(new Storage());
    vm.setCompatibilityMode(false); vm.setTurboMode(true); vm.setCompilerOptions({enabled: true});
    let loader;
    const Scratch = {vm, extensions: {unsandboxed: true, register: x => { loader = x; }},
        BlockType: {COMMAND: 'command', REPORTER: 'reporter'}, ArgumentType: {STRING: 'string', NUMBER: 'number'}};
    require('node:vm').runInNewContext(fs.readFileSync(process.argv[2], 'utf8'), {Scratch, TextDecoder});
    await loader.openFile(await fs.openAsBlob(process.env.BONSAI_MODEL_PACKAGE));
    vm.extensionManager.addBuiltinExtension('bonsailoader', class { constructor() { return loader; } });
    vm.start(); await vm.loadProject(fs.readFileSync(process.argv[1]));
    const stage = vm.runtime.getTargetForStage();
    const bytes = fs.readFileSync(process.argv[3]);
    stage.lookupVariableByNameAndType('BS input', 'list').value =
        Array.from({length: bytes.length / 4}, (_, i) => bytes.readFloatLE(i * 4));
    stage.lookupVariableByNameAndType('BS mode', '').value = 17;
    stage.lookupVariableByNameAndType('BS head rows', '').value = 248320;
    stage.lookupVariableByNameAndType('BS tile size', '').value = 640;
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
            project = Path(temp) / 'cancel-m4.sb3'
            generated = subprocess.run([sys.executable, str(ROOT / 'tools/generate_project.py'), str(project), '--benchmark'],
                                       capture_output=True, text=True)
            self.assertEqual(generated.returncode, 0, generated.stderr)
            pre_norm = ROOT / 'fixtures/native/m4/head/output-norm-input.f32'
            run = subprocess.run(['node', '-e', script, str(project), str(ROOT / 'extension/loader.js'), str(pre_norm)],
                                 capture_output=True, text=True, timeout=60)
            self.assertEqual(run.returncode, 0, run.stderr)
            info = json.loads(run.stdout)
            self.assertEqual(info['requests'], 4)
            self.assertEqual(info['scanned'], 4 * 640)
            self.assertEqual(info['tail'], 'error')
            self.assertLess(info['elapsedMs'], 250.0)

    def test_pipeline_reloads_dense_lists_and_rejects_invalid_head_and_precancel(self):
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
    await loader.openFile(await fs.openAsBlob(process.env.BONSAI_MODEL_PACKAGE));
    if (loader.fileStatus() !== 'ok') throw Error(loader.fileStatus());
    vm.extensionManager.addBuiltinExtension('bonsailoader', class { constructor() { return loader; } });
    vm.start(); await vm.loadProject(fs.readFileSync(process.argv[1]));
    const stage = vm.runtime.getTargetForStage();
    const scalar = name => stage.lookupVariableByNameAndType('BS ' + name, '');
    const list = name => stage.lookupVariableByNameAndType('BS ' + name, 'list');
    async function run(size, tile, cancel, mode = 17, tensor = 'token_embd.weight') {
        list('input').value = Array(size).fill(1);
        scalar('mode').value = mode; scalar('head rows').value = 1;
        scalar('head tensor').value = tensor;
        scalar('tile size').value = tile;
        if (cancel !== null) scalar('cancel').value = cancel;
        scalar('greedy token').value = 777; scalar('done').value = 0;
        await vm.greenFlag();
        while (Number(scalar('done').value) !== 1) {
            if (compileError) throw compileError;
            await new Promise(r => setTimeout(r, 5));
        }
        return {output: list('output').value.slice(), valid: Number(scalar('head valid').value),
                token: Number(scalar('greedy token').value), top: list('top tokens').value.slice()};
    }
    const baseline = await run(2048, 1, 0);
    list('signs').value = Array(512).fill(0); list('norm weights').value = Array(512).fill(0);
    const short = await run(2048, 1, 0);
    list('signs').value = Array(2048).fill(0); list('norm weights').value = Array(2048).fill(0);
    const same = await run(2048, 1, 0);
    const invalid = await run(2048, 0, 1);
    const afterInvalid = await run(2048, 1, null);
    const invalidInput = await run(1, 1, 1);
    const afterInvalidInput = await run(2048, 1, null);
    const missing = await run(4096, 1, 1);
    const afterMissing = await run(2048, 1, null);
    const cancelled = await run(2048, 1, 1);
    const afterCancel = await run(2048, 1, null);
    const headBaseline = await run(2048, 1, 0, 11);
    const headInvalid = await run(2048, 0, 1, 11);
    const headAfterInvalid = await run(2048, 1, null, 11);
    const loadGroups = loader.loadGroups.bind(loader);
    loader.loadGroups = (...args) => {
        if (args[0].TENSOR === 'missing.tensor') scalar('cancel').value = 1;
        return loadGroups(...args);
    };
    const headFailed = await run(2048, 1, 0, 11, 'missing.tensor');
    const headAfterFailed = await run(2048, 1, null, 11);
    if (!attempts.length || attempts.some(x => !x)) throw Error('Scratch thread did not compile');
    process.stdout.write(JSON.stringify({baseline, short, same, invalid, afterInvalid, invalidInput,
        afterInvalidInput, missing, afterMissing, cancelled, afterCancel,
        headBaseline, headInvalid, headAfterInvalid, headFailed, headAfterFailed}));
    vm.quit();
})().catch(e => { console.error(e); process.exit(1); });
"""
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp) / 'pipeline.sb3'
            generated = subprocess.run([sys.executable, str(ROOT / 'tools/generate_project.py'), str(project), '--benchmark'],
                                       capture_output=True, text=True)
            self.assertEqual(generated.returncode, 0, generated.stderr)
            run = subprocess.run(['node', '-e', script, str(project), str(ROOT / 'extension/loader.js')],
                                 capture_output=True, text=True, timeout=60)
            self.assertEqual(run.returncode, 0, run.stderr)
            result = json.loads(run.stdout)
            self.assertEqual(result['baseline']['valid'], 1)
            self.assertEqual(len(result['baseline']['output']), 1)
            self.assertEqual(result['baseline']['top'], [0])
            self.assertEqual(result['short'], result['baseline'])
            self.assertEqual(result['same'], result['baseline'])
            for name in ('afterCancel', 'afterInvalid', 'afterInvalidInput', 'afterMissing'):
                self.assertEqual(result[name], result['baseline'])
            for name, error in (('invalid', 'invalid dimensions'), ('invalidInput', 'invalid dimensions'),
                                ('cancelled', 'error'), ('missing', 'error')):
                self.assertEqual(result[name], {'output': [error], 'valid': 0, 'token': 777, 'top': []})
            self.assertEqual(result['headInvalid']['output'], ['invalid dimensions'])
            self.assertEqual(result['headFailed']['output'], ['error'])
            self.assertEqual(result['headAfterInvalid'], result['headBaseline'])
            self.assertEqual(result['headAfterFailed'], result['headBaseline'])

if __name__ == '__main__':
    unittest.main()
