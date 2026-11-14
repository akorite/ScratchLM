'use strict';

// Measure the real loader against real TurboWarp list objects, without running model math.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {performance} = require('node:perf_hooks');
const sandbox = require('node:vm');
const root = process.env.BONSAI_TURBOWARP_VM;
if (!root || process.argv.length !== 4) throw new Error('Usage: BONSAI_TURBOWARP_VM=<checkout> node transport.js <package.bsp> <project.sb3>');
const VirtualMachine = require(path.join(root, 'src/index.js'));
const ScratchStorage = require('scratch-storage');

async function main() {
    const vm = new VirtualMachine();
    vm.attachStorage(new ScratchStorage());
    vm.start();
    try {
        await vm.loadProject(fs.readFileSync(process.argv[3]));
        const stage = vm.runtime.getTargetForStage();
        let loader;
        const Scratch = {vm, extensions: {unsandboxed: true, register: x => { loader = x; }},
            BlockType: {COMMAND: 'command', REPORTER: 'reporter'}, ArgumentType: {STRING: 'string', NUMBER: 'number'}};
        sandbox.runInNewContext(fs.readFileSync(path.join(__dirname, '../extension/loader.js'), 'utf8'), {Scratch, TextDecoder});
        const file = await fs.openAsBlob(process.argv[2]);
        await loader.openFile(file);
        assert.equal(loader.fileStatus(), 'ok');
        const tensor = loader.tensors.get('token_embd.weight');
        const first = new Uint8Array(await file.slice(loader.payloadStart + tensor.offset, loader.payloadStart + tensor.offset + 34).arrayBuffer());
        const read = loader.read.bind(loader);
        let readMs = 0;
        loader.read = async (...args) => {
            const start = performance.now();
            try { return await read(...args); } finally { readMs += performance.now() - start; }
        };
        const results = [];
        for (const budget of [65536, 262144, 1048576, 4194304]) {
            const groups = Math.floor(budget / 34);
            for (let repetition = 0; repetition < 3; repetition++) {
                readMs = 0;
                const start = performance.now();
                const returned = loader.loadGroups({ID: 'bench', TENSOR: 'token_embd.weight', START: 0, COUNT: groups,
                    Q: 'l_packed', SCALE: 'l_input'}, {target: stage});
                const sync = !(returned && typeof returned.then === 'function');
                if (returned) await returned;
                const elapsedMs = performance.now() - start;
                const q = stage.lookupVariableById('l_packed').value;
                const scales = stage.lookupVariableById('l_input').value;
                assert.equal(q.length, groups * 32); assert.equal(scales.length, groups);
                assert.equal(q[0], first[2]); assert.equal(scales[0], first[0] + 256 * first[1]);
                results.push({budget_bytes: budget, actual_bytes: groups * 34, groups, repetition, window_hit: sync,
                    read_ms: readMs, copy_and_request_ms: elapsedMs - readMs, total_ms: elapsedMs,
                    observed_rss_bytes: process.memoryUsage().rss});
            }
        }
        const report = {method: 'Node invokes the actual extension on TurboWarp VM list objects; no Scratch script runs',
            cache: 'served from the loader read-ahead window when covered; read_ms>0 only on refill',
            window_bytes: loader.windowBytes,
            node: process.version, vm_revision: '0cb82afe62506f3d9f2ff849c95593f0473b5d2c',
            package_codec: loader.codec(), source_sha256: loader.manifest.source.sha256, results};
        process.stdout.write(JSON.stringify(report, null, 2) + '\n');
    } finally { vm.quit(); }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
