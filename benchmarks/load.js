'use strict';

// Stream every package tensor through the real loader into Scratch lists in the
// tile shapes the decoder uses. No Scratch script runs, so this is the data
// movement floor: it excludes the per-request frame parking the VM adds.
const fs = require('node:fs');
const path = require('node:path');
const {performance} = require('node:perf_hooks');
const sandbox = require('node:vm');
const root = process.env.BONSAI_TURBOWARP_VM;
if (!root || !process.env.BONSAI_MODEL_PACKAGE || process.argv.length < 3) {
    throw new Error('Usage: BONSAI_MODEL_PACKAGE=<pkg> BONSAI_TURBOWARP_VM=<checkout> node load.js <project.sb3>');
}
const VirtualMachine = require(path.join(root, 'src/index.js'));
const ScratchStorage = require('scratch-storage');

async function main() {
    const vm = new VirtualMachine();
    vm.attachStorage(new ScratchStorage());
    vm.start();
    try {
        let loader;
        const Scratch = {vm, extensions: {unsandboxed: true, register: x => { loader = x; }},
            BlockType: {COMMAND: 'command', REPORTER: 'reporter'},
            ArgumentType: {STRING: 'string', NUMBER: 'number'}};
        sandbox.runInNewContext(fs.readFileSync(path.join(__dirname, '../extension/loader.js'), 'utf8'),
            {Scratch, TextDecoder});
        vm.extensionManager.addBuiltinExtension('bonsailoader', class { constructor() { return loader; } });
        await vm.loadProject(fs.readFileSync(process.argv[2]));
        const stage = vm.runtime.getTargetForStage();
        const tileRows = Number(process.env.BONSAI_TILE_ROWS || 64);
        const openStart = performance.now();
        await loader.openFile(await fs.openAsBlob(process.env.BONSAI_MODEL_PACKAGE));
        if (loader.fileStatus() !== 'ok') throw new Error(loader.fileStatus());
        const openMs = performance.now() - openStart;
        const read = loader.read.bind(loader);
        let readMs = 0;
        loader.read = async (...args) => {
            const start = performance.now();
            try { return await read(...args); } finally { readMs += performance.now() - start; }
        };
        if (loader.prefetch) await loader.prefetch.promise;
        const start = performance.now();
        let requestMs = 0, requests = 0, groups = 0, bytes = 0, rows = 0;
        const tensors = [];
        for (const [name, tensor] of loader.tensors) {
            if (tensor.type_id !== 142) continue;
            const rowGroups = Math.round(tensor.length / tensor.dimensions[0] / 34);
            const tensorRows = tensor.dimensions[0];
            for (let row = 0; row < tensorRows; row += tileRows) {
                const count = Math.min(tileRows, tensorRows - row) * rowGroups;
                const begin = performance.now();
                const returned = loader.loadGroups({ID: 'load', TENSOR: name, START: row * rowGroups,
                    COUNT: count, Q: 'l_packed', SCALE: 'l_scales'}, {target: stage});
                if (returned && typeof returned.then === 'function') await returned;
                requestMs += performance.now() - begin;
                requests += 1;
                groups += count;
            }
            tensors.push({name, rows: tensorRows, groups: tensor.length / 34});
            rows += tensorRows;
            bytes += tensor.length;
        }
        const streamMs = performance.now() - start;
        const report = {
            schema: 'bonsai-load-all/v1', codec: loader.codec(), tile_rows: tileRows,
            tensors: tensors.length, rows, groups, bytes, requests,
            open_ms: openMs, stream_ms: streamMs, request_ms: requestMs, read_ms: readMs,
            window_bytes: loader.windowBytes,
            payload_length: loader.manifest.payload_length,
            peak_rss_bytes: process.resourceUsage().maxRSS * 1024,
            note: 'Loader data movement only in 64-row decoder tiles; the Scratch driver adds one VM frame per request.'};
        process.stdout.write(JSON.stringify(report) + '\n');
    } finally { vm.quit(); }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
