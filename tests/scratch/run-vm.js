'use strict';

const fs = require('fs');
const path = require('path');
const root = process.env.BONSAI_TURBOWARP_VM;
if (!root) throw new Error('BONSAI_TURBOWARP_VM must name the pinned VM checkout');
const VirtualMachine = require(path.join(root, 'src/index.js'));
const Thread = require(path.join(root, 'src/engine/thread.js'));
const ScratchStorage = require('scratch-storage');

const attempts = [];
const tryCompile = Thread.prototype.tryCompile;
Thread.prototype.tryCompile = function () {
    tryCompile.call(this);
    attempts.push(Boolean(this.isCompiled));
};

async function run() {
    const vm = new VirtualMachine();
    let compileError = null;
    vm.on('COMPILE_ERROR', (target, error) => { compileError = error; });
    vm.attachStorage(new ScratchStorage());
    vm.setCompatibilityMode(false);
    vm.setTurboMode(true);
    vm.setCompilerOptions({enabled: true});
    let loader, heartbeat, loaderReadMs = 0, loaderCommandMs = 0, loaderRequests = 0;
    const m4 = process.argv[3] === '--m4';
    const head = process.argv[3] === '--head' || m4;
    const stream = head || process.argv[3] === '--stream';
    if (process.argv[3] === '--package' || process.argv[3] === '--dot' || stream) {
        const Scratch = {vm, extensions: {unsandboxed: true, register: value => { loader = value; }},
            BlockType: {COMMAND: 'command', REPORTER: 'reporter'},
            ArgumentType: {STRING: 'string', NUMBER: 'number'}};
        const source = fs.readFileSync(path.join(__dirname, '../../extension/loader.js'), 'utf8');
        require('node:vm').runInNewContext(source, {Scratch, TextDecoder});
        await loader.openFile(await fs.openAsBlob(process.env.BONSAI_MODEL_PACKAGE));
        if (loader.fileStatus() !== 'ok') throw new Error(loader.fileStatus());
        if (head) {
            const read = loader.read.bind(loader);
            loader.read = async (...args) => {
                const start = performance.now();
                try { return await read(...args); } finally { loaderReadMs += performance.now() - start; }
            };
            const load = loader.loadGroups.bind(loader);
            loader.loadGroups = (...args) => {
                const start = performance.now();
                loaderRequests++;
                const result = load(...args);
                if (result && typeof result.then === 'function') {
                    return result.finally(() => { loaderCommandMs += performance.now() - start; });
                }
                loaderCommandMs += performance.now() - start;
                return result;
            };
        }
        if (!stream) {
            const loadGroups = loader.loadGroups.bind(loader);
            loader.loadGroups = async (...args) => {
                await loadGroups(...args);
                loader.read = () => { throw new Error('loader disabled'); };
                loader.file = null;
            };
        }
        vm.extensionManager.addBuiltinExtension('bonsailoader', class { constructor() { return loader; } });
    }
    vm.runtime.setFramerate(Number(process.env.BONSAI_FRAMERATE || 250));
    vm.start();
    try {
        await vm.loadProject(fs.readFileSync(process.argv[2]));
        const stage = vm.runtime.getTargetForStage();
        const input = stage.lookupVariableByNameAndType('BS input', 'list');
        const output = stage.lookupVariableByNameAndType('BS output', 'list');
        const done = stage.lookupVariableByNameAndType('BS done', '');
        if (!input || !output || !done) throw new Error('project test variables are missing');
        if (!loader) input.value = JSON.parse(process.argv[3]);
        if (process.argv[3] === '--dot' || stream || process.env.BONSAI_ACTIVATION_F32) {
            const bytes = fs.readFileSync(process.env.BONSAI_ACTIVATION_F32 || path.join(__dirname, '../../fixtures/native/layer3-q-input.f32'));
            if (bytes.length % 4) throw new Error('invalid F32 fixture');
            input.value = Array.from({length: bytes.length / 4}, (_, i) => bytes.readFloatLE(i * 4));
        }
        if (process.env.BONSAI_SIGNS_F32) {
            const signs = stage.lookupVariableByNameAndType('BS signs', 'list');
            if (!signs) throw new Error('project signs list is missing');
            const bytes = fs.readFileSync(process.env.BONSAI_SIGNS_F32);
            if (bytes.length % 4) throw new Error('invalid F32 signs fixture');
            signs.value = Array.from({length: bytes.length / 4}, (_, i) => bytes.readFloatLE(i * 4));
        }
        if (process.env.BONSAI_NORM_WEIGHTS_F32) {
            const nw = stage.lookupVariableByNameAndType('BS norm weights', 'list');
            if (!nw) throw new Error('project norm weights list is missing');
            const bytes = fs.readFileSync(process.env.BONSAI_NORM_WEIGHTS_F32);
            if (bytes.length % 4) throw new Error('invalid F32 norm weights fixture');
            nw.value = Array.from({length: bytes.length / 4}, (_, i) => bytes.readFloatLE(i * 4));
        }
        if (process.env.BONSAI_TOKEN_ID !== undefined) {
            const tid = stage.lookupVariableByNameAndType('BS token id', '');
            if (!tid) throw new Error('project token id variable is missing');
            tid.value = Number(process.env.BONSAI_TOKEN_ID);
        }
        if (process.argv[3] === '--dot' || head) {
            const codec = stage.lookupVariableByNameAndType('BS codec', '');
            if (!codec) throw new Error('project codec variable is missing');
            codec.value = loader.codec() === 'raw' ? 0 : loader.codec() === 'bslut4' ? 1 : 2;
        }
        done.value = 0;
        const mode = stage.lookupVariableByNameAndType('BS mode', '');
        if (process.argv[4] && !mode) throw new Error('project mode variable is missing');
        if (mode) mode.value = m4 ? 17 : process.argv[3] === '--head' ? 11 : process.argv[3] === '--dot' ? Number(process.argv[4] || 2) : Number(process.argv[4] || 0);
        if (head) {
            stage.lookupVariableByNameAndType('BS head rows', '').value = Number(process.argv[4]);
            stage.lookupVariableByNameAndType('BS head tensor', '').value = process.argv[5];
            stage.lookupVariableByNameAndType('BS tile size', '').value = Number(process.argv[6] || 120);
        }
        const iterations = stage.lookupVariableByNameAndType('BS iterations', '');
        if (!head && process.argv[5]) {
            if (!iterations) throw new Error('project iteration variable is missing');
            iterations.value = Number(process.argv[5]);
        }
        let lastTick = performance.now(), worstGap = 0;
        heartbeat = setInterval(() => {
            const now = performance.now();
            worstGap = Math.max(worstGap, now - lastTick);
            lastTick = now;
        }, 5);
        await vm.greenFlag();
        await new Promise((resolve, reject) => {
            const deadline = Date.now() + Number(process.env.BONSAI_TIMEOUT_MS || 30000);
            const timer = setInterval(() => {
                if (compileError || Date.now() > deadline) {
                    clearInterval(timer);
                    reject(compileError || new Error('project timed out'));
                } else if (Number(done.value) === 1) {
                    clearInterval(timer);
                    resolve();
                }
            }, 5);
        });
        clearInterval(heartbeat);
        worstGap = Math.max(worstGap, performance.now() - lastTick);
        if (!attempts.length || attempts.some(compiled => !compiled)) throw new Error('a Scratch thread did not compile');
        if (loader && !stream && (loader.status({ID: 'resident'}) !== 'ok' || loader.copied({ID: 'resident'}) !== (process.argv[3] === '--dot' ? input.value.length / 128 : 5) || loader.file !== null)) {
            throw new Error('compiled loader block did not complete before isolation');
        }
        if (head && (Number(stage.lookupVariableByNameAndType('BS head valid', '').value) !== 1 ||
                     Number(stage.lookupVariableByNameAndType('BS scanned', '').value) !== Number(process.argv[4]))) {
            throw new Error(`head scan failed: ${loader.status({ID: 'head'})} ${loader.error({ID: 'head'})}`);
        }
        const greedyVar = stage.lookupVariableByNameAndType('BS greedy token', '');
        const greedyLogitVar = stage.lookupVariableByNameAndType('BS greedy logit', '');
        const secondLogitVar = stage.lookupVariableByNameAndType('BS second logit', '');
        const scannedVar = stage.lookupVariableByNameAndType('BS scanned', '');
        if (process.env.BONSAI_BENCHMARK) {
            const get = name => stage.lookupVariableByNameAndType(`BS benchmark ${name}`, '').value;
            process.stdout.write(JSON.stringify({output: output.value, quant_tick_seconds: get('quant'),
                setup_tick_seconds: get('setup'), compute_tick_seconds: get('compute'),
                scratch_transfer_and_scheduler_seconds: head ? get('transfer') : null,
                framerate: vm.runtime.frameLoop ? vm.runtime.frameLoop.framerate : null,
                scanned: scannedVar ? Number(scannedVar.value) : null,
                greedy_token: m4 && greedyVar ? Number(greedyVar.value) : null,
                greedy_logit: m4 && greedyLogitVar ? Number(greedyLogitVar.value) : null,
                second_logit: m4 && secondLogitVar ? Number(secondLogitVar.value) : null,
                loader_requests: loaderRequests, loader_read_ms: loaderReadMs,
                loader_command_ms: loaderCommandMs,
                worst_event_loop_gap_ms: worstGap, peak_rss_bytes: process.resourceUsage().maxRSS * 1024}) + '\n');
        } else if (m4) {
            process.stdout.write(JSON.stringify({
                scanned: Number(scannedVar.value),
                greedy_token: Number(greedyVar.value),
                greedy_logit: Number(greedyLogitVar.value),
                second_logit: Number(secondLogitVar.value),
                logits: output.value,
            }) + '\n');
        } else {
            process.stdout.write(JSON.stringify(output.value) + '\n');
        }
    } finally {
        clearInterval(heartbeat);
        vm.quit();
    }
}

run().catch(error => { console.error(error); process.exitCode = 1; });
