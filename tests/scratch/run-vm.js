'use strict';

const fs = require('fs');
const path = require('path');
const root = process.env.BONSAI_TURBOWARP_VM;
if (!root) throw new Error('BONSAI_TURBOWARP_VM must name the pinned VM checkout');
const VirtualMachine = require(path.join(root, 'src/index.js'));
const Thread = require(path.join(root, 'src/engine/thread.js'));
const ScratchStorage = require('scratch-storage');

const attempts = [];
let compileMs = 0;
const tryCompile = Thread.prototype.tryCompile;
Thread.prototype.tryCompile = function () {
    const started = performance.now();
    tryCompile.call(this);
    compileMs += performance.now() - started;
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
    // modes 22-24 stream tables across many requests; 14-16 load once and then must not read again
const loaderMode = ['22', '23', '24', '40', '41', '42', '43'].includes(process.argv[4]);
    // Projects always declare the storage extension; it only opens a package when one is supplied.
    if (true) {
        const Scratch = {vm, extensions: {unsandboxed: true, register: value => { loader = value; }},
            BlockType: {COMMAND: 'command', REPORTER: 'reporter'},
            ArgumentType: {STRING: 'string', NUMBER: 'number'}};
        const source = fs.readFileSync(path.join(__dirname, '../../extension/loader.js'), 'utf8');
        require('node:vm').runInNewContext(source, {Scratch, TextDecoder});
        if (process.env.BONSAI_MODEL_PACKAGE) {
            await loader.openFile(await fs.openAsBlob(process.env.BONSAI_MODEL_PACKAGE));
            if (loader.fileStatus() !== 'ok') throw new Error(loader.fileStatus());
        }
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
        if (!stream && !loaderMode) {
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
        if (!loader || process.argv[3].startsWith('[')) input.value = JSON.parse(process.argv[3]);
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
        if (process.env.BONSAI_KV_KEY && process.env.BONSAI_KV_VALUE) {
            const tokens = Number(process.env.BONSAI_KV_TOKENS || 1);
            const floats = file => {
                const bytes = fs.readFileSync(file);
                return Array.from({length: bytes.length / 4}, (_, i) => bytes.readFloatLE(i * 4));
            };
            const kList = stage.lookupVariableByNameAndType('BS k cache', 'list');
            const vList = stage.lookupVariableByNameAndType('BS v cache', 'list');
            if (!kList || !vList) throw new Error('project KV cache lists are missing');
            const k = floats(process.env.BONSAI_KV_KEY);
            const v = floats(process.env.BONSAI_KV_VALUE);
            // Scratch keeps token-major cells of 512; the native V cache is [head][dim][cell].
            kList.value = k.slice(0, tokens * 512);
            const out = [];
            for (let t = 0; t < tokens; t++) {
                for (let h = 0; h < 2; h++) {
                    for (let d = 0; d < 256; d++) out.push(v[h * 65536 + d * 256 + t]);
                }
            }
            vList.value = out;
        }
        if (process.env.BONSAI_DN_STATE || process.env.BONSAI_CONV_STATE) {
            const load = (file, name) => {
                const list = stage.lookupVariableByNameAndType(name, 'list');
                if (!list) throw new Error(`project list ${name} is missing`);
                const bytes = fs.readFileSync(file);
                list.value = Array.from({length: bytes.length / 4}, (_, i) => bytes.readFloatLE(i * 4));
            };
            if (process.env.BONSAI_DN_STATE) load(process.env.BONSAI_DN_STATE, 'BS state');
            if (process.env.BONSAI_CONV_STATE) load(process.env.BONSAI_CONV_STATE, 'BS conv state');
        }
        const secondPrompt = process.env.BONSAI_SECOND_PROMPT || null;
        const secondAfter = Number(process.env.BONSAI_SECOND_AFTER || 0);
        if (process.env.BONSAI_KV_CELLS !== undefined) {
            const cells = Number(process.env.BONSAI_KV_CELLS);
            const kCache = stage.lookupVariableByNameAndType('BS k cache', 'list');
            const vCache = stage.lookupVariableByNameAndType('BS v cache', 'list');
            if (!kCache || !vCache) throw new Error('project KV caches are missing');
            kCache.value = new Array(cells * 512).fill(0);
            vCache.value = new Array(cells * 512).fill(0);
            const stepVar = stage.lookupVariableByNameAndType('BS step', '');
            if (stepVar) stepVar.value = cells;
            const positionVar = stage.lookupVariableByNameAndType('BS layer position', '');
            if (positionVar) positionVar.value = cells;
        }
        if (process.env.BONSAI_LOOP !== undefined) {
            const loopVar = stage.lookupVariableByNameAndType('BS loop', '');
            if (loopVar) loopVar.value = Number(process.env.BONSAI_LOOP);
        }
        if (process.env.BONSAI_MAX_TOKENS) {
            const maxTokens = stage.lookupVariableByNameAndType('BS max tokens', '');
            if (maxTokens) maxTokens.value = Number(process.env.BONSAI_MAX_TOKENS);
        }
        if (process.env.BONSAI_OUT_TOKENS) {
            const outList = stage.lookupVariableByNameAndType('BS out tokens', 'list');
            if (!outList) throw new Error('project out-token list is missing');
            outList.value = JSON.parse(process.env.BONSAI_OUT_TOKENS);
        }
        if (process.env.BONSAI_PROMPT !== undefined) {
            const promptVar = stage.lookupVariableByNameAndType('BS prompt', '');
            if (promptVar) promptVar.value = process.env.BONSAI_PROMPT;
            const askVar = stage.lookupVariableByNameAndType('BS ask', '');
            if (askVar) askVar.value = 0;
        }
        if (process.env.BONSAI_PROMPT_TOKENS) {
            const prompt = stage.lookupVariableByNameAndType('BS prompt tokens', 'list');
            if (!prompt) throw new Error('project prompt token list is missing');
            prompt.value = process.env.BONSAI_PROMPT_TOKENS.split(',').map(Number);
        }
        if (process.env.BONSAI_LAYER !== undefined) {
            const layer = stage.lookupVariableByNameAndType('BS layer', '');
            if (!layer) throw new Error('project layer variable is missing');
            layer.value = Number(process.env.BONSAI_LAYER);
        }
        if (process.env.BONSAI_LAYER_POSITION !== undefined) {
            const lp = stage.lookupVariableByNameAndType('BS layer position', '');
            if (!lp) throw new Error('project layer position variable is missing');
            lp.value = Number(process.env.BONSAI_LAYER_POSITION);
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
        const prompts = process.env.BONSAI_PROMPTS ? JSON.parse(process.env.BONSAI_PROMPTS) : null;
        const promptVar = stage.lookupVariableByNameAndType('BS prompt', '');
        const steps = prompts ? prompts.length : Number(process.env.BONSAI_STEPS || 1);
        const produced = [];
        for (let step = 0; step < steps; step++) {
            if (secondPrompt && step === secondAfter) {
                const promptVar2 = stage.lookupVariableByNameAndType('BS prompt', '');
                const startedVar = stage.lookupVariableByNameAndType('BS started', '');
                if (!promptVar2 || !startedVar) throw new Error('project prompt/started variables are missing');
                promptVar2.value = secondPrompt;
                startedVar.value = 0;
                output.value = [];
            }
            if (prompts) {
                if (!promptVar) throw new Error('project prompt variable is missing');
                promptVar.value = prompts[step];
                output.value = [];
            }
            done.value = 0;
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
            const producedToken = stage.lookupVariableByNameAndType('BS greedy token', '');
            if (prompts) produced.push({prompt: prompts[step], tokens: output.value.map(Number)});
            else if (producedToken) produced.push(Number(producedToken.value));
        }
        clearInterval(heartbeat);
        worstGap = Math.max(worstGap, performance.now() - lastTick);
        if (!attempts.length || attempts.some(compiled => !compiled)) throw new Error('a Scratch thread did not compile');
        const packaged = process.argv[3] === '--package' || process.argv[3] === '--dot';
        if (loader && packaged && (loader.status({ID: 'resident'}) !== 'ok' || loader.copied({ID: 'resident'}) !== (process.argv[3] === '--dot' ? input.value.length / 128 : 5) || loader.file !== null)) {
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
        const topTokens = stage.lookupVariableByNameAndType('BS top tokens', 'list');
        const topLogits = stage.lookupVariableByNameAndType('BS top logits', 'list');
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
                top_k_tokens: m4 ? topTokens.value.map(Number) : null,
                top_k_logits: m4 ? topLogits.value.map(Number) : null,
                loader_requests: loaderRequests, loader_read_ms: loaderReadMs,
                loader_command_ms: loaderCommandMs,
                worst_event_loop_gap_ms: worstGap, peak_rss_bytes: process.resourceUsage().maxRSS * 1024}) + '\n');
        } else if (m4) {
            process.stdout.write(JSON.stringify({
                scanned: Number(scannedVar.value),
                greedy_token: Number(greedyVar.value),
                greedy_logit: Number(greedyLogitVar.value),
                second_logit: Number(secondLogitVar.value),
                top_k_tokens: topTokens.value.map(Number),
                top_k_logits: topLogits.value.map(Number),
                logits: output.value,
            }) + '\n');
        } else if (process.env.BONSAI_DEBUG_REQUESTS) {
            const reqs = process.env.BONSAI_DEBUG_REQUESTS.split(',');
            const st = vm.runtime.getTargetForStage();
            process.stdout.write(JSON.stringify(reqs.map(id => ({
                id, status: loader.status({ID: id}), copied: loader.copied({ID: id}),
                error: loader.error({ID: id})}))) + '\n');
        } else if (process.env.BONSAI_DUMP) {
            const [name, file] = process.env.BONSAI_DUMP.split(':');
            const list = vm.runtime.getTargetForStage().lookupVariableByNameAndType(name, 'list');
            fs.writeFileSync(file, JSON.stringify(list.value));
            process.stdout.write('[]\n');
        } else if (process.env.BONSAI_DEBUG_VARS) {
            const names = process.env.BONSAI_DEBUG_VARS.split(',');
            const stage2 = vm.runtime.getTargetForStage();
            const full = process.env.BONSAI_DEBUG_FULL === '1';
            const read = spec => {
                const [name, at] = spec.split('@');
                const l = stage2.lookupVariableByNameAndType(name, 'list');
                const v = stage2.lookupVariableByNameAndType(name, '');
                if (!l) return v ? v.value : null;
                if (full && !at) return l.value.map(Number);
                if (at) return l.value[Number(at) - 1];
                return `list:${l.value.length}:${JSON.stringify(l.value.slice(0, 4))}:${JSON.stringify(l.value.slice(-4))}`;
            };
            const state = Object.fromEntries(names.map(n => [n, read(n)]));
            state.peak_rss_bytes = process.resourceUsage().maxRSS * 1024;
            state.compile_ms = compileMs;
            process.stdout.write(JSON.stringify(state) + '\n');
        } else if (prompts) {
            process.stdout.write(JSON.stringify({prompts: produced}) + '\n');
        } else if (steps > 1) {
            process.stdout.write(JSON.stringify({tokens: produced}) + '\n');
        } else if (process.argv[4] === '42') {
            const greedy = stage.lookupVariableByNameAndType('BS greedy token', '');
            process.stdout.write(JSON.stringify({token: Number(greedy.value)}) + '\n');
        } else if (Number(mode.value) === 18) {
            process.stdout.write(JSON.stringify({output: output.value,
                top_k_tokens: topTokens.value.map(Number), top_k_logits: topLogits.value.map(Number)}) + '\n');
        } else {
            process.stdout.write(JSON.stringify(output.value) + '\n');
        }
    } finally {
        clearInterval(heartbeat);
        vm.quit();
    }
}

run().catch(error => { console.error(error); process.exitCode = 1; });
