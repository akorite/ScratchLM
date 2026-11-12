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
    vm.start();
    vm.setCompatibilityMode(false);
    vm.setTurboMode(true);
    vm.setCompilerOptions({enabled: true});
    let loader;
    if (process.argv[3] === '--package' || process.argv[3] === '--dot') {
        const Scratch = {vm, extensions: {unsandboxed: true, register: value => { loader = value; }},
            BlockType: {COMMAND: 'command', REPORTER: 'reporter'},
            ArgumentType: {STRING: 'string', NUMBER: 'number'}};
        const source = fs.readFileSync(path.join(__dirname, '../../extension/loader.js'), 'utf8');
        require('node:vm').runInNewContext(source, {Scratch, TextDecoder});
        await loader.openFile(await fs.openAsBlob(process.env.BONSAI_MODEL_PACKAGE));
        if (loader.fileStatus() !== 'ok') throw new Error(loader.fileStatus());
        const loadGroups = loader.loadGroups.bind(loader);
        loader.loadGroups = async (...args) => {
            await loadGroups(...args);
            loader.read = () => { throw new Error('loader disabled'); };
            loader.file = null;
        };
        vm.extensionManager.addBuiltinExtension('bonsailoader', class { constructor() { return loader; } });
    }
    try {
        await vm.loadProject(fs.readFileSync(process.argv[2]));
        const stage = vm.runtime.getTargetForStage();
        const input = stage.lookupVariableByNameAndType('BS input', 'list');
        const output = stage.lookupVariableByNameAndType('BS output', 'list');
        const done = stage.lookupVariableByNameAndType('BS done', '');
        if (!input || !output || !done) throw new Error('project test variables are missing');
        if (!loader) input.value = JSON.parse(process.argv[3]);
        if (process.argv[3] === '--dot') {
            const bytes = fs.readFileSync(path.join(__dirname, '../../fixtures/native/layer3-q-input.f32'));
            input.value = Array.from({length: bytes.length / 4}, (_, i) => bytes.readFloatLE(i * 4));
            const codec = stage.lookupVariableByNameAndType('BS codec', '');
            if (!codec) throw new Error('project codec variable is missing');
            codec.value = loader.codec() === 'raw' ? 0 : 1;
        }
        done.value = 0;
        const mode = stage.lookupVariableByNameAndType('BS mode', '');
        if (process.argv[4] && !mode) throw new Error('project mode variable is missing');
        if (mode) mode.value = process.argv[3] === '--dot' ? 2 : Number(process.argv[4] || 0);
        await vm.greenFlag();
        await new Promise((resolve, reject) => {
            const deadline = Date.now() + 30000;
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
        if (!attempts.length || attempts.some(compiled => !compiled)) throw new Error('a Scratch thread did not compile');
        if (loader && (loader.status({ID: 'resident'}) !== 'ok' || loader.copied({ID: 'resident'}) !== (process.argv[3] === '--dot' ? 16 : 5) || loader.file !== null)) {
            throw new Error('compiled loader block did not complete before isolation');
        }
        process.stdout.write(JSON.stringify(output.value) + '\n');
    } finally {
        vm.quit();
    }
}

run().catch(error => { console.error(error); process.exitCode = 1; });
