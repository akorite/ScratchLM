'use strict';

const fs = require('fs');
const path = require('path');
const root = process.env.BONSAI_TURBOWARP_VM;
if (!root) throw new Error('BONSAI_TURBOWARP_VM must name the pinned VM checkout');
const VirtualMachine = require(path.join(root, 'src/index.js'));
const Thread = require(path.join(root, 'src/engine/thread.js'));
const ScratchStorage = require('scratch-storage');
const attempts = [];
const compile = Thread.prototype.tryCompile;
Thread.prototype.tryCompile = function () { compile.call(this); attempts.push(Boolean(this.isCompiled)); };

async function main() {
    const vm = new VirtualMachine();
    let error;
    vm.on('COMPILE_ERROR', (target, value) => { error = value; });
    vm.attachStorage(new ScratchStorage());
    vm.start();
    vm.setCompatibilityMode(false);
    vm.setTurboMode(true);
    vm.setCompilerOptions({enabled: true});
    try {
        await vm.loadProject(fs.readFileSync(process.argv[2]));
        const stage = vm.runtime.getTargetForStage();
        const variable = name => stage.lookupVariableByNameAndType(`BS ${name}`, '');
        const input = stage.lookupVariableByNameAndType('BS input', 'list');
        const output = stage.lookupVariableByNameAndType('BS output', 'list');
        const results = [];
        for (const test of JSON.parse(fs.readFileSync(process.argv[3], 'utf8'))) {
            input.value = test.input;
            variable('mode').value = test.mode;
            variable('done').value = 0;
            await vm.greenFlag();
            await new Promise((resolve, reject) => {
                const deadline = Date.now() + 60000;
                const timer = setInterval(() => {
                    if (error || Date.now() > deadline) { clearInterval(timer); reject(error || new Error('text project timed out')); }
                    else if (Number(variable('done').value) === 1) { clearInterval(timer); resolve(); }
                }, 5);
            });
            results.push({output: output.value.slice(), backdrop: stage.currentCostume + 1});
        }
        if (!attempts.length || attempts.some(value => !value)) throw new Error('a Scratch thread did not compile');
        process.stdout.write(JSON.stringify(results) + '\n');
    } finally {
        vm.quit();
    }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
