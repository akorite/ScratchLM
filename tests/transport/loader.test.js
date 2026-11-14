const test = require('node:test');
const assert = require('node:assert/strict');
const {File} = require('node:buffer');
const fs = require('node:fs');
const vm = require('node:vm');

function setup() {
  let extension;
  const Scratch = {extensions: {unsandboxed: true, register: x => { extension = x; }}, BlockType: {COMMAND: 'command', REPORTER: 'reporter'}, ArgumentType: {STRING: 'string', NUMBER: 'number'}};
  vm.runInNewContext(fs.readFileSync('extension/loader.js', 'utf8'), {Scratch, TextDecoder, Uint8Array, DataView, Number, JSON, Map, Object, Array, String, Error, Math, console});
  const variables = Object.fromEntries(['q', 's', 'f'].map(id => [id, {id, name: id, type: 'list', value: [999]}]));
  const target = {variables, lookupVariableById(id) { return this.variables[id]; }};
  extension.runtime = {targets: [target]};
  return {extension, target, variables};
}
function packageFile(codec = 'raw') {
  const group = (scale, byte) => Uint8Array.from([scale & 255, scale >> 8, ...Array(32).fill(byte)]);
  const dense = new Uint8Array(12); const view = new DataView(dense.buffer);
  [1.5, -2.25, 0].forEach((x, i) => view.setFloat32(i * 4, x, true));
  const payload = new Blob([group(0x8001, 0x24), group(0x7c00, 0x50), dense]);
  const manifest = {schema: 'bonsaiscratch-package/v1', codec, endianness: 'little', payload_length: payload.size,
    metadata: {foo: {type_id: 8, value: 'bar'}}, tensors: [
      {name: 'weights', type_id: 142, dimensions: [256, 1], offset: 0, length: 68},
      {name: 'dense', type_id: 0, dimensions: [3], offset: 68, length: 12}]};
  const json = Buffer.from(JSON.stringify(manifest)); const header = Buffer.alloc(16);
  header.write('BNSPKG1\0'); header.writeBigUInt64LE(BigInt(json.length), 8);
  return new File([header, json, payload], 'model.bsp');
}
const groups = (id, start, count, q = 'q', s = 's') => ({ID: id, TENSOR: 'weights', START: start, COUNT: count, Q: q, SCALE: s});
const dense = (id, start, count) => ({ID: id, TENSOR: 'dense', START: start, COUNT: count, DEST: 'f'});

test('indexed metadata and group ranges preserve storage and reuse list variables', async () => {
  for (const codec of ['raw', 'bslut4']) {
    const {extension: e, target, variables: v} = setup(); await e.openFile(packageFile(codec));
    assert.equal(e.metadataCount(), 1); assert.equal(e.metadataKey({INDEX: 1}), 'foo');
    assert.equal(e.metadataValue({KEY: 'foo'}), 'bar'); assert.equal(e.tensorName({INDEX: 2}), 'dense');
    assert.equal(e.tensorInfo({TENSOR: 'weights', FIELD: 'dimensions'}), '[256,1]');
    const q = v.q, s = v.s, qBuffer = q.value, scaleBuffer = s.value;
    await e.loadGroups(groups('first', 0, 1), {target});
    assert.equal(e.status({ID: 'first'}), 'ok'); assert.equal(e.copied({ID: 'first'}), 1);
    assert.equal(q.value, qBuffer); assert.equal(s.value, scaleBuffer);
    assert.equal(v.q, q); assert.equal(v.s, s); assert.deepEqual(Array.from(q.value), Array(32).fill(0x24)); assert.deepEqual(Array.from(s.value), [0x8001]);
    await e.loadGroups(groups('last', 1, 1), {target});
    assert.deepEqual(Array.from(q.value), Array(32).fill(0x50)); assert.deepEqual(Array.from(s.value), [0x7c00]);
    await e.loadGroups(groups('eof', 2, 0), {target}); assert.equal(e.status({ID: 'eof'}), 'ok'); assert.deepEqual(Array.from(q.value), []);
  }
});

test('dense ranges and invalid requests do not corrupt destinations', async () => {
  const {extension: e, target, variables: v} = setup(); await e.openFile(packageFile());
  await e.loadDense(dense('a', 1, 2), {target}); assert.deepEqual(Array.from(v.f.value), [-2.25, 0]);
  for (const [id, start, count] of [['negative', -1, 1], ['past', 3, 1], ['fraction', 0.5, 1], ['huge', 0, 1048577]]) {
    await e.loadDense(dense(id, start, count), {target}); assert.equal(e.status({ID: id}), 'error'); assert.deepEqual(Array.from(v.f.value), [-2.25, 0]);
  }
  await e.loadGroups(groups('alias', 0, 1, 'q', 'q'), {target}); assert.equal(e.status({ID: 'alias'}), 'error');
});

test('short reads, cancellation, stale IDs and changed ownership cannot publish', async () => {
  const {extension: e, target, variables: v} = setup(); const file = packageFile();
  e.windowBytes = 0; await e.openFile(file);
  const originalSlice = file.slice.bind(file); let release;
  file.slice = (...args) => { const blob = originalSlice(...args); return {arrayBuffer: () => new Promise(resolve => { release = async () => resolve(await blob.arrayBuffer()); })}; };
  let pending = e.loadGroups(groups('cancel', 0, 1), {target}); e.cancel({ID: 'cancel'}); await release(); await pending;
  assert.equal(e.status({ID: 'cancel'}), 'cancelled'); assert.deepEqual(v.q.value, [999]);
  pending = e.loadGroups(groups('same', 0, 1), {target}); const firstRelease = release;
  const newer = e.loadGroups(groups('same', 1, 1), {target}); await release(); await newer; await firstRelease(); await pending;
  assert.deepEqual(Array.from(v.s.value), [0x7c00]); assert.equal(e.status({ID: 'same'}), 'ok');
  pending = e.loadGroups(groups('owner', 0, 1), {target}); const old = v.q; target.variables.q = {id: 'q', type: 'list', value: []}; await release(); await pending;
  assert.equal(e.status({ID: 'owner'}), 'error'); assert.equal(target.variables.q.value.length, 0); assert.deepEqual(Array.from(old.value), Array(32).fill(0x50));
  file.slice = (...args) => ({arrayBuffer: async () => (await originalSlice(...args).arrayBuffer()).slice(0, 1)});
  await e.loadGroups(groups('short', 0, 1), {target}); assert.equal(e.status({ID: 'short'}), 'error');
});

test('window hits return non-promise and misses refill then publish', async () => {
  const {extension: e, target, variables: v} = setup(); const file = packageFile();
  e.windowBytes = 68; await e.openFile(file);
  assert.equal(e.fileStatus(), 'ok');
  const first = e.loadGroups(groups('miss', 0, 1), {target});
  assert.equal(first && typeof first.then, 'function'); assert.equal(e.status({ID: 'miss'}), 'pending');
  await first;
  assert.equal(e.status({ID: 'miss'}), 'ok'); assert.equal(e.copied({ID: 'miss'}), 1);
  assert.deepEqual(Array.from(v.q.value), Array(32).fill(0x24));
  const hit = e.loadGroups(groups('hit', 0, 1), {target});
  assert.equal(hit, undefined); assert.equal(e.status({ID: 'hit'}), 'ok');
  const second = e.loadGroups(groups('hit2', 1, 1), {target});
  assert.equal(second, undefined); assert.deepEqual(Array.from(v.s.value), [0x7c00]);
  await e.loadDense(dense('d', 0, 3), {target});
  assert.deepEqual(Array.from(v.f.value), [1.5, -2.25, 0]);
});

test('requests larger than the window read through and refill adopts prefetched windows', async () => {
  const {extension: e, target, variables: v} = setup(); const file = packageFile();
  e.windowBytes = 10; await e.openFile(file);
  await e.loadGroups(groups('over', 0, 1), {target});
  assert.equal(e.status({ID: 'over'}), 'ok'); assert.deepEqual(Array.from(v.q.value), Array(32).fill(0x24));
  e.windowBytes = 64 * 1024 * 1024;
  const {extension: e2, target: t2, variables: v2} = setup();
  e2.windowBytes = 40; await e2.openFile(packageFile());
  await e2.loadGroups(groups('a', 0, 1), {target: t2});
  e2.windowBytes = 0;
  e2.window = null;
  e2.windowBytes = 40;
  const miss = e2.loadGroups(groups('b', 1, 1), {target: t2});
  await miss;
  assert.equal(e2.status({ID: 'b'}), 'ok'); assert.deepEqual(Array.from(v2.s.value), [0x7c00]);
});
