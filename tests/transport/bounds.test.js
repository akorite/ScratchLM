const test = require('node:test');
const assert = require('node:assert/strict');
const {File} = require('node:buffer');
const fs = require('node:fs');
const vm = require('node:vm');

function setup() {
  let loader;
  const Scratch = {extensions: {unsandboxed: true, register: x => { loader = x; }}, BlockType: {}, ArgumentType: {}};
  vm.runInNewContext(fs.readFileSync('extension/loader.js', 'utf8'), {Scratch, TextDecoder});
  const lists = {q: {id: 'q', type: 'list', value: []}, s: {id: 's', type: 'list', value: []}};
  const target = {variables: lists, lookupVariableById(id) { return this.variables[id]; }};
  loader.runtime = {targets: [target]};
  return {loader, target, lists};
}
function file() {
  const count = Math.floor(4194304 / 34), data = new Uint8Array(count * 34);
  for (let i = 0; i < count; i++) { data[i * 34] = i & 255; data[i * 34 + 2] = 0x55; }
  const manifest = {schema: 'bonsaiscratch-package/v1', codec: 'raw', endianness: 'little', metadata: {}, payload_length: data.length, tensors: [{name: 'w', type_id: 142, dimensions: [128, count], offset: 0, length: data.length}]};
  const json = Buffer.from(JSON.stringify(manifest)), header = Buffer.alloc(16);
  header.write('BNSPKG1\0'); header.writeBigUInt64LE(BigInt(json.length), 8);
  return new File([header, json, data], 'large.bsp');
}
const args = (id, start, count) => ({ID: id, TENSOR: 'w', START: start, COUNT: count, Q: 'q', SCALE: 's'});

test('all nominal group budgets copy complete groups including the last', async () => {
  const {loader: e, target, lists} = setup(); await e.openFile(file());
  for (const budget of [65536, 262144, 1048576, 4194304]) {
    const count = Math.floor(budget / 34);
    await e.loadGroups(args(String(budget), 0, count), {target});
    assert.equal(e.status({ID: String(budget)}), 'ok'); assert.equal(e.copied({ID: String(budget)}), count);
    assert.equal(lists.q.value.length, count * 32); assert.equal(lists.s.value.length, count);
    assert.equal(lists.q.value[(count - 1) * 32], 0x55); assert.equal(lists.s.value[count - 1], (count - 1) & 255);
  }
  await e.loadGroups(args('over', 0, Math.floor(4194304 / 34) + 1), {target});
  assert.equal(e.status({ID: 'over'}), 'error'); assert.equal(lists.s.value.length, Math.floor(4194304 / 34));
});

test('new destination owner makes older differently named request stale', async () => {
  const {loader: e, target, lists} = setup(), f = file(); await e.openFile(f);
  const slice = f.slice.bind(f); let release;
  f.slice = (...args) => { const blob = slice(...args); return {arrayBuffer: () => new Promise(resolve => {release = async () => resolve(await blob.arrayBuffer());})}; };
  const old = e.loadGroups(args('old', 0, 1), {target}), oldRelease = release;
  const newer = e.loadGroups(args('new', 1, 1), {target}); await release(); await newer; await oldRelease(); await old;
  assert.equal(e.status({ID: 'old'}), 'stale'); assert.equal(e.status({ID: 'new'}), 'ok'); assert.deepEqual(Array.from(lists.s.value), [1]);
});
