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
function packageFile(codec = 'raw', metadata = {foo: {type_id: 8, value: 'bar'}}) {
  const group = (scale, byte) => Uint8Array.from([scale & 255, scale >> 8, ...Array(32).fill(byte)]);
  const dense = new Uint8Array(12); const view = new DataView(dense.buffer);
  [1.5, -2.25, 0].forEach((x, i) => view.setFloat32(i * 4, x, true));
  const payload = new Blob([group(0x8001, 0x24), group(0x7c00, 0x50), dense]);
  const manifest = {schema: 'bonsaiscratch-package/v1', codec, endianness: 'little', payload_length: payload.size,
    metadata, tensors: [
      {name: 'weights', type_id: 142, dimensions: [256, 1], offset: 0, length: 68},
      {name: 'dense', type_id: 0, dimensions: [3], offset: 68, length: 12}]};
  const json = Buffer.from(JSON.stringify(manifest)); const header = Buffer.alloc(16);
  header.write('BNSPKG1\0'); header.writeBigUInt64LE(BigInt(json.length), 8);
  return new File([header, json, payload], 'model.bsp');
}
const groups = (id, start, count, q = 'q', s = 's') => ({ID: id, TENSOR: 'weights', START: start, COUNT: count, Q: q, SCALE: s});
const dense = (id, start, count) => ({ID: id, TENSOR: 'dense', START: start, COUNT: count, DEST: 'f'});

test('metadata string arrays copy exactly and respect cancellation and ownership', async () => {
  const {extension: e, target, variables: v} = setup();
  const tokens = Array.from({length: 248320}, (_, i) => `token-${i}`);
  await e.openFile(packageFile('raw', {tokens: {type_id: 9, value: tokens}, wrong: {type_id: 9, value: ['ok', 3]}}));
  const list = v.f, buffer = list.value;
  const args = id => ({ID: id, KEY: 'tokens', DEST: 'f'});
  await e.loadStrings(args('copy'), {target});
  assert.equal(e.status({ID: 'copy'}), 'ok');
  assert.equal(e.copied({ID: 'copy'}), tokens.length);
  assert.equal(v.f, list); assert.equal(list.value, buffer);
  assert.deepEqual(Array.from(list.value), tokens);
  list.value[0] = 'sentinel';
  const pending = e.loadStrings(args('cancel'), {target}); e.cancel({ID: 'cancel'}); await pending;
  assert.equal(e.status({ID: 'cancel'}), 'cancelled');
  assert.equal(list.value[0], 'sentinel');
  for (const key of ['missing', 'wrong']) {
    await e.loadStrings({ID: key, KEY: key, DEST: 'f'}, {target});
    assert.equal(e.status({ID: key}), 'error');
    assert.equal(list.value[0], 'sentinel');
  }
  const old = e.loadStrings(args('old'), {target});
  const newer = e.loadStrings(args('new'), {target});
  await Promise.all([old, newer]);
  assert.equal(e.status({ID: 'old'}), 'stale');
  assert.equal(e.status({ID: 'new'}), 'ok');
});

test('indexed metadata and group ranges preserve storage and reuse list variables', async () => {
  for (const codec of ['raw', 'bslut4', 'bslut5']) {
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
test('unknown codec fails file selection loudly', async () => {
  const {extension: e} = setup();
  await e.openFile(packageFile('bslut6'));
  assert.ok(e.fileStatus().startsWith('error:'));
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
  assert.equal(e.window, null);
  const {extension: e2, target: t2, variables: v2} = setup(); const file2 = packageFile();
  const slice = file2.slice.bind(file2), reads = [];
  file2.slice = (start, end) => { reads.push([start, end]); return slice(start, end); };
  e2.windowBytes = 40; await e2.openFile(file2);
  await e2.loadGroups(groups('a', 0, 1), {target: t2});
  const nextStart = file2.size - 80 + 34;
  assert.equal(reads.filter(([start]) => start === nextStart).length, 1);
  await e2.loadGroups(groups('b', 1, 1), {target: t2});
  assert.equal(reads.filter(([start]) => start === nextStart).length, 1);
  assert.equal(e2.status({ID: 'b'}), 'ok'); assert.deepEqual(Array.from(v2.s.value), [0x7c00]);
});

test('matching prefetch failures are reported, even if they have already settled', async () => {
  const {extension: e, target, variables} = setup(); const file = packageFile();
  const slice = file.slice.bind(file), payloadStart = file.size - 80;
  let failed = false;
  file.slice = (start, end) => {
    if (start === payloadStart && !failed) {
      failed = true;
      return {arrayBuffer: () => Promise.reject(new Error('prefetch failed'))};
    }
    return slice(start, end);
  };
  e.windowBytes = 40; await e.openFile(file);
  await new Promise(resolve => setImmediate(resolve));
  await e.loadGroups(groups('failed', 0, 1), {target});
  assert.equal(e.status({ID: 'failed'}), 'error');
  assert.equal(e.error({ID: 'failed'}), 'prefetch failed');
  assert.deepEqual(variables.q.value, [999]);
  await e.loadGroups(groups('retry', 0, 1), {target});
  assert.equal(e.status({ID: 'retry'}), 'ok');
  assert.deepEqual(Array.from(variables.s.value), [0x8001]);
});

test('a sequential prefetch failure reaches the matching request', async () => {
  const {extension: e, target, variables} = setup(); const file = packageFile();
  const slice = file.slice.bind(file), nextStart = file.size - 80 + 34;
  let rejectNext;
  file.slice = (start, end) => start === nextStart
    ? {arrayBuffer: () => new Promise((resolve, reject) => { rejectNext = reject; })}
    : slice(start, end);
  e.windowBytes = 40; await e.openFile(file);
  await e.loadGroups(groups('first', 0, 1), {target});
  const next = e.loadGroups(groups('next', 1, 1), {target});
  rejectNext(new Error('sequential prefetch failed'));
  await next;
  assert.equal(e.status({ID: 'next'}), 'error');
  assert.equal(e.error({ID: 'next'}), 'sequential prefetch failed');
  assert.deepEqual(Array.from(variables.s.value), [0x8001]);
});

test('discarded initial and sequential prefetch failures do not escape', async () => {
  const {extension: e, target} = setup(); const file = packageFile();
  const slice = file.slice.bind(file), payloadStart = file.size - 80;
  const rejects = [];
  file.slice = (start, end) => {
    if (start === payloadStart || start === payloadStart + 34) {
      return {arrayBuffer: () => new Promise((resolve, reject) => { rejects.push(reject); })};
    }
    return slice(start, end);
  };
  e.windowBytes = 40; await e.openFile(file);
  await e.openFile(packageFile());
  rejects.shift()(new Error('discarded initial'));
  await new Promise(resolve => setImmediate(resolve));

  const file2 = packageFile(), slice2 = file2.slice.bind(file2), start2 = file2.size - 80;
  let rejectTail;
  file2.slice = (start, end) => start === start2 + 34
    ? {arrayBuffer: () => new Promise((resolve, reject) => { rejectTail = reject; })}
    : slice2(start, end);
  await e.openFile(file2);
  await e.loadGroups(groups('first', 0, 1), {target});
  assert.equal(typeof rejectTail, 'function');
  await e.openFile(packageFile());
  rejectTail(new Error('discarded sequential'));
  await new Promise(resolve => setImmediate(resolve));
});

test('cancelled concurrent readers share an in-flight prefetch without corrupting the survivor', async () => {
  const {extension: e, target, variables} = setup(); const file = packageFile();
  target.variables.q2 = {id: 'q2', type: 'list', value: [999]};
  target.variables.s2 = {id: 's2', type: 'list', value: [999]};
  const slice = file.slice.bind(file), payloadStart = file.size - 80;
  let release, payloadReads = 0;
  file.slice = (start, end) => {
    const blob = slice(start, end);
    if (start !== payloadStart) return blob;
    payloadReads++;
    return {arrayBuffer: () => new Promise(resolve => { release = async () => resolve(await blob.arrayBuffer()); })};
  };
  e.windowBytes = 40; await e.openFile(file);
  const cancelled = e.loadGroups(groups('cancelled', 0, 1), {target});
  const survivor = e.loadGroups(groups('survivor', 0, 1, 'q2', 's2'), {target});
  e.cancel({ID: 'cancelled'});
  assert.equal(payloadReads, 1);
  await release(); await Promise.all([cancelled, survivor]);
  assert.equal(e.status({ID: 'cancelled'}), 'cancelled');
  assert.deepEqual(variables.q.value, [999]);
  assert.equal(e.status({ID: 'survivor'}), 'ok');
  assert.deepEqual(target.variables.s2.value, [0x8001]);
});

test('an older refill does not discard a newer prefetch', async () => {
  const {extension: e, target, variables} = setup(); const file = packageFile();
  const slice = file.slice.bind(file), payloadStart = file.size - 80;
  let release, nextReads = 0;
  file.slice = (start, end) => {
    const blob = slice(start, end);
    if (start === payloadStart + 34) nextReads++;
    return start === payloadStart + 68
      ? {arrayBuffer: () => new Promise(resolve => { release = async () => resolve(await blob.arrayBuffer()); })}
      : blob;
  };
  e.windowBytes = 40; await e.openFile(file);
  const older = e.loadDense(dense('older', 0, 1), {target});
  await e.loadGroups(groups('first', 0, 1), {target});
  assert.equal(nextReads, 1);
  await release(); await older;
  await e.loadGroups(groups('next', 1, 1), {target});
  assert.equal(e.status({ID: 'next'}), 'ok');
  assert.deepEqual(Array.from(variables.s.value), [0x7c00]);
  assert.equal(nextReads, 1);
});

test('a cancelled refill cannot replace a newer window', async () => {
  const {extension: e, target, variables} = setup(); const file = packageFile();
  const slice = file.slice.bind(file), payloadStart = file.size - 80;
  let release;
  file.slice = (start, end) => {
    const blob = slice(start, end);
    return start === payloadStart + 34
      ? {arrayBuffer: () => new Promise(resolve => { release = async () => resolve(await blob.arrayBuffer()); })}
      : blob;
  };
  e.windowBytes = 40; await e.openFile(file);
  const old = e.loadGroups(groups('cancel', 1, 1), {target});
  e.cancel({ID: 'cancel'});
  await e.loadDense(dense('new', 0, 3), {target});
  await release(); await old;
  assert.equal(e.status({ID: 'cancel'}), 'cancelled');
  assert.deepEqual(variables.q.value, [999]);
  assert.equal(e.loadDense(dense('hit', 0, 3), {target}), undefined);
  assert.equal(e.status({ID: 'hit'}), 'ok');
  assert.deepEqual(Array.from(variables.f.value), [1.5, -2.25, 0]);
});

test('a stale refill cannot replace the current file window', async () => {
  const {extension: e, target, variables} = setup(); const oldFile = packageFile();
  const slice = oldFile.slice.bind(oldFile), payloadStart = oldFile.size - 80;
  let release;
  oldFile.slice = (start, end) => {
    const blob = slice(start, end);
    return start === payloadStart
      ? {arrayBuffer: () => new Promise(resolve => { release = async () => resolve(await blob.arrayBuffer()); })}
      : blob;
  };
  e.windowBytes = 40; await e.openFile(oldFile);
  const old = e.loadGroups(groups('old', 0, 1), {target});
  await e.openFile(packageFile());
  await e.loadDense(dense('new', 0, 3), {target});
  await release(); await old;
  assert.equal(e.status({ID: 'old'}), 'stale');
  assert.equal(e.loadDense(dense('hit', 0, 3), {target}), undefined);
  assert.equal(e.status({ID: 'hit'}), 'ok');
  assert.deepEqual(Array.from(variables.f.value), [1.5, -2.25, 0]);
});

test('a window that already reaches EOF does not prefetch its own tail', async () => {
  const {extension: e, target} = setup(); const file = packageFile();
  const slice = file.slice.bind(file), reads = [];
  file.slice = (start, end) => { reads.push([start, end]); return slice(start, end); };
  e.windowBytes = 128; await e.openFile(file);
  await e.loadDense(dense('first', 0, 1), {target});
  assert.equal(e.loadDense(dense('second', 1, 1), {target}), undefined);
  assert.equal(e.status({ID: 'second'}), 'ok');
  assert.equal(reads.filter(([start]) => start >= file.size - 80).length, 1);
});
