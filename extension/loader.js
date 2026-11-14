(function (Scratch) {
  'use strict';
  if (!Scratch.extensions.unsandboxed) throw new Error('Bonsai loader requires an unsandboxed extension');
  const MAX_BYTES = 4 * 1024 * 1024;
  const MAX_MANIFEST = 64 * 1024 * 1024;
  const integer = n => Number.isSafeInteger(Number(n)) && Number(n) >= 0;
  const text = value => typeof value === 'string' ? value : JSON.stringify(value);
  const arg = (type, defaultValue) => ({type, defaultValue});

  class Loader {
    constructor() {
      this.runtime = Scratch.vm && Scratch.vm.runtime;
      this.file = null;
      this.manifest = null;
      this.window = null;
      this.prefetch = null;
      this.windowBytes = 64 * 1024 * 1024;
      this.generation = 0;
      this.requests = new Map();
      this.owners = new WeakMap();
      this.openStatus = 'no file';
    }
    getInfo() {
      const S = Scratch.ArgumentType.STRING, N = Scratch.ArgumentType.NUMBER;
      return {id: 'bonsailoader', name: 'Bonsai storage', blocks: [
        {opcode: 'selectFile', blockType: Scratch.BlockType.COMMAND, text: 'select Bonsai package'},
        {opcode: 'fileStatus', blockType: Scratch.BlockType.REPORTER, text: 'package status'},
        {opcode: 'codec', blockType: Scratch.BlockType.REPORTER, text: 'package codec'},
        {opcode: 'metadataCount', blockType: Scratch.BlockType.REPORTER, text: 'metadata count'},
        {opcode: 'metadataKey', blockType: Scratch.BlockType.REPORTER, text: 'metadata key [INDEX]', arguments: {INDEX: arg(N, 1)}},
        {opcode: 'metadataValue', blockType: Scratch.BlockType.REPORTER, text: 'metadata value [KEY]', arguments: {KEY: arg(S, '')}},
        {opcode: 'tensorCount', blockType: Scratch.BlockType.REPORTER, text: 'tensor count'},
        {opcode: 'tensorName', blockType: Scratch.BlockType.REPORTER, text: 'tensor name [INDEX]', arguments: {INDEX: arg(N, 1)}},
        {opcode: 'tensorInfo', blockType: Scratch.BlockType.REPORTER, text: 'tensor [TENSOR] field [FIELD]', arguments: {TENSOR: arg(S, ''), FIELD: arg(S, 'dimensions')}},
        {opcode: 'loadGroups', blockType: Scratch.BlockType.COMMAND, text: 'load groups [START] count [COUNT] of [TENSOR] into q list id [Q] scale list id [SCALE] request [ID]', arguments: {START: arg(N, 0), COUNT: arg(N, 1), TENSOR: arg(S, ''), Q: arg(S, ''), SCALE: arg(S, ''), ID: arg(S, '1')}},
        {opcode: 'loadDense', blockType: Scratch.BlockType.COMMAND, text: 'load floats [START] count [COUNT] of [TENSOR] into list id [DEST] request [ID]', arguments: {START: arg(N, 0), COUNT: arg(N, 1), TENSOR: arg(S, ''), DEST: arg(S, ''), ID: arg(S, '1')}},
        {opcode: 'cancel', blockType: Scratch.BlockType.COMMAND, text: 'cancel request [ID]', arguments: {ID: arg(S, '1')}},
        {opcode: 'status', blockType: Scratch.BlockType.REPORTER, text: 'request [ID] status', arguments: {ID: arg(S, '1')}},
        {opcode: 'copied', blockType: Scratch.BlockType.REPORTER, text: 'request [ID] copied', arguments: {ID: arg(S, '1')}},
        {opcode: 'error', blockType: Scratch.BlockType.REPORTER, text: 'request [ID] error', arguments: {ID: arg(S, '1')}}
      ]};
    }
    async selectFile() {
      const input = document.createElement('input');
      input.type = 'file'; input.accept = '.bsp';
      const file = await new Promise(resolve => {
        input.addEventListener('change', () => resolve(input.files[0] || null), {once: true});
        input.addEventListener('cancel', () => resolve(null), {once: true});
        input.click();
      });
      if (file) await this.openFile(file);
    }
    async read(file, offset, length) {
      const buffer = await file.slice(offset, offset + length).arrayBuffer();
      if (buffer.byteLength !== length) throw new Error('short read');
      return buffer;
    }
    async openFile(file) {
      const generation = ++this.generation;
      this.file = null; this.manifest = null; this.tensors = null; this.window = null; this.prefetch = null; this.openStatus = 'loading';
      for (const request of this.requests.values()) if (request.status === 'pending') request.status = 'stale';
      try {
        if (!integer(file.size) || file.size < 16) throw new Error('invalid package size');
        const head = new Uint8Array(await this.read(file, 0, 16));
        if (String.fromCharCode(...head.subarray(0, 8)) !== 'BNSPKG1\0') throw new Error('invalid package magic');
        const view = new DataView(head.buffer, head.byteOffset, head.byteLength);
        const length = view.getUint32(8, true) + view.getUint32(12, true) * 4294967296;
        if (!integer(length) || length > MAX_MANIFEST || length > file.size - 16) throw new Error('invalid manifest length');
        const manifest = JSON.parse(new TextDecoder('utf-8', {fatal: true}).decode(await this.read(file, 16, length)));
        const payloadStart = 16 + length;
        if (manifest.schema !== 'bonsaiscratch-package/v1' || manifest.endianness !== 'little' || !['raw', 'bslut4'].includes(manifest.codec) || !integer(manifest.payload_length) || manifest.payload_length !== file.size - payloadStart || !manifest.metadata || typeof manifest.metadata !== 'object' || Array.isArray(manifest.metadata) || !Array.isArray(manifest.tensors)) throw new Error('invalid manifest');
        const tensors = new Map();
        for (const tensor of manifest.tensors) {
          if (!tensor || typeof tensor.name !== 'string' || !tensor.name || tensors.has(tensor.name) || !integer(tensor.offset) || !integer(tensor.length) || tensor.offset > manifest.payload_length || tensor.length > manifest.payload_length - tensor.offset || !Array.isArray(tensor.dimensions) || !tensor.dimensions.length || tensor.dimensions.length > 4 || tensor.dimensions.some(d => !integer(d) || d === 0)) throw new Error('invalid tensor');
          const elements = tensor.dimensions.reduce((a, b) => a * b, 1);
          if (!Number.isSafeInteger(elements) || (tensor.type_id === 0 ? tensor.length !== elements * 4 : tensor.type_id === 142 ? tensor.dimensions[0] % 128 !== 0 || tensor.length !== elements / 128 * 34 : true)) throw new Error('invalid tensor shape or type');
          tensors.set(tensor.name, tensor);
        }
        if (generation !== this.generation) return;
        this.file = file; this.manifest = manifest; this.tensors = tensors; this.payloadStart = payloadStart; this.openStatus = 'ok';
        this.prefetch = {start: payloadStart, promise: this.readWindow(file, payloadStart, generation)};
        if (this.runtime && typeof this.runtime.setFramerate === 'function') this.runtime.setFramerate(250);
      } catch (error) {
        if (generation === this.generation) this.openStatus = `error: ${error.message}`;
      }
    }
    async readWindow(file, offset, generation) {
      const length = Math.max(0, Math.min(this.windowBytes, file.size - offset));
      const buffer = await this.read(file, offset, length);
      if (generation !== this.generation) return null;
      return {start: offset, end: offset + buffer.byteLength, buffer};
    }
    maybePrefetch(end) {
      const window = this.window;
      if (!this.file || !window || this.prefetch || end >= this.file.size) return;
      if (end <= window.end - Math.min(this.windowBytes >> 2, 8 * 1024 * 1024)) return;
      this.prefetch = {start: end, promise: this.readWindow(this.file, end, this.generation)};
    }
    publish(buffer, groups, count, lists) {
      const view = new DataView(buffer), bytes = new Uint8Array(buffer);
      if (groups) {
        const q = lists[0].value, scale = lists[1].value;
        q.length = count * 32; scale.length = count;
        for (let i = 0; i < count; i++) {
          scale[i] = view.getUint16(i * 34, true);
          for (let j = 0; j < 32; j++) q[i * 32 + j] = bytes[i * 34 + 2 + j];
        }
        lists[0]._monitorUpToDate = false; lists[1]._monitorUpToDate = false;
      } else {
        const values = lists[0].value;
        values.length = count;
        for (let i = 0; i < count; i++) values[i] = view.getFloat32(i * 4, true);
        lists[0]._monitorUpToDate = false;
      }
    }
    fileStatus() { return this.openStatus; }
    codec() { return this.manifest ? this.manifest.codec : ''; }
    metadataCount() { return this.manifest ? Object.keys(this.manifest.metadata).length : 0; }
    metadataKey(args) { return this.manifest && integer(args.INDEX) ? Object.keys(this.manifest.metadata)[Number(args.INDEX) - 1] || '' : ''; }
    metadataValue(args) { const entry = this.manifest && this.manifest.metadata[String(args.KEY)]; return entry && Object.prototype.hasOwnProperty.call(entry, 'value') ? text(entry.value) : ''; }
    tensorCount() { return this.manifest ? this.manifest.tensors.length : 0; }
    tensorName(args) { const tensor = this.manifest && integer(args.INDEX) && this.manifest.tensors[Number(args.INDEX) - 1]; return tensor ? tensor.name : ''; }
    tensorInfo(args) { const tensor = this.tensors && this.tensors.get(String(args.TENSOR)); const field = String(args.FIELD); return tensor && ['name', 'type_id', 'dimensions', 'source_offset', 'offset', 'length'].includes(field) && tensor[field] !== undefined ? text(tensor[field]) : ''; }
    status(args) { const r = this.requests.get(String(args.ID)); return r ? r.status : 'unknown'; }
    copied(args) { const r = this.requests.get(String(args.ID)); return r ? r.copied : 0; }
    error(args) { const r = this.requests.get(String(args.ID)); return r ? r.error : ''; }
    cancel(args) { const r = this.requests.get(String(args.ID)); if (r && r.status === 'pending') r.status = 'cancelled'; }
    loadGroups(args, util) { return this.load(args, util.target, true); }
    loadDense(args, util) { return this.load(args, util.target, false); }
    load(args, target, groups) {
      const id = String(args.ID), previous = this.requests.get(id);
      if (previous && previous.status === 'pending') previous.status = 'stale';
      const request = {status: 'pending', copied: 0, error: ''};
      this.requests.set(id, request);
      try {
        const file = this.file, generation = this.generation;
        if (!file || !this.manifest) throw new Error('no package loaded');
        const tensor = this.tensors.get(String(args.TENSOR));
        if (!tensor || tensor.type_id !== (groups ? 142 : 0)) throw new Error('unknown tensor or wrong type');
        const start = Number(args.START), count = Number(args.COUNT), unit = groups ? 34 : 4;
        if (!integer(start) || !integer(count) || count > Math.floor(MAX_BYTES / unit) || start > tensor.length / unit || count > tensor.length / unit - start) throw new Error('invalid range');
        const ids = groups ? [String(args.Q), String(args.SCALE)] : [String(args.DEST)];
        const lists = ids.map(id => target && target.lookupVariableById(id));
        if (!target || !this.runtime || !this.runtime.targets.includes(target) || lists.some(list => !list || list.type !== 'list' || !Array.isArray(list.value)) || new Set(lists).size !== lists.length) throw new Error('invalid destination list');
        for (const list of lists) {
          const owner = this.owners.get(list);
          if (owner && owner.status === 'pending') owner.status = 'stale';
          this.owners.set(list, request);
        }
        const abs = this.payloadStart + tensor.offset + start * unit, length = count * unit;
        const window = this.window;
        if (window && abs >= window.start && abs + length <= window.end) {
          this.publish(window.buffer.slice(abs - window.start, abs - window.start + length), groups, count, lists);
          request.copied = count; request.status = 'ok';
          this.maybePrefetch(abs + length);
          return undefined;
        }
        return this.refill(request, id, generation, file, abs, length, groups, count, target, ids, lists);
      } catch (error) {
        if (request.status === 'pending') { request.status = 'error'; request.error = String(error.message || error); }
        return undefined;
      }
    }
    async refill(request, id, generation, file, abs, length, groups, count, target, ids, lists) {
      try {
        let buffer;
        const prefetch = this.prefetch;
        if (prefetch && abs >= prefetch.start && abs + length <= prefetch.start + this.windowBytes) {
          this.prefetch = null;
          const window = await prefetch.promise;
          if (window) { this.window = window; this.maybePrefetch(window.end); }
          if (request.status !== 'pending' || this.requests.get(id) !== request || generation !== this.generation) return;
          if (this.window && abs >= this.window.start && abs + length <= this.window.end) {
            buffer = this.window.buffer.slice(abs - this.window.start, abs - this.window.start + length);
          } else {
            buffer = await this.read(file, abs, length);
          }
        } else if (length > this.windowBytes) {
          buffer = await this.read(file, abs, length);
        } else {
          const window = await this.readWindow(file, abs, generation);
          if (window && window.start === abs) { this.window = window; this.prefetch = null; this.maybePrefetch(window.end); }
          if (request.status !== 'pending' || this.requests.get(id) !== request || generation !== this.generation) return;
          if (this.window && abs >= this.window.start && abs + length <= this.window.end) {
            buffer = this.window.buffer.slice(abs - this.window.start, abs - this.window.start + length);
          } else {
            buffer = await this.read(file, abs, length);
          }
        }
        if (request.status !== 'pending' || this.requests.get(id) !== request || generation !== this.generation) return;
        if (!this.runtime.targets.includes(target) || lists.some((list, i) => target.lookupVariableById(ids[i]) !== list || list.type !== 'list' || !Array.isArray(list.value) || this.owners.get(list) !== request)) throw new Error('destination ownership changed');
        this.publish(buffer, groups, count, lists);
        request.copied = count; request.status = 'ok';
      } catch (error) {
        if (request.status === 'pending') { request.status = 'error'; request.error = String(error.message || error); }
      }
    }
  }
  Scratch.extensions.register(new Loader());
})(Scratch);
