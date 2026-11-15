(function (Scratch) {
  'use strict';
  if (!Scratch.extensions.unsandboxed) throw new Error('Bonsai loader requires an unsandboxed extension');
  const MAX_BYTES = 4 * 1024 * 1024;
  const MAX_MANIFEST = 64 * 1024 * 1024;
  const integer = n => Number.isSafeInteger(Number(n)) && Number(n) >= 0;
  const text = value => typeof value === 'string' ? value : JSON.stringify(value);
  const arg = (type, defaultValue) => ({type, defaultValue});

  // This view only presents VM state and dispatches user input; inference stays in blocks.
  class ChatView {
    constructor(loader) {
      this.loader = loader;
      this.vm = Scratch.vm;
      this.host = document.createElement('div');
      this.host.id = 'bonsai-chat';
      const root = this.host.attachShadow({mode: 'open'});
      root.innerHTML = `
        <style>
          :host{font:14px/1.5 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#ececec}
          *{box-sizing:border-box}button,textarea{font:inherit}button{cursor:pointer}
          button:focus-visible,textarea:focus-visible{outline:2px solid #77d7b5;outline-offset:3px}
          button:disabled{opacity:.45;cursor:not-allowed}button{border:1px solid #424242;background:#303030;color:inherit;border-radius:9px;padding:8px 12px}
          [hidden]{display:none!important}
          .launcher{position:fixed;bottom:18px;right:18px;z-index:1000;background:#222;color:#eee;box-shadow:0 4px 24px #0004}
          dialog{padding:0;border:1px solid #424242;border-radius:16px;background:#212121;color:inherit;width:min(940px,calc(100vw - 32px));height:calc(100dvh - 32px);max-height:900px}
          dialog::backdrop{background:#0009}.shell{display:flex;flex-direction:column;height:100%}
          header{display:flex;align-items:center;justify-content:space-between;padding:16px 24px;border-bottom:1px solid #343434;gap:12px}
          .brand{font-size:18px;font-weight:650;letter-spacing:-.5px}.tag{font-size:11px;color:#aaa;margin-left:10px;letter-spacing:.04em}
          .actions{display:flex;gap:8px;align-items:center}.close{border:0;background:none;font-size:22px;padding:0 8px}
          .feed{flex:1;overflow:auto;padding:28px max(24px,calc((100% - 680px)/2));scroll-behavior:auto}
          .welcome{margin:clamp(16px,8vh,80px) 0 40px}.welcome h1{font-size:clamp(24px,4vw,34px);letter-spacing:-1px;line-height:1.2;margin:0 0 14px}
          .muted{color:#aaa}.welcome p{max-width:460px}.message{margin:0 0 28px;white-space:pre-wrap;overflow-wrap:anywhere}
          .message strong{display:block;font-size:12px;color:#aaa;margin-bottom:8px}.message.user{background:#303030;padding:14px 18px;border-radius:18px;max-width:90%;margin-left:auto;width:fit-content}
          .message.user strong{color:#bbb}.reply{min-height:24px}.reply:empty::after{content:"Working locally…";color:#999}
          footer{padding:0 24px 16px;width:min(100%,744px);align-self:center}
          .status{padding:12px 4px;min-height:74px}.statusline{display:flex;justify-content:space-between;gap:12px;font-size:12px;color:#bdbdbd}
          .phase{font-weight:600;color:#e0e0e0}.detail{font-size:12px;color:#999;margin-top:5px;overflow-wrap:anywhere}
          progress{display:block;width:100%;height:4px;margin-top:8px;accent-color:#77d7b5;border:0}
          .composer{background:#303030;border:1px solid #484848;border-radius:20px;padding:14px}
          textarea{display:block;resize:vertical;width:100%;min-height:60px;max-height:160px;border:0;background:none;color:#eee;padding:0 2px}
          textarea::placeholder{color:#999}.composer-actions{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-top:10px}
          .hint{font-size:11px;color:#aaa}.send{background:#ececec;color:#171717;border:0;font-weight:600;min-width:76px}.stop{background:#38302d}
          .note{font-size:11px;text-align:center;color:#929292;margin:9px 0 0}
          @media(max-width:560px){header{padding:12px 16px}.tag{display:none}.feed{padding:20px 18px}footer{padding:0 12px 12px}.hint{display:none}.welcome{margin-top:24px}}
        </style>
        <button class="launcher" type="button">Open Bonsai chat</button>
        <dialog aria-label="Bonsai local chat">
          <div class="shell">
            <header><div><span class="brand">Bonsai</span><span class="tag">LOCAL · SCRATCH INFERENCE</span></div>
              <div class="actions"><button type="button" class="package">Select model</button><button type="button" class="close" aria-label="Close chat">×</button></div></header>
            <main class="feed" aria-label="Conversation">
              <section class="welcome"><h1>What would you like to explore?</h1>
                <p class="muted">A small model, running right here. Select your Bonsai package, then send a message.</p>
                <p class="muted">Inference is slow. The status below shows actual token and layer progress—not a countdown.</p></section>
              <div class="messages"></div>
            </main>
            <footer>
              <div class="status"><div class="statusline"><span class="phase" role="status" aria-live="polite">Select a model to begin</span><span class="counter"></span></div>
                <progress aria-label="Current phase progress" hidden></progress><div class="detail"></div></div>
              <form class="composer"><textarea aria-label="Message Bonsai" placeholder="Message Bonsai" rows="2" required></textarea>
                <div class="composer-actions"><span class="hint">Enter to send · Shift+Enter for a new line</span><div><button type="button" class="stop" hidden>Stop</button> <button type="submit" class="send" disabled>Send</button></div></div>
              </form>
              <p class="note">Messages stay in this tab. Each send starts a fresh model context; earlier messages are display-only.</p>
            </footer>
          </div>
        </dialog>`;
      this.el = selector => root.querySelector(selector);
      document.body.appendChild(this.host);
      this.el('.launcher').onclick = () => this.show();
      this.el('.close').onclick = () => this.el('dialog').close();
      this.el('.package').onclick = async () => {
        try { await loader.selectFile(); } catch (error) { loader.openStatus = `error: ${error.message}`; }
        this.refresh();
      };
      this.el('form').onsubmit = event => { event.preventDefault(); this.send(); };
      this.el('textarea').onkeydown = event => {
        if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
          event.preventDefault();
          if (!this.el('.send').disabled) this.el('form').requestSubmit();
        }
      };
      this.el('.stop').onclick = () => {
        this.set('BS stop', 1);
        this.set('BS stop reason', 'user');
        this.vm.stopAll();
        this.finish('Stopped');
      };
      this.onStop = () => { if (this.running) this.finish('Stopped'); };
      loader.runtime.on('PROJECT_STOP_ALL', this.onStop);
      this.timer = setInterval(() => this.refresh(), 250);
      this.refresh();
      this.show();
    }
    variable(name) {
      const stage = this.loader.runtime.getTargetForStage();
      return stage && Object.values(stage.variables).find(v => v.name === name);
    }
    get(name, fallback = 0) { const v = this.variable(name); return v ? v.value : fallback; }
    set(name, value) { const v = this.variable(name); if (v) v.value = value; }
    show() {
      this.set('BS ask', 0);
      if (!this.el('dialog').open) this.el('dialog').showModal();
      this.el('textarea').focus();
    }
    message(role, text) {
      const article = document.createElement('article');
      article.className = `message ${role === 'You' ? 'user' : 'assistant'}`;
      const label = document.createElement('strong');
      label.textContent = role;
      const body = document.createElement('div');
      body.textContent = text;
      article.append(label, body);
      this.el('.messages').appendChild(article);
      return body;
    }
    send() {
      const input = this.el('textarea');
      const prompt = input.value.trim();
      if (!prompt || this.running || this.loader.openStatus !== 'ok') return;
      // Stop any old native ask/green-flag script before replacing its prompt.
      this.vm.stopAll();
      this.el('.welcome').hidden = true;
      this.message('You', prompt);
      this.reply = this.message('Bonsai', '');
      this.reply.className = 'reply';
      for (const [name, value] of Object.entries({
        'BS prompt': prompt, 'BS text': '', 'BS ask': 0, 'BS mode': 24, 'BS loop': 1,
        'BS started': 0, 'BS stop': 0, 'BS done': 0, 'BS step': 0,
        'BS prompt length': 0, 'BS max tokens': 128, 'BS ui phase': 'tokenizing'
      })) this.set(name, value);
      input.value = '';
      this.vm.greenFlag();
      this.running = true;
      this.began = performance.now();
      this.refresh();
      this.el('.feed').scrollTop = this.el('.feed').scrollHeight;
    }
    finish(label) {
      this.running = false;
      this.el('.phase').textContent = label;
      this.el('progress').hidden = true;
      this.el('.stop').hidden = true;
      this.el('.send').disabled = this.loader.openStatus !== 'ok';
      this.el('.package').disabled = false;
      this.el('textarea').disabled = false;
      if (this.reply && !this.reply.textContent) this.reply.textContent = 'No response generated.';
    }
    refresh() {
      const available = this.loader.openStatus === 'ok';
      this.el('.package').disabled = !!this.running;
      this.el('.send').disabled = !available || !!this.running;
      this.el('.stop').hidden = !this.running;
      this.el('textarea').disabled = !!this.running;
      if (!this.running) {
        if (!this.reply || !available) {
          this.el('.phase').textContent = available ? 'Ready' :
            this.loader.openStatus === 'loading' ? 'Opening model package…' :
            this.loader.openStatus.startsWith('error:') ? this.loader.openStatus : 'Select a model to begin';
          this.el('.detail').textContent = available ? this.loader.file.name : 'Your .bsp file stays on your device.';
        }
        return;
      }
      const feed = this.el('.feed');
      const atBottom = feed.scrollHeight - feed.scrollTop - feed.clientHeight < 60;
      const response = String(this.get('BS text', ''));
      if (this.reply.textContent !== response) this.reply.textContent = response;
      if (atBottom) feed.scrollTop = feed.scrollHeight;
      const prompt = Number(this.get('BS prompt length'));
      const step = Number(this.get('BS step'));
      const phase = this.get('BS ui phase', 'tokenizing');
      const generated = prompt > 0 ? Math.max(0, (this.get('BS out tokens', []).length || 0) - prompt) : 0;
      const bar = this.el('progress');
      bar.hidden = false;
      if (phase === 'layers') {
        const layer = Number(this.get('BS layer')) + 1;
        this.el('.phase').textContent = `${step < prompt ? 'Reading prompt' : 'Generating'} · layer ${layer} / 24`;
        bar.max = 24; bar.value = layer - 1;
      } else if (phase === 'head') {
        const rows = Number(this.get('BS head rows'));
        const scanned = Number(this.get('BS scanned'));
        this.el('.phase').textContent = 'Choosing next token';
        bar.max = rows || 1; bar.value = scanned;
      } else {
        this.el('.phase').textContent = phase === 'embedding' ? 'Loading token embedding' :
          phase === 'rendering' ? 'Decoding response text' : 'Tokenizing your message';
        bar.removeAttribute('value');
      }
      this.el('.counter').textContent = `${generated} tokens · ${Math.floor((performance.now() - this.began) / 1000)}s`;
      this.el('.detail').textContent = phase === 'tokenizing' ? 'Preparing tokenizer and chat template' :
        phase === 'head' ? `${Number(this.get('BS scanned')).toLocaleString()} / ${Number(this.get('BS head rows')).toLocaleString()} vocabulary rows scored` :
        prompt && step < prompt ? `Prompt token ${Math.min(step + 1, prompt)} of ${prompt} · ${generated} response tokens` :
        `${generated} response tokens · ${step} / 128 context positions used`;
      if (Number(this.get('BS done')) === 1 || Number(this.get('BS stop')) === 1) {
        const reason = this.get('BS stop reason', '');
        this.finish(reason === 'failure' ? 'Generation failed — check model package' :
          reason === 'context' ? 'Context limit reached' : reason === 'user' ? 'Stopped' : 'Complete');
      }
    }
    dispose() {
      clearInterval(this.timer);
      this.loader.runtime.removeListener('PROJECT_STOP_ALL', this.onStop);
      this.host.remove();
    }
  }

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
      if (typeof document !== 'undefined' && this.runtime && this.runtime.on && document.body) {
        const mount = () => {
          if (this.chat) this.chat.dispose();
          this.chat = null;
          const stage = this.runtime.getTargetForStage();
          if (stage && Object.values(stage.variables).some(v => v.name === 'BS prompt')) this.chat = new ChatView(this);
        };
        this.runtime.on('PROJECT_LOADED', mount);
        this.runtime.on('RUNTIME_DISPOSED', () => { if (this.chat) this.chat.dispose(); });
        mount();
      }
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
        {opcode: 'loadStrings', blockType: Scratch.BlockType.COMMAND, text: 'load metadata strings [KEY] into list id [DEST] request [ID]', arguments: {KEY: arg(S, ''), DEST: arg(S, ''), ID: arg(S, '1')}},
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
        if (manifest.schema !== 'bonsaiscratch-package/v1' || manifest.endianness !== 'little' || !['raw', 'bslut4', 'bslut5'].includes(manifest.codec) || !integer(manifest.payload_length) || manifest.payload_length !== file.size - payloadStart || !manifest.metadata || typeof manifest.metadata !== 'object' || Array.isArray(manifest.metadata) || !Array.isArray(manifest.tensors)) throw new Error('invalid manifest');
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
        this.prefetch.promise.catch(() => {});
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
      if (!this.file || !window || this.prefetch || window.end >= this.file.size || end >= this.file.size) return;
      if (end <= window.end - Math.min(this.windowBytes >> 2, 8 * 1024 * 1024)) return;
      this.prefetch = {start: end, promise: this.readWindow(this.file, end, this.generation)};
      this.prefetch.promise.catch(() => {});
    }
    publish(buffer, offset, length, groups, count, lists) {
      const view = new DataView(buffer, offset, length), bytes = new Uint8Array(buffer, offset, length);
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
    async loadStrings(args, util) {
      const id = String(args.ID), previous = this.requests.get(id);
      if (previous && previous.status === 'pending') previous.status = 'stale';
      const request = {status: 'pending', copied: 0, error: ''};
      this.requests.set(id, request);
      try {
        const generation = this.generation, target = util.target, key = String(args.KEY), dest = String(args.DEST);
        if (!this.file || !this.manifest) throw new Error('no package loaded');
        const entry = Object.prototype.hasOwnProperty.call(this.manifest.metadata, key) && this.manifest.metadata[key];
        const values = entry && entry.value;
        if (!Array.isArray(values) || values.some(value => typeof value !== 'string')) throw new Error('metadata is not a string array');
        const list = target && target.lookupVariableById(dest);
        if (!target || !this.runtime || !this.runtime.targets.includes(target) || !list || list.type !== 'list' || !Array.isArray(list.value)) throw new Error('invalid destination list');
        const owner = this.owners.get(list);
        if (owner && owner.status === 'pending') owner.status = 'stale';
        this.owners.set(list, request);
        await Promise.resolve();
        if (request.status !== 'pending' || this.requests.get(id) !== request || generation !== this.generation) return;
        if (!this.runtime.targets.includes(target) || target.lookupVariableById(dest) !== list || list.type !== 'list' || !Array.isArray(list.value) || this.owners.get(list) !== request) throw new Error('destination ownership changed');
        list.value.length = values.length;
        for (let i = 0; i < values.length; i++) list.value[i] = values[i];
        list._monitorUpToDate = false;
        request.copied = values.length; request.status = 'ok';
      } catch (error) {
        if (request.status === 'pending') { request.status = 'error'; request.error = String(error.message || error); }
      }
    }
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
          this.publish(window.buffer, abs - window.start, length, groups, count, lists);
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
        let buffer, offset = 0;
        if (length > this.windowBytes) {
          buffer = await this.read(file, abs, length);
        } else {
          const prefetch = this.prefetch;
          let window;
          if (prefetch && abs >= prefetch.start && abs + length <= prefetch.start + this.windowBytes) {
            try {
              window = await prefetch.promise;
            } catch (error) {
              if (request.status === 'pending' && this.requests.get(id) === request && generation === this.generation && this.prefetch === prefetch) this.prefetch = null;
              throw error;
            }
          } else {
            window = await this.readWindow(file, abs, generation);
          }
          if (request.status !== 'pending' || this.requests.get(id) !== request || generation !== this.generation) return;
          if (!this.runtime.targets.includes(target) || lists.some((list, i) => target.lookupVariableById(ids[i]) !== list || list.type !== 'list' || !Array.isArray(list.value) || this.owners.get(list) !== request)) throw new Error('destination ownership changed');
          if (window) {
            this.window = window;
            if (this.prefetch === prefetch) this.prefetch = null;
            this.maybePrefetch(abs + length);
          }
          const win = this.window;
          if (win && abs >= win.start && abs + length <= win.end) {
            buffer = win.buffer; offset = abs - win.start;
          } else {
            buffer = await this.read(file, abs, length);
          }
        }
        if (request.status !== 'pending' || this.requests.get(id) !== request || generation !== this.generation) return;
        if (!this.runtime.targets.includes(target) || lists.some((list, i) => target.lookupVariableById(ids[i]) !== list || list.type !== 'list' || !Array.isArray(list.value) || this.owners.get(list) !== request)) throw new Error('destination ownership changed');
        this.publish(buffer, offset, length, groups, count, lists);
        request.copied = count; request.status = 'ok';
      } catch (error) {
        if (request.status === 'pending') { request.status = 'error'; request.error = String(error.message || error); }
      }
    }
  }
  Scratch.extensions.register(new Loader());
})(Scratch);
