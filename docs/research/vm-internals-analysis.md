# VM internals: TurboWarp list ops, scheduler, and the 33 s "scratch_transfer_and_scheduler"

Source: `/tmp/bonsaiscratch-reference/scratch-vm-0cb82afe62506f3d9f2ff849c95593f0473b5d2c` (rev 0cb82afe...). Harness: `tests/scratch/run-vm.js`, `benchmarks/head.py`, `extension/loader.js`.

## 1. Compiled list ops — real per-op JS cost (src/compiler/jsgen.js, jsexecute.js)

Lists are plain JS arrays (`Variable.value`, src/engine/variable.js:28 — `this.value = []` for `LIST_TYPE`). All hot list ops compile to direct array access + one dirty-flag write. Per-write cost is ~1–3 ns; there is **no resize logic, no monitor repaint per write, no bounds-check function call on the fast path**.

- `item (i) of list` → `listGet(list.value, idx)`, jsexecute.js:479-486. Fast path: `typeof idx === 'number' && idx >= 1` → `list[(idx|0)-1] ?? ''` (undefined → empty string). jsgen.js:252 special-cases `NUMBER_INDEX` even further to a raw `list.value[i-1] ?? ""`. No `listIndex()` call, no clamp on the fast path.
- `replace item i of list with v` → `listReplace`, jsexecute.js:493-510. Fast path (line 501): `typeof idx === 'number' && idx >= 1 && idx <= list.value.length` → `list.value[(idx|0)-1] = value; list._monitorUpToDate = false;`. Slow path goes through `listIndex` (jsexecute.js:457-464) which handles `'last'`/`'random'`/string coercion.
- `add v to list` → `list.value.push(item); list._monitorUpToDate = false;` (jsgen.js:693-694). O(1) amortized.
- `delete i of list` → `listDelete` (jsexecute.js:530-548): `shift`/`pop`/`splice` depending on index; middle delete is O(n) memmove. `delete all` → `list.value = []` (new array, jsgen.js:714 — note: replaces `.value`, which would break an extension holding a stale array reference).
- `insert at` → `splice` O(n) (jsexecute.js:516-525).
- `item # of v in list` / `contains` → `listFind`, jsexecute.js:301-321 — O(n) linear scan with per-element `+value`/lowercased-string compare. Expensive on big lists; avoid in inner loops.
- `length of list` → `list.value.length` (jsgen.js:263).
- Interpreter versions (uncompiled monitor threads etc.) live in src/blocks/scratch3_data.js:121-170 — same semantics, slightly slower (`lookupOrCreateList` + `Cast.toListIndex` per call).

`_monitorUpToDate` is only a dirty flag for the *visible list monitor*: the monitor's `getListContents` (scratch3_data.js:99-104) returns `list.value` unchanged if the flag is set, else `list.value.slice()` (full O(n) copy once per monitor refresh). **It costs one boolean store per list write; zero cost if no monitor is on stage.** The `monitors: []` in tools/generate_project.py:360 means the benchmark project has no monitors — the flag is pure dead store.

## 2. Internal representation

There is no custom packed repr — `list.value` is a normal JS `Array`. V8 element kinds apply: all-number contents ⇒ `PACKED_DOUBLE_ELEMENTS` (fast, unboxed); any string/object ⇒ `PACKED_ELEMENTS` (boxed, slower reads and the `?? ''` checks get marginally pricier); holes ⇒ `HOLEY_*`. The loader's `values.length = count` then fill (loader.js:130-131) keeps the array packed if you overwrite every slot; skipping slots makes it HOLEY. Keeping all numbers (the loader's bytes/uint16 scales) is already optimal — mixing strings into a math list would silently demote element kind and deopt the compiled `listGet` reads.

## 3. loadGroups yield behavior & where 33.4 s goes

`loadGroups` is an `async` command block (loader.js:94). Compiled extension calls run through `executeInCompatibilityLayer` (jsexecute.js:116-210): the returned Promise is handed to `waitPromise` (jsexecute.js:117-138), which sets `thread.status = STATUS_PROMISE_WAIT` (== 1) and `yield`s the compiled generator. So **every loader call = one promise + one generator suspend/resume round**. The `.then` callback flips status back to `STATUS_RUNNING` (line 129) — resume happens on the *next pass* of `sequencer.stepThreads`'s inner `for` loop (sequencer.js:104-131), usually the same tick if any thread is still active.

`stepThreads` budget: `WORK_TIME = 0.75 * currentStepTime` (sequencer.js:74) with `currentStepTime = 1000/30` (runtime.js:346, setInterval at 30 fps in tw-frame-loop.js:102-104; turboMode does NOT shorten the interval — it only makes the while-loop ignore `redrawRequested`, sequencer.js:92). Per `_step` (runtime.js:2526-2604): edge-activated hat scan (2546-2551), `_pushMonitors` → `monitorBlocks.runAllMonitored` (2553, blocks.js:881-898 — zero monitored blocks here), `stepThreads` (2561), monitor-state emit (2601-2604), renderer draw (skipped in Node, no renderer). These per-tick overheads are µs-scale; they do not explain 33 s.

Where the 33.4 s actually goes: `loader_read_ms` (0.5 s) + `loader_command_ms` (0.8 s) + `loader_copy_and_overhead_ms` (0.27 s) are measured *inside* load; `compute_tick` ~0.7-1.2 s is timed inside Scratch. The residual ~32 s is **scheduler idle time between `stepThreads` passes**: 2070 requests × ~16 ms of event-loop/promise latency each. Each `loadGroups` promise resolves on a *microtask* while `stepThread` has already returned; the generator `.next()` for resume only runs when the sequencer loops again (or the next 33 ms `_step` if the inner loop exited with `numActiveThreads === 0`, sequencer.js:92-93). Worst-case path (only thread in PROMISE_WAIT when the inner loop finishes) parks the thread until the next `setInterval` tick — i.e., up to 33 ms per loader call. Timer checks (`Timer.timeElapsed()` per inner pass, sequencer.js:92) and thread bookkeeping are nanoseconds; the real cost is *waiting*, not executing.

## 4. Warp mode / `run without screen refresh`

`warp_timer: false` in head-xeon.json means the driver script is NOT warp (procedures marked `warp: true` only for leaf math procs, generate_project.py:47,113). In compiled warp scripts, `yieldLoop`/`yieldStuckOrNotWarp` emit `if (isStuck()) yield` (jsgen.js:1078-1084) where `isStuck` checks the 500 ms sequencer timer every 100 calls (jsexecute.js:35-41). Non-warp scripts `yield` unconditionally per loop iteration — but note each `yield` is just a generator suspend costing one inner-loop pass (microseconds), NOT a frame, unless no thread stays RUNNING. Warp would matter if compute blocks ran per-iteration yields inside a non-warp driver — the measured ~1 s `compute_tick` already reflects whatever yields the generated project has. Putting the *whole* head scan in a `run without screen refresh` procedure would collapse all per-iteration generator yields to `isStuck()` checks — relevant only if compute time becomes the bottleneck after transfer is fixed.

## 5. Cheapest deposit of N numbers into a list

The loader already does the optimal thing (loader.js:121-132): `list.value` is the raw JS array — set `.length`, write `arr[i] = number`, set `list._monitorUpToDate = false` once. 65,280 writes is a few hundred µs; **a single block call can write any N in one tick** — a long-running block doesn't yield unless its code explicitly yields/awaits (command blocks don't re-enter the sequencer mid-execution; they only delay the next `_step`).

The one remaining cost is the `async` keyword itself (loader.js:94-96): it forces the `waitPromise` suspend/resume round per call. Cheapest design:

- Make the command **synchronous**: return a value, not a Promise. `executeInCompatibilityLayer` only enters `waitPromise` when `isPromise(returnValue)` (jsexecute.js:170). A sync `loadGroups` that does `list.value` writes inline completes in ONE generator step — zero yields, zero promise allocations, zero scheduler idle. That removes ~all of the 32 s residual.
- How to go sync when the source is a Blob: read the tensor once (`await file.slice(...).arrayBuffer()`) during `openFile`/a first request, cache the `ArrayBuffer` on `this`, then per-request copies are pure sync `DataView`/`Uint8Array` reads + `list.value` writes. `await this.read` (loader.js:116) is what makes every call async today.
- Keep writing into `list.value` in place (`length =` + element stores) rather than replacing `.value` — replacing the array invalidates references and creates a new hidden-class transition; also `delete all` in compiled code assigns `list.value = []` (jsgen.js:714), so don't cache the array across calls — re-read `list.value` each invocation.
- Don't skip `_monitorUpToDate = false` (one boolean store, needed only if a list monitor could be visible — keep it, it's free).
- If batching further: one `loadGroups` call can fill multiple lists / larger tiles; the per-call fixed cost is what dominates (promise + suspend + scheduler gap), not per-element work.

Net: the fix is "one `await` per tensor, not per request" + sync command blocks. No VM changes needed; `.value` mutation is already the intended fast path (TurboWarp's own compiled ops write it directly).
