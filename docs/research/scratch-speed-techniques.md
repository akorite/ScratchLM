# TurboWarp/Scratch data-movement & compute speed techniques

Context: ScratchLM/BonsaiScratch. A 248,320-item head scan takes ~35 s, ~33 s of it
Scratch-side transfer/scheduler, ~0.4 s arithmetic. Goal: move data extension→Scratch
cheaply under the rule "extension does file I/O + copies values into lists; all math in blocks".

Key measured anchors (all from TurboWarp docs/source unless noted):
- Compiled `replace item` = `list.value[index] = v; list._monitorUpToDate = false;`
  — an array store plus a flag. Quicksort over 200,000 list items: **45 ms** on a 4790k.
  So pure compiled list writes are ~10–50 ns/item → a 248k write burst should cost ~5–15 ms.
- **33 s for 248k items ≈ 133 µs/element** — three orders of magnitude above a compiled
  `replace item`. That means the cost is *not* the list write itself; it is per-element
  overhead around it (per-item extension calls, non-warp yields, or sandbox delays).
- Sandbox doc: **every call to a sandboxed extension block forces ≥1 frame of delay,
  even in warp mode / turbo mode.** Max ~30–60 calls/sec. If the current pipeline calls
  a sandboxed extension block per item (or per batch), that alone explains the 33 s.
- Project https://turbowarp.org/782840280 decodes a **700,000-char string in 0.05 s**
  in Scratch → string `letter i of` loops run at ~14 M chars/s when compiled+warped.
- Sequencer: warp-mode threads that yield are **re-executed immediately in the same tick**
  (sequencer.js: "In warp mode, yielded blocks are re-executed immediately") — one warp
  thread runs to completion; there is no parallel compute.

## Ranked techniques

### 1. Single unsandboxed extension call that writes the whole list — est. 1000–10000× (33 s → ~3–30 ms/row) — STRONGEST
Unsandboxed extension blocks run as plain JS in the VM thread and get full VM access:
```js
const vm = Scratch.vm;
const target = util.target; // or stage
const list = target.lookupVariableByNameAndType('weights', 'list');
list.value = arr;            // arr built natively; or arr.push(...) in a JS loop
list._monitorUpToDate = false;
```
`list.value` is a plain JS array; assignment is O(1)-ish and building it natively is
memcpy-speed. This stays inside the allowed constraint ("copying values into Scratch
lists") — the extension still only does I/O + a copy, zero math in JS.
Evidence:
- docs.turbowarp.org/development/extensions/unsandboxed ("CAN access parts of the project
  that were not passed in as an argument, such as variables", blocks are "actually
  instantaneous — no forced 1 frame delay")
- docs.turbowarp.org/development/extensions/sandbox (forced ≥1 frame delay per sandboxed
  call — "makes custom extensions almost useless" for tight loops)
- github.com/TurboWarp/extensions/issues/109 — GarboMuffin confirms the frame-delay
  bottleneck diagnosis; user's script ran "1 letter per frame".
- scratch-vm src/engine/variable.js — list `.value` is a plain array; `_monitorUpToDate`
  is the only bookkeeping.

### 2. If the extension must stay sandboxed: return one big value per call, not per item — est. 50–500×
Sandbxed blocks can't touch lists, but a reporter can return an arbitrarily large
string in ONE call (postMessage cost is flat, not per byte):
- **String packing**: extension returns row encoded as UTF-16 chars. 8 trits/char
  (3^8=6561 ≤ 65536) → 248,320 trits = ~31k chars. Scratch decodes with
  `letter i of` / `unicode of` in one warp block: ~14 M chars/s → **~20–50 ms/row**
  including decode. Decode is visible blocks, so the math rule is preserved.
- Or return a delimited string and split in blocks — but repeated `join`/`letter`
  slicing is slower; prefer fixed-width chars.
- Even returning one *number* per call caps at ~30–60/s sandboxed → still 1+ hour.
  Never call a sandboxed block inside a per-element loop. Batch per call.
Evidence: extensions issue #109 (same failure mode + fix), sandbox doc.

### 3. Always compute inside `run without screen refresh` + warp timer OFF — est. 2–10× on all inner loops, enables everything below
- Warp scripts run to completion in one tick; yields are re-executed instantly
  (sequencer.js). A non-warp repeat loop yields each iteration → 30–60 iterations/s max.
  That alone turns ms into seconds.
- New compiler (Sept 2025, now default): only eliminates numeric casts and specializes
  when the script can't be interrupted — i.e. **warp blocks with the warp timer disabled**.
  The warp timer (500 ms forced pause) is ON in the editor; package the project
  (packager.turbowarp.org) or disable it to get ~2× better codegen.
- Numbers + one script: keep all decode/copy work inside one warp custom block.
Evidence: docs.turbowarp.org/new-compiler, /warp-timer, sequencer.js lines ~226.

### 4. Pack values into numbers — est. up to 33× less transfer, trivial decode
Scratch numbers are binary64: pack **33 ternary digits per number** (3^33 ≈ 5.6e15 < 2^53)
or 15 decimal digits. A 248k-trit row becomes ~7.5k list slots — 33× fewer cells to fill
and to scan. Decode per weight = ~3–5 cheap compiled ops (`mod`, `floor`, `÷`); at compiled
speeds (~100 M ops/s) that's ~10 ms/row of extra visible math. Worth it only if transfer
stays per-item; if #1 lands, packing matters more for memory (2 B model is the real problem).
Caveat: keep packed lists homogeneous (all numbers) — mixed types deoptimize V8 element kinds.

### 5. Hide all list/variable monitors — avoids O(n) per-frame copies
When a list monitor is shown, `getListContents` runs `list.value.slice()` every frame —
a 248k-element copy per frame (~ms+ each). Keep all monitors hidden during decode.
The `_monitorUpToDate` flag makes hidden-monitor lists free.
Evidence: scratch3_data.js getListContents/updateMonitor; listReplace sets the flag only.

### 6. Replace `item # of` / `contains` with direct indexing — O(n) → O(1)
`item # of [x] in [list]` and `[list] contains` are linear scans with Scratch's weird
comparability rules (scratch3_data.js). For LUTs (e.g. char→value), use `unicode of` /
`letter i of` or direct numeric indexing — never search.
For 2D weights: Scratch lists are flat; use `item (row*cols + col)` index math — O(1).

### 7. Loop hygiene that compounds — est. 1.5–3×
- `repeat (n)` compiles to a `for`; hoist `length of list` and any constant reporters
  out of loops (new compiler may do this, but don't rely on it).
- Prefer `replace item` over `insert`/`delete` mid-list (those are O(n) splice ops).
- Preallocate once (`delete all` then `add` to fill once), then only `replace`.
- Build strings with ONE `join`, never `join` inside a loop (O(n²) rope rebuild).
- Keep variables numeric (avoid `join` writing strings into loop counters — defeats the
  new compiler's type specialization).
- Loop unrolling (~4–8× per iteration batch) helps in old compiler; less needed now.

### 8. Parallelism: essentially none for compute — don't bother
- JS is single-threaded; Scratch threads are cooperative. Broadcast-and-wait and clones
  just add scheduler overhead and serialize; a warp thread hogs the tick anyway.
- Two non-warp scripts interleave per-block; warp = run-to-completion. Splitting the
  scan across N clones gives ~no speedup and N× bookkeeping.
- The one useful case: a *waiting* thread (extension Promise for file I/O) lets other
  threads run — you can overlap file load with compute, nothing more.
Evidence: sequencer.js stepThreads loop; docs.turbowarp.org/how.

### 9. What fast projects actually do (for the record)
- Linux-in-Scratch, SHA-256 at 3010 H/s, raycasters: everything lives in one warp
  custom block; data in flat lists or packed strings; LUTs for decode; zero extension
  calls in inner loops. New compiler made Linux boot 21 s → 12 s.
- Save-code / video projects (Bad Apple etc.): bulk data travels as ONE big string,
  decoded in a warp loop — matches technique #2 exactly, and GarboMuffin's 700k-char/
  0.05 s datapoint is from exactly this pattern.
- Nobody fast does per-element extension calls; the community bottleneck fix is always
  "batch it into one call / one string".

## Bottom line for ScratchLM
Diagnosis of the 33 s: per-element overhead, not list writes — almost certainly
per-item (or per-small-batch) extension calls, likely sandboxed (frame-delayed) or
inside a non-warp loop. Fix order:
1. Make the extension unsandboxed (file-loaded extensions can run unsandboxed via
   "Run extension without sandbox") and fill `list.value` in ONE block call → ~ms/row.
2. If it must stay sandboxed: return one packed UTF-16 string per row (8 trits/char),
   decode in one warp block → ~20–50 ms/row, and 33× memory savings vs flat trits.
3. Run decode+matvec in warp custom blocks, warp timer off, monitors hidden.

## Evidence links
- https://docs.turbowarp.org/how — compiled codegen, listReplace source, quicksort 200k/45 ms
- https://docs.turbowarp.org/new-compiler — type specialization, warp-timer caveat, Linux-in-Scratch numbers
- https://docs.turbowarp.org/development/extensions/sandbox — ≥1 frame delay per sandboxed call
- https://docs.turbowarp.org/development/extensions/unsandboxed — Scratch.vm access, instant blocks
- https://github.com/TurboWarp/extensions/issues/109 — real-world per-call frame delay + fix
- https://turbowarp.org/782840280 — 700k-char string decoded in 0.05 s (cited by GarboMuffin in #109)
- scratch-vm src/blocks/scratch3_data.js, src/engine/variable.js, src/engine/sequencer.js (develop branch)
- https://docs.turbowarp.org/warp-timer — 500 ms forced pause, disable for packaged runs
