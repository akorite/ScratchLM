# Post-M3 speed research — scheduler amortization and deeper LUTs

**Headline measured fact for the compute direction:** Bonsai's trit zero density is a uniform **~32.6%** across all 151 PQ2 tensors (sampled ~1-in-32 groups of `model-raw.bsp`; per-tensor spread <0.005). BITCOS's ~40–50% sparsity doesn't apply here; CSR costs ~3 steps/nonzero ≈ 2 steps/weight at this density vs ~0.6 for a g=5 LUT. **CSR is ruled out; deeper LUT is the compute direction.**

Date: 2026-11-13. Supplements `docs/m3-decision.md`. All measurements on the same Xeon host, Node VM, pinned VM `0cb82afe62506f3d9f2ff849c95593f0473b5d2c`, compiler on, no renderer. Not target-hardware results.


## 1. Root cause of `scratch_transfer_and_scheduler`

The head scan's 33.4 s is **scheduler pacing, not data movement**. Two compounding mechanisms, verified against the pinned VM source:

- **Parked threads.** `loadGroups` is `async`; its Promise is handed to `waitPromise`, which sets `STATUS_PROMISE_WAIT` and yields (`jsexecute.js:117-138`). Microtasks cannot run while `stepThreads` occupies the JS stack, so promise resolution — and the status-restoring `.then` — lands between `setInterval` ticks (`tw-frame-loop.js:102-104`). Every async `loadGroups` costs ≥1 frame. 2,070 tiles × ~16.7 ms ≈ 34.5 s. The extension's own work is 1.6 s; the copy ~0.27 s.
- **75% duty bound.** `stepThreads` runs until `WORK_TIME = 0.75 × currentStepTime` elapses (`sequencer.js:74`; `currentStepTime = 1000/framerate`). Non-warp loop iterations each emit a `yield` (`jsgen.js:1055-1072`), but a yield costs only a generator suspend — microseconds — not a frame. The real bound: **wall ≈ work ÷ 0.75**, plus one frame per parking wait. Tile-120 does ~12 ms of in-tick work ≈ WORK_TIME (12.5 ms), so it only *looks* like "one iteration per frame."
- CPU profile of a 48 k-row run: ~50% idle, 12% `Timer.timeElapsed`, ~8% sequencer. Confirms pacing, not work.

Fix space: remove parks (synchronous loader or batched calls), reduce parked iterations (bigger tiles), or shrink the frame interval (higher framerate). `control_wait 0` also calls `requestRedraw()` each tile — harmless under turbo mode but a second hidden pacer if turbo is ever off.

## 2. Measured variants (48,000 head rows, raw codec, tile loop unmodified)

| Configuration | Transfer+scheduler (s) | Compute tick (s) | Requests | Worst event-loop gap |
|---|---|---|---|---|
| Baseline 60 fps, tile 120 | 6.46 | 0.19 | 400 | 34.5 ms |
| 250 fps, tile 120 | 1.52 | 4.07 | 400 | 29.6 ms |
| 60 fps, tile 7,680 | 0.14 | 3.85 | 7 | 41.4 ms |
| 250 fps, tile 1,920 | 0.10 | 4.29 | 25 | 27.0 ms |
| 250 fps, tile 7,680 | 0.07 | 4.26 | 7 | 34.4 ms |
| Sync-cached loader, 60 fps, tile 120 | 6.48 | 0.12 | 400 | 34.5 ms |
| Warp-wrapped driver, sync loader, tile 120 | 6.50 | ~0 | 400 | 55.6 ms |

(Compute ticks are internally inconsistent — 0.12–0.19 s at tile 120 vs ~4.3 s at big tiles for identical per-row work; `sensing_timer` is real `Date.now`, so the small-tile readings are an artifact, not per-row time. Take the big-tile values: ~4.3 s per 48 k rows ≈ 90 µs/row *elapsed*, which the 75% duty bound puts at ≤ ~67 µs/row actual work. The sync-loader row did genuinely execute synchronously — verified in the variant source — so its 6.48 s = 400 × ~12 ms ÷ 0.75 is explained by the duty bound alone, no parks needed. The warp row's unchanged 6.50 s and 55.6 ms gap remain anomalous; the variant is rejected anyway.)


Conclusions:

- **250 fps (a TurboWarp project setting, cap 250 — `runtime.setFramerate`, `runtime.js:2677-2683`) collapses the scheduler tax 4×.** Legitimate in the packaged project; the deployed app controls it.
- **Tiles ≥ ~1,920 rows make wall ≈ work.** Tile 7,680 fits just under the 4 MiB loader bound (7,680×16×34 B = 4.18 MB). Cancellation is checked per tile iteration, so latency ≈ tile *wall* time: 1,920 rows ≈ 170 ms work → ~230 ms wall at 250 fps — under the 250 ms target but thin; 7,680 ≈ 690 ms work → ~920 ms wall, over. ~1,000–2,000-row tiles at 250 fps is the responsive sweet spot, not maximal tiles.
- Sync loader (ArrayBuffer cached at `openFile`, `load()` non-async) is still valuable as the production shape — removes the promise path and per-request event-loop exposure — but must be paired with fps/tile changes. Whole-file caching pushed Node RSS to 1.26 GB; a bounded sliding window (e.g. 64–256 MB read-ahead, async refill, sync copies inside) fits the 6 GB browser-memory gate and keeps "bounded transport" honest.

Projected decode with 250 fps + ~1,920-row tiles + sync window: head ≈ 22 s → **~75–85 s/token**, inside the 30–120 s demonstration band on this host, before any kernel change.

## 3. Kernel-level literature transfer (compute floor ≤ ~90 µs/row elapsed, ~67 µs/row work)

From `docs/research/ternary-matvec-literature.md` (T-MAC arXiv:2407.00088, bitnet.cpp TL1/TL2 arXiv:2410.16144, LUT-GEMM arXiv:2206.09557, FLUTE arXiv:2407.10960, BITCOS arXiv:2609.16338, llama.cpp TQ1_0/TQ2_0):

- Our BS-LUT4 (81-entry per 4 trits) is already the right family — bitnet.cpp's activation-side LUT scheme. The literature caps group size at 2–4 trits due to register/SMEM limits; **Scratch has none, so g=5 (243 entries) or g=6 (729) wins in principle**: ~0.6 block steps/weight vs 0.75 at g=4 (1.25× fewer lookups), plus a cheaper DP table build (LUT[p] = LUT[⌊p/3⌋] ± a_last, zero multiplies). Two corrections to the naive gain: the build is unmeasured, and 2048/6144 are not divisible by 5 or 6 — group padding and misalignment with the 256-element Q8 scale blocks need a codec that handles partial groups. Realistic expectation ≈ 1.3–1.6× on the scan → decode ≈ 50–65 s/token; microbenchmark before committing.
- Precompute the base-3 **index stream offline** (FLUTE's offline restructuring) so the hot loop is `acc + item (j·3^g + idx) of LUT` — no unpacking math in blocks. The `.bsp` package can carry a packed-index codec; decoding stays Scratch-side math.
- **CSR index-list fallback measured and rejected** (BITCOS: real ternary models are 29–51% zeros): Bonsai is uniformly ~32.6% zeros — see headline. ~3 steps/nonzero ≈ 2 steps/weight vs ~0.6 at g=5.
- Dead ends confirmed: MADDNESS/a4.8-style activation quantization (no int8 fast path in binary64), mirror-consolidated LUTs (conditional negate per lookup costs more than halved build), trit-packing into numbers for the index stream (marginal vs flat index list), Scratch thread parallelism (single-threaded; adds scheduler overhead only).
- Community technique file `docs/research/scratch-speed-techniques.md` corroborates: batch all extension→list transfer in one unsandboxed call (already our design), hide monitors, avoid `item # of`/`contains` (O(n) scans), keep lists homogeneously numeric (V8 `PACKED_DOUBLE_ELEMENTS`), decode strings via `letter of`/`unicode of` if a packed-string transport is ever needed (~14 M chars/s in warp).

## 4. Revised outlook vs the M3 record

| Stage | M3 estimate | Post-research estimate (same host) |
|---|---|---|
| Decode/token | 129–155 s | ~75–85 s (transport fixes) → ~50–65 s (g=5–6 LUT) |
| Head scan | 34.97 s | ~17–22 s → ~11–17 s |
| Classification | research-only | demonstration band plausible on host; target still unmeasured |
| Prefill 128 tokens | ~4.6–5.6 h | ~2.7–3.9 h (linear in positions; still hours — report independently) |

Revised recommendation stands within the Scratch boundary: **redesign the transport/driver path (250 fps + ~1–2 k-row tiles + sync-window loader) and microbenchmark the g=5/6 LUT before any M4 work.** The 5-minute pause trigger was never in danger; the demonstration budget is now plausible rather than failed.

## 5. Risks / unverified

- 250 fps raises the `_step` tick rate; the 4 ms `setInterval` floor has zero headroom, and Chrome background-tab throttling (clamps non-rAF timers toward ~1 Hz) would turn decode into ~7 min/token — a hard gate for unattended decode; verify foreground behavior in a packaged build.
- `item of` cost on 300k+ entry LUTs assumed O(1) — microbenchmark before committing g=6 flat LUTs; fall back to per-k-block LUTs if index cost grows.
- 2048/6144 not divisible by 5 or 6: g=5/6 LUTs need a padded/partial-group codec and realignment with the 256-element Q8 scale blocks — design work, not just a microbenchmark.
- Sync-window loader changes cancellation semantics (window misses must remain cancellable); keep `request`/`owners` staleness checks.
- All projections still host-only; the Ryzen-class target can shift every number.
