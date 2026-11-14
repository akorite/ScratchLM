# M3 decision — speed redesign, kernel comparison, and viability

Date: 2026-11-14. Host: Intel Xeon Platinum 8488C, Linux 7.0.0-1014-aws, Node v24.22.1, TurboWarp VM `0cb82afe62506f3d9f2ff849c95593f0473b5d2c`, compiler verified active, turbo mode, warp timer off, 250 fps, 1,280-row tiles (width 2048; 426 rows at width 6144), synchronous 64 MiB read-ahead window loader, no renderer. **All results are host-specific; the Ryzen 7 Pro-class target was not measured.**

**Measurement caveat:** The original head and tensor runs below wrapped every `loadGroups` call in a promise, including synchronous window hits. Their wall times describe that instrumented path, not the synchronous fast path; their `loader_copy_and_overhead_ms` values are invalid because background reads overlap commands. The corrected harness reports read and command durations separately, which may overlap; its read timer starts after package opening, so the initial prefetch may be excluded.

## Verdict

**M5 readiness remains conditional on the remaining integration gates.** The current `bslut5` M4 head median is 21.68 s, approximately the plan's ~22 s target; full-token decode projections and target hardware remain unverified.

- Adopted codec: **`bslut5`** (5 trits per index byte, 26 index bytes + 6 zero spare bytes inside each 34-byte record, parameterized DP LUT build + cursor inner loop). On `benchmarks/results/kernels-xeon.json` (10M contributions, identical Q8_K inputs): `bslut5` setup+compute totals `0.537 s` (w=2048) / `0.587 s` (w=6144) vs cursor `bslut4` `0.566 s` / `0.599 s` vs `raw-lut4` `0.597 s` / `0.667 s` vs `scalar-raw` `1.869 s` / `1.830 s`. On the full 248,320-row head (`benchmarks/results/head-xeon.json`), `bslut5` finishes in **`20.88 s` median** (`21.06 s` worst) vs `22.76 s` (`bslut4`) and `24.80 s` (`raw`), with bit-identical `output_f32_sha256` across all three codecs.
- Projected decode: **`74–102 s/token`** core matvec+setup+transfer (`~74–122 s/token` with up to 20% secondary-op margin) → **demonstration band (30–120 s)** on this host, lifted from the prior `2.2–3.1 min/token` research-only pause.
- Projected prefill (weights re-streamed per position, linear in tokens): 1 token ≈ `74–102 s`, 16 ≈ `20–27 min`, 128 ≈ `2.6–3.6 h`. Prefill remains hours-scale and unsuitable for interactive long prompts; it is reported independently.

## Corrected-harness rerun

Three runs per case on the same Xeon host, 250 fps and 1,280-row tiles: the 248,320-row `bslut5` head had a 22.10 s median and 23.01 s worst case, versus 23.62 s for `bslut4` and 25.44 s for raw. The M4 head median was 22.64 s (22.97 s worst); host-sorted output matched the native top-8 order in every run. See `benchmarks/results/head-sync-xeon.json`, `tensors-sync-xeon.json`, and `head-m4-sync-xeon.json`. The earlier measurements and projections below are historical and have not been recalculated for this rerun.

The plan specifies an approximate ~22 s head target rather than a strict 22.00 s cutoff. Measured head timing is host-specific and does not establish full-token decode or prefill performance.

At the retuned 640-row default, three compiled M4 head runs in `benchmarks/results/head-m4-topk-xeon.json` had a 21.68 s median and 23.40 s worst wall time. Scratch-selected top-8 and greedy tokens matched the native fixture in all three; maximum error across all 248,320 logits was `1.91e-6`, below half the native `0.990841` top-1/top-2 margin. The full native logits were regenerated outside Git from the locked model and reference build.

## Evidence

| Measurement | Prior baseline (60 fps, tile 120, async) | Redesign (`bslut5`, 250 fps, tile 1280, sync window) | Source |
|---|---|---|---|
| Full head, 248,320 rows × 2048 | 34.97 s raw / 34.66 s bslut4 | **20.88 s** bslut5 (22.76 s bslut4, 24.80 s raw); identical output SHA-256 | `benchmarks/results/head-xeon.json` |
| Head transfer + scheduler time | 33.4 s | **1.00 s** (97% reduction) | same |
| `blk.3.attn_q.weight` (4,096×2,048, incl. VM startup) | 1.38 s raw / 1.25 s bslut4 | 1.65 s bslut5 / 1.66 s bslut4 / 1.72 s raw (compute 0.67 s bslut5) | `benchmarks/results/tensors-xeon.json` |
| `blk.0.ffn_down.weight` (2,048×6,144, incl. VM startup) | 1.84 s raw / 1.42 s bslut4 | 2.13 s bslut5 / 1.96 s bslut4 / 2.25 s raw (compute 1.08 s bslut5) | same |
| 10M contributions, width 2048 | 0.355 s bslut4 compute | 0.477 s bslut5 + 0.060 s setup (**0.537 s**) vs 0.541 s + 0.025 s bslut4 (**0.566 s**) | `benchmarks/results/kernels-xeon.json` |
| 10M contributions, width 6144 | 0.341 s bslut4 compute | 0.476 s bslut5 + 0.111 s setup (**0.587 s**) vs 0.547 s + 0.052 s bslut4 (**0.599 s**) | same |
| Transport, 64 KiB–4 MiB budgets | 0.5–10.8 ms + ~16.7 ms promise park in VM | **0.36–67.3 ms** sync window hits; 148–165 ms initial cold window fill | `benchmarks/results/transport-xeon-*.json` |
| Cancellation latency (tile 1280, `bslut5`) | — (gap 38.3 ms) | **166.1 ms** steady-state abort (`test_head.py`); max loop gap **84.0 ms** (`bslut5`) | `tests/scratch/test_head.py`, `head-xeon.json` |
| Peak RSS, head run | ≤ 236 MB | **≤ 511 MB** (Node VM, 64 MiB read-ahead window, no renderer) | `head-xeon.json` |

Note on `item of` scaling at large LUT size: width-2048 (`bslut5` LUT ≈ 101k entries) and width-6144 (`bslut5` LUT ≈ 303k entries) achieve identical 10M-contribution compute times (`0.477 s` vs `0.476 s`), confirming O(1) list indexing up to ~303k entries in the tested VM.

## Decode projection (`bslut5`, shape-weighted, uncertain)

Model PQ2_0 payload: 151 tensors, 819,712 rows, 499.40 MB, 1.8801B multiply-accumulate contributions. Two bounds:

- **Head-rate bound:** subtract ~0.9 s one-time VM+window startup from the 20.88 s head wall (`19.98 s` for 508.6M contributions) and scale by `1.8801B / 0.5086B` → **`73.9 s/token`** (`1.23 min`). Assumes layers achieve the same amortized scan efficiency as the 248,320-row head.
- **Kernel+setup bound:** apply the measured 10M-contribution `bslut5` compute rates (`0.477 s` at width 2048, `0.476 s` at width 6144), add all 151 per-tensor DP LUT setups (`0.060 s` × 103 + `0.111 s` × 48 ≈ `11.5 s`), plus ~2 s for 8 sequential 64 MiB window refills across 499.4 MB → **`101.9 s/token`** (`1.70 min`).

Reported core range: **`74–102 s/token`**; adding up to 20% headroom for secondary operations (Hadamard transforms, RMSNorm, recurrent/KV state updates, F32 rounding) gives **`~74–122 s/token`** (`1.2–2.0 min/token`).

Prefill reuses the same weight stream per prompt position with no KV-batched matmul: linear in prompt length (`~2.6–3.6 h` at 128 tokens).

## Gate assessment

| Gate | Status |
|---|---|
| Numerical gates | **M4 head passes** — Scratch top-8 and greedy match, and the full-logit maximum error is `1.91e-6` against a native margin of `0.990841` |
| Head scan ≤ ~22 s on host | **Approximately met** — the current M4 head median is `21.68 s` (`23.40 s` worst) with `bslut5` |
| Projected decode ≤ 120 s/token on host | **Unconfirmed for corrected rerun** — historical estimate was `74–102 s/token` core, before secondary-op margin |
| Cancellation ≤ 250 ms & gap recorded | **Retuned** — a 1,280-row M4 cancellation test took `330.0 ms`; the default is now 640 rows, and the focused cancellation tests passed their 250 ms bound. Earlier 1,280-row measurements are in the evidence table |
| Memory bounded | **Pass on host** — ≤ 511 MB RSS in Node VM (includes 64 MiB read-ahead window); browser 6 GB/8 GB thresholds remain unexercised in a browser renderer |
| Decode > 5 min/token | **No** — pause trigger cleared |

## M4 end-to-end head evidence (Xeon host)

After integrating `output_norm` (mode 5 RMS norm + F32-rounded `output_norm.weight`), explicit ±1 `prism.hadamard.signs.2048` dense loading, normalized block-512 Sylvester-Walsh-Hadamard transform, Q8_K quantization, `bslut5` DP LUT scan across all 248,320 `token_embd.weight` rows (zero pruning), and deterministic lower-ID-tie-breaking greedy selection in Scratch (`benchmarks/results/head-m4-xeon.json`, 3 repetitions, tile 1,280, `bslut5`):

| Metric | Measured result | Source |
|---|---|---|
| End-to-end M4 head wall time | **24.16 s** median (`27.73 s` worst; compute `21.72–25.61 s`) | `benchmarks/results/head-m4-xeon.json` |
| Vocabulary rows scanned | 248,320 / 248,320 (complete vocabulary) | same |
| Greedy next token (token 1 input) | **321** (exact match to native reference) | `fixtures/native/m4/head/manifest.json` |
| Top-8 token ID order | `[321, 220, 369, 3709, 318, 198, 2, 653]`, obtained by host sorting (exact match) | same |
| Top-8 max abs logit error vs margin | `9.54e-7` top-8-only error vs `0.990841` top-1/top-2 margin; not the all-logit maximum required for acceptance | same |
| Worst event-loop gap | **96.3 ms** (`66.7–96.3 ms` across 3 runs) | `benchmarks/results/head-m4-xeon.json` |
| Peak RSS | **515.1 MB** (Node VM, no renderer) | same |
| M4 cancellation latency (mid-scan abort) | `< 250 ms` at tile 640 in the current focused test; a 1,280-row run took `330.0 ms` | `tests/scratch/test_head.py` |

**M5 readiness:** Scratch-side top-8, greedy selection and the full-logit margin check pass on this Xeon host. Full-token decode remains a projection, and the full decoder and per-layer state remain unimplemented.

## Limitations

Host-only results on Intel Xeon Platinum 8488C in headless Node VM (no browser renderer); Ryzen 7 Pro-class target unmeasured; Chrome background-tab timer throttling (~1 Hz non-rAF clamp) would stall 250 fps unattended execution in background tabs; browser renderer memory unmeasured; prefill remains linear in prompt length (`~2.6–3.6 h` at 128 tokens) and is reported independently from decode.
