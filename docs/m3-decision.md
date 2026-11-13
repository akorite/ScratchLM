# M3 decision — kernel comparison and viability

Date: 2026-11-13. Host: Intel Xeon Platinum 8488C, Linux 7.0.0-1014-aws, Node v24.22.1, TurboWarp VM `0cb82afe62506f3d9f2ff849c95593f0473b5d2c`, compiler verified active, turbo mode, warp timer off, no renderer. **All results are host-specific; the Ryzen 7 Pro-class target was not measured.**

## Verdict

**Conditional redesign within the Scratch boundary; no full decoder integration yet.**

- Kernel choice: **BS-LUT4**. Per `benchmarks/results/kernels-xeon.json` (10M contributions, identical Q8_K inputs, generic-kernel F32 boundaries): BS-LUT4 0.34–0.36 s, raw-LUT4 0.36–0.37 s, scalar 1.40–1.44 s. Raw and BS-LUT4 outputs are bit-identical (`output_f32_sha256` matches across codecs in the head and tensor runs).
- Projected decode: **2.2–2.6 min/token** → plan classification *research-only* (2–5 min). Below the 5-minute pause trigger, above the 30–120 s demonstration target.
- Projected prefill (weights re-streamed per position, head rate as lower bound): 1 token ≈ 2.2–2.6 min, 16 ≈ 35–41 min, 128 ≈ 4.6–5.6 h. Prefill is unusable for interactive prompts and is reported independently per the gate.

## Evidence

| Measurement | Result | Source |
|---|---|---|
| Full head, 248,320 rows × 2048, raw | 34.97 s median wall (3 runs) | `benchmarks/results/head-xeon.json` |
| Full head, BS-LUT4 | 34.66 s median wall; identical output SHA-256 | same |
| `blk.3.attn_q.weight` full tensor (4,096×2,048) | 1.38 s raw / 1.25 s bslut4 median wall | `tensors-xeon.json` |
| `blk.0.ffn_down.weight` full tensor (2,048×6,144) | 1.84 s raw / 1.42 s bslut4 median wall | same |
| 1M contributions, width 2048 | scalar 0.178 s, raw-LUT4 0.063 s, BS-LUT4 0.049 s | `kernels-xeon.json` |
| 10M contributions, width 6144 | scalar 1.397 s, raw-LUT4 0.356 s, BS-LUT4 0.341 s | same |
| Transport, 64 KiB–4 MiB budgets | median 0.5–10.8 ms per request, warm | `transport-xeon-*.json` |
| Worst uninterrupted interval | 38.3 ms event-loop gap (head run) | `head-xeon.json` |
| Peak RSS, head run | ≤ 236 MB (Node VM, no renderer) | `head-xeon.json` |
| Numerical gates | all compiled-VM goldens pass; 38 pytest + 5 node tests green | this repo |

## Decode projection (shape-weighted, uncertain)

Model PQ2 payload: 151 tensors, 819,712 rows, 499.4 MB; 6,960 tiles of 120 rows (the benchmark tile size). Two bounds:

- **Bytes bound:** stream the whole PQ2 payload at the measured head rate 3.86 MB/s → **129 s**. This amortizes arithmetic inside the measured head stream; it assumes other tensors stream as efficiently, which is optimistic — the attn_q tile ran at a higher per-byte cost.
- **Tile bound:** apply the measured per-tile costs (17 ms width-2048, 102 ms width-6144, from the head and ffn_down medians) → **155 s**. Treats per-tile overhead as dominant, which the head contradicts slightly; the head's 65,280-byte tiles cost ~17 ms while attn_q's identical-size tiles cost ~39 ms, so per-request overhead is not fully linear.

Reported range: **129–155 s/token**, plus output/logit handling already inside the head measurement. Secondary work (normalization, embedding, state updates, per-request command latency) is only partially covered; add ≤ 20% uncertainty → **~2.2–3.1 min/token**.

Prefill reuses the same stream per position with no KV reuse measured yet: linear in positions until the attention path is measured.

## Gate assessment

| Gate | Status |
|---|---|
| Numerical gates pass | Pass — compiled-VM goldens green (38 tests), exact discrete checks, frozen tensor tolerances in `docs/numerical-contract.md` |
| End-to-end results support demonstration budget | **Fail** — 2.2–3.1 min/token is research-only, not the 30–120 s demonstration target |
| Memory bounded | Pass on host — ≤ 236 MB RSS measured; the 6 GB/8 GB thresholds are browser targets and were not exercised |
| Cancellation bounded | Partial — functional cancellation proven in `tests/transport/loader.test.js`; worst measured event-loop gap 38.3 ms, well under the 250 ms target, but no browser measurement |
| Decode > 5 min/token | No — pause trigger not reached |

## Required before M4/M5

1. The bottleneck is Scratch-side transfer and scheduler time (33.4 of 34.97 s in the head), not arithmetic or disk. Redesign options inside the Scratch boundary: larger effective tiles, fewer list writes per group (e.g. packed multi-value writes), or processing from a smaller resident window before revisiting the all-Scratch rule.
2. If demonstration-budget decode is a product requirement, it fails on this host and cannot be claimed on the target without redesign plus target measurement.
3. Prefill must be disclosed as hours-scale at 128 tokens in any published artifact.

## Limitations

Host-only results; Node VM without browser renderer; scratch tick timer is coarse, so wall-clock fields are the primary timings; cancellation and memory gates partially verified; head input is a captured transformed activation, not normalized model output (projection, not logits).
