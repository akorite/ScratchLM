---
title: BonsaiScratch M0–M3 - Plan
type: feat
date: 2026-11-12
artifact_contract: ce-unified-plan/v1
product_contract_source: user-provided BonsaiScratch revised implementation plan v1.1
execution: code
---
# BonsaiScratch M0–M3 - Plan

## Goal Capsule

Implement and measure the first four milestones of `docs/bonsaiscratch-plan-v1.1.md`, stopping before full decoder integration until the measured M3 decision. The original plan governs every numerical detail, acceptance criterion, and later milestone. Work is one branch and, when a remote exists, one PR. Never claim target hardware performance from another machine.

## Product Contract

### Summary

Build reproducible native reference tooling, lossless model packaging, an I/O-only TurboWarp extension, visible Scratch numerical primitives, and compiled-runtime kernel benchmarks. Make an evidence-based proceed/redesign decision.

### Problem Frame

The pinned model and native runtime were audited, but Scratch inference, transport throughput, and browser memory have not been validated.

### Requirements

- R1. Lock model `prism-ml/Ternary-Bonsai-2-2B-gguf` revision `a6032a5089ac27e96be3a80c7b42d1ad0bd89887`, file `Ternary-Bonsai-2-2B-PQ2_0.gguf`, length 517312992, SHA-256 `d5f01a4ecaf49239653308ee732ad1a0c5f802c0ddb997549fc7b694c8b45928`; native source `e3de715222b2667c54a267441159a5729f11155e`, release `prism-b10777-e3de715`; audited VM `0cb82afe62506f3d9f2ff849c95593f0473b5d2c`. Pin the deployed application/compiler separately.
- R2. Inventory all 320 tensors, their shapes, orientation, offsets and payloads; enforce 151 PQ2_0 and 169 F32, 24 layers with the 3 DeltaNet/1 attention schedule, 2048 hidden width, 6144 MLP width, 248320 vocabulary, tied output, EOS 248046, tokenizer metadata, and Hadamard-v2 block-512 signs and associations.
- R3. Produce deterministic native explicit-token fixtures and capture operator inputs, transformed activations, Q8_K payloads, layer/state checkpoints and logits with recorded native build, backend, CPU features, seed, one thread, F32 caches, flash attention off, requested/effective context and greedy decoding.
- R4. Preserve exact PQ2_0 codes and FP16 scale bits, F32 payload bytes and transform/tokenizer metadata. Pack raw and BS-LUT4 losslessly as 34-byte groups, reject code 3, reverse-check every group, and retain checksums, format version, offsets, group mapping and dimensions.
- R5. The deployed extension only selects files, reads metadata and copies bounded storage values into existing Scratch lists. Scratch performs every neural operation, including FP16/F32 handling, activation quantization, LUT construction, matvec, reductions and text handling. Prove resident tile computation with the extension disabled.
- R6. Implement a reproducible visible-block SB3 and actual compiled TurboWarp tests for FP16 zeros/subnormals/nonfinite rejection, native Q8_K 256-element signed-maximum grouping, native rounding and F32 boundaries, scalar ternary and dense dot, RMS/L2 norm, SiLU, softmax, trig behavior, and case-sensitive UTF-16/UTF-8 text handling. Quantized payloads and lookup indices are exact; floating operators use documented, measured per-operator tolerances.
- R7. Transport uses whole groups, `floor(budget/34)` for 64 KiB, 256 KiB, 1 MiB and 4 MiB nominal budgets; report actual counts and bytes. Correctly handle partial rows, EOF, short reads, invalid offsets, cancellation, stale requests and reusable buffers without exposing partially copied data. Non-warp driver yields outside bounded warp tiles.
- R8. Compare scalar, raw LUT4, and BS-LUT4 using identical quantized activations and scale rules; exercise real width-2048 and width-6144 rows and representative tensors, full 248320-row vocabulary head, 1M/10M contribution microbenchmarks, sustained streaming, and split timing components.
- R9. Report conditions, raw repeated timings, warm/cold and cache state, memory peaks, worst uninterrupted interval, cancellation latency, shape-weighted decode and 1/16/128 prompt prefill projections including uncertainty. Separate projections from measurements and decode from prefill. A projected decode over five minutes/token pauses full integration. Target goals are 50 ms tiles, 250 ms cancellation, under 6 GB memory; repeated peaks above 8 GB trigger redesign.

### Scope Boundaries

M4–M8, full decoder integration, production tokenizer, vision, training, long context and sampling are not part of the first implementation objective. Target is Windows/Linux on Ryzen 7 Pro-class hardware with 16 GB; the user explicitly authorized current non-target hardware for this run. Current-host results cannot establish target performance.

## Planning Contract

### Key Technical Decisions

- KTD1. The pinned CPU PQ2_0 × Q8_K path is the oracle; unrestricted float is diagnostic only. Signed abs-max and rounding must follow pinned source, not Scratch `round`. Source: `ggml/src/ggml-quants.c` and `ggml/src/ggml-cpu/ggml-cpu.c` at the runtime pin.
- KTD2. Raw PQ2 bytes and BS-LUT4 carry equal storage size; select the complete path by measured performance, not lookup domain size.
- KTD3. The loader is unsandboxed for direct list access, storage-only, validates list identity and ownership, and publishes only completed requests. Compiled tests must assert the VM actually compiled relevant threads because the compiler may fall back.
- KTD4. Golden fixtures and model package generators are reproducible; full model and large fixture sets remain outside Git.

### Assumptions

- This empty repository has no existing language, build, or package conventions. Offline tooling may use Python standard library; generated projects and extension use the pinned TurboWarp VM and ordinary Scratch blocks.
- On the available Intel Xeon, benchmark results are host-specific, not target results; missing target-machine checks remain explicit in the decision.

## Implementation Units

### U1. Provenance and native reference

**Goal:** Make model identity, tensor inventory and native fixtures reproducible.

**Requirements:** R1–R3. **Dependencies:** none.

**Files:** `model-lock.json`, `runtime-lock.json`, `tools/inspect_gguf.py`, `tools/reference/`, `fixtures/`, `tests/test_inspection.py`, `docs/numerical-contract.md`, `README.md`.

**Approach:** Parse locked GGUF metadata and inventory without changing payloads. Verify size/hash before using a model. Build the pinned native runtime with explicit recorded flags; instrument graph and CPU quantization staging where needed. Export deterministic token-ID, operator and state goldens with schemas and error metrics.

**Test scenarios:** Wrong size/hash is rejected; first/last tensor boundaries and all 320 tensors are verified; explicit token fixture repeated twice yields identical discrete results; requested/effective context and CPU flags are recorded; F32 exceptions and tokenizer flags survive inventory.

**Verification:** Independent reproducible inventory and deterministic discrete fixtures match the locked native build; numerical baselines and tolerance methodology documented.

### U2. Lossless model packaging

**Goal:** Produce and validate both bounded model representations.

**Requirements:** R1, R2, R4. **Dependencies:** U1.

**Files:** `tools/package_model.py`, `tests/test_codec.py`, `docs/model-format.md`, compact real-model fixtures.

**Approach:** Preserve 2-byte scale and 32-byte weight groups, unchanged F32 data, signs and tokenizer data. Record offsets/checksums and reverse each group to original source before accepting a package. Keep large payloads outside Git.

**Test scenarios:** All 81 four-trit combinations round-trip; unexpected q=3 is rejected; first/last groups and actual tensor tile round-trip bit-exactly; F32 NaN payload bits are copied unchanged; malformed and truncated files fail without yielding a valid package.

**Verification:** Whole actual model group and scale reverse-check passes for both encodings with measured package checksum.

### U3. Storage-only loader and bounded transport

**Goal:** Load valid package ranges into existing Scratch lists safely.

**Requirements:** R4, R5, R7. **Dependencies:** U2.

**Files:** `extension/loader.js`, `tests/transport/`, `docs/extension.md`.

**Approach:** Select package locally, expose metadata and group-aligned ternary/dense range requests; bind target/list identity before asynchronous I/O and reject stale/cancelled/invalid/short reads. After validation, copy each bounded chunk without yielding to Scratch scripts and publish counts only after the copy; retain the list objects expected by the compiler. Shrink the chunk when copy duration exceeds the responsiveness target. No model math or runtime recoding.

**Test scenarios:** First and last group, split row and each chunk budget; exact EOF and invalid or short ranges; cancellation and stale completion; buffer/list reuse with no stale or overwritten data; dense F32 and FP16 storage bit preservation; compiled Scratch reads resident data after extension disable.

**Verification:** Actual compiled VM transport tests and independent file-read/list-copy timings; source inspection confirms storage-only behavior.

### U4. Visible numerical and text primitives

**Goal:** Execute native-compatible primitive operations in visible Scratch blocks.

**Requirements:** R3, R5, R6, R7. **Dependencies:** U1, U3.

**Files:** `scratch/`, `tools/generate_project.py`, `tests/scratch/`, `dist/primitives.sb3`, `docs/numerical-contract.md`.

**Approach:** Generate reproducible inspectable block graphs and a non-warp bounded driver. Quantize in 256-element Q8_K groups with explicit F32 operation boundaries; implement FP16 decoding and error status in blocks. Compare native discrete fixtures exactly. Independently justify and freeze operator-specific floating tolerances before acceptance testing, then validate against separate native fixtures. Handle text case and UTF-16 surrogate/UTF-8 conversion in Scratch.

**Test scenarios:** Positive and negative signed extrema and ties; zero groups; exact native rounding boundaries; FP16 zero, signed zero, subnormals and nonfinite rejection; cancellation-heavy real model rows; dense, normalization, SiLU, softmax and trig against native; text case, surrogate pairs and UTF-8 boundaries. Compiled threads are asserted compiled and work completes with loader disabled after tile load.

**Verification:** All primitive goldens pass actual compiled pinned TurboWarp, with discrete exactness and published operator error statistics.

### U5. Kernels, streaming and decision

**Goal:** Measure complete paths and decide viability without asserting target-hardware results.

**Requirements:** R5–R9. **Dependencies:** U1–U4.

**Files:** `benchmarks/`, `scratch/`, `tools/generate_benchmark.py`, `tests/scratch/`, `dist/benchmark.sb3`, `docs/m3-decision.md`.

**Approach:** Use identical real transformed/quantized activation fixtures for scalar, raw and BS-LUT4 kernels; preserve partial row sums and apply each scale once. Separate read, list-copy, quantization, setup, arithmetic and scheduling. Exercise the complete head and representative tensors on the actual compiled VM. Record exact CPU/OS/runtime/compiler, repetitions, settings and error, with uncertainty in full-token and prefill estimates.

**Test scenarios:** Split tiles produce the same native-compatible result as unchunked execution; each kernel agrees on real inputs; full vocabulary head visits every row; 1M/10M workloads and repeated streams report valid counts, bounded memory and measured cancellation; extension-disabled resident tiles remain correct.

**Verification:** Raw results, measurements and a proceed/redesign decision exist; over-five-minute decode projections pause full integration; prefill and target-hardware limitations are stated separately.

## Verification Contract

Use deterministic unit and integration tests for parser, codec and transport, real native fixtures, and generated SB3 execution on the actual pinned compiled VM. Verify the compiler did not fall back, the extension does no model math, and the resident tile works without it. Record real hardware, software, compiler settings, raw timings, memory and cancellation. Native output is not a substitute for Scratch execution, and host-specific results are not target results.

## Definition of Done

All M0–M3 exit conditions in `docs/bonsaiscratch-plan-v1.1.md` have evidence or an explicit gate failure. No full decoder integration before the M3 decision. Every release artifact is reproducible; large artifacts stay outside Git; remove throwaway code. If target hardware remains unavailable, report that limitation and do not claim target validation.
