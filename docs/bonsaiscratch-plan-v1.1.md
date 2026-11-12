# BonsaiScratch: revised implementation plan

**Version:** 1.1, incorporating the technical audit of 12 November 2026
**Target:** TurboWarp on Windows/Linux; Ryzen 7 Pro-class hardware with 16 GB RAM. Exact CPU SKU, OS, browser, and TurboWarp build must be recorded before target benchmarks.
**Starting point:** The selected GGUF and native reference have been audited. Scratch inference, transport throughput, and browser memory remain unvalidated.
**First implementation objective:** Complete M0–M3 and make a measured decision before committing to the complete decoder.

## 1. Goal and acceptance result

Build a visible Scratch-block implementation of autoregressive inference using the actual PrismML Ternary Bonsai 2 2B weights. Deliver a TurboWarp-compatible `.sb3`, external model package, I/O-only loader extension, reproducible reference fixtures, benchmarks, and documentation.

For fixed token IDs and the locked model, the project must produce a validated next-token logits vector or demonstrably matching top-1/top-k predictions under a documented numerical contract. It must then generate successive tokens while preserving recurrent and attention state.

Initial scope is text-only, greedy decoding, and a maximum **128 total prompt-plus-generated tokens**. Vision, long-context support, training, and real-time chat are outside the first release. Expand context only after correctness and memory are measured.

Success can be a slow, correct research demonstration. Performance claims must distinguish measured results from projections.

## 2. Non-negotiable execution boundary

|Responsibility|Execution location|
|---|---|
|Download, inspect, verify, and losslessly repackage weights|Offline tooling|
|Generate reference fixtures and analyze test results|Offline native/Python tooling|
|Select local files, read metadata, copy bounded packed data into lists|Minimal TurboWarp extension|
|Decode ternary weights and FP16 scales; quantize activations|Scratch blocks|
|Construct activation LUTs; perform all matrix operations and reductions|Scratch blocks|
|Hadamard/sign transforms, norms, RoPE, nonlinearities, attention|Scratch blocks|
|Convolution, DeltaNet recurrence, KV updates, output head|Scratch blocks|
|Production tokenization, detokenization, token selection|Scratch blocks|

The deployed project must never call an external inference engine, server, native binary, GPU runtime, or JavaScript model-math routine. Generating `.sb3` block graphs offline is allowed; generated arithmetic must remain inspectable Scratch blocks.

The extension may read little-endian storage values and copy them, including FP16 words and F32 payloads. It must not recode ternary weights at runtime, build neural LUTs, transform activations, compute logits, or select tokens. Offline repacking may recode weights losslessly.

Review extension source and inspect its call paths. After a tile is loaded, disable the extension and demonstrate that Scratch computes that tile's contribution independently. Repeat for each kernel family. Future tiles still require I/O; the isolation test applies to resident data.

## 3. Locked model and architecture

Use this exact baseline:

|Item|Value|
|---|---|
|Model repository|`prism-ml/Ternary-Bonsai-2-2B-gguf`|
|Model revision|`a6032a5089ac27e96be3a80c7b42d1ad0bd89887`|
|File|`Ternary-Bonsai-2-2B-PQ2_0.gguf`|
|File bytes|517,312,992|
|SHA-256|`d5f01a4ecaf49239653308ee732ad1a0c5f802c0ddb997549fc7b694c8b45928`|
|Native runtime|PrismML `prism-b10777-e3de715`|
|Runtime commit|`e3de715222b2667c54a267441159a5729f11155e`|
|Audited TurboWarp VM source|`0cb82afe62506f3d9f2ff849c95593f0473b5d2c`|

Pin the actual TurboWarp application/compiler build separately; inspecting a VM commit does not establish which build a deployed website uses.

Architecture facts to enforce in generated manifests and blocks:

- 24 layers: three Gated DeltaNet layers followed by one full-attention layer, repeated six times.
- Hidden width 2,048; MLP width 6,144; vocabulary 248,320; tied embedding/output weights.
- Full attention: eight query heads and two KV heads, each dimension 256. Q projection has **4,096 outputs**, comprising 2,048 query and 2,048 gate channels. K and V each have 512 outputs.
- Linear attention: 16 key/value heads with dimension 128; convolution kernel length four.
- 320 tensors: 151 PQ2_0 and 169 F32. Preserve every F32 exception, including convolution, normalization, time-step, and SSM parameters.
- Hadamard version two, block size 512, normalized Sylvester transform, explicit signs for widths 2,048 and 6,144. Follow the actual tensor associations and inverse embedding path.
- EOS is **248046, `<|im_end|>`**. Pin the actual tokenizer, special-token flags, and chat template from the GGUF.

## 4. Numerical contract

### 4.1 Canonical baseline

Target the pinned **Prism CPU PQ2_0 × Q8_K path**. Quantize activations in Scratch using its 256-element grouping, signed abs-max scale selection, rounding, clamp behavior, and F32 operation boundaries. Each activation group spans two 128-weight PQ2 groups. Preserve the native order of transforms, activation quantization, integer accumulation, and scale application.

Loading F32 values into Scratch does not enforce F32 arithmetic: Scratch normally computes with binary64 Numbers. Implement explicit F32 rounding where the selected contract requires it, especially in activation quantization, and measure its cost. Do not substitute Scratch `round` for the native rounding rule.

Keep an unrestricted-floating-point dot product as a diagnostic oracle only. If the native-compatible approach proves impractical, changing the canonical arithmetic requires an explicit contract revision, new goldens, and measured comparison against native inference.

### 4.2 Validation layers

1. **Exact storage:** original ternary codes, FP16 scale bits, F32 payloads, tensor ordering, offsets, and lossless reverse recoding.
2. **Exact discrete arithmetic:** Q8_K integer payloads and LUT indexing against the pinned quantizer. Validate activation-scale bits at the selected F32 boundaries.
3. **Floating-point operators:** absolute/relative error, RMS error, and relevant norm or cosine metrics against native fixtures.
4. **Stateful inference:** per-layer hidden states, recurrent matrices, convolution history, KV contents, logits, and subsequent token IDs.

Calibrate and freeze per-operator tolerances before performance optimization. The audit's five-row native test already showed that omitting activation quantization exceeded the original proposed tolerance on four rows. A blanket tolerance cannot repair a different computation.

Document degree/radian handling for RoPE and Scratch's rounded trig blocks. Handle zero and subnormal FP16 scales explicitly. Reject unsupported nonfinite scale inputs with a status; do not rely on NaN propagation through Scratch's coercions.

For top-1 acceptance, record the reference top-1/top-2 margin and maximum logit error. A reference margin greater than twice the maximum error is sufficient to preserve top-1. Investigate near ties at logit level. A matching token alone is insufficient evidence of operator correctness.

## 5. Repository and artifacts

Proposed repository structure:

```text
bonsaiscratch/
  README.md
  model-lock.json
  runtime-lock.json
  tools/                 # inspection, repacking, reference fixtures, project generation
  extension/             # I/O-only loader source
  scratch/               # inspectable block sources / project generator inputs
  tests/                 # codec, transport, numerical and stateful checks
  fixtures/              # compact versioned fixtures and generation manifests
  benchmarks/            # harnesses, machine metadata, raw result schema
  docs/                  # numerical contract, formats, execution boundary, setup
  dist/                  # generated .sb3 and extension release artifacts
```

Keep full model files and large generated fixture sets outside ordinary Git history. Provide deterministic download/generation commands and checksums. Version the generators and compact correctness fixtures. Every released `.sb3` must be reproducible from its block sources and generator version.

The model package will contain a versioned manifest, packed ternary payload, unchanged F32 payloads, transform/sign data, and tokenizer assets. Record source and repacked checksums, endianness, tensor dimensions/orientation, tensor type, offsets, lengths, group map, codec, and special tokens.

## 6. Transport and scheduling design

### 6.1 Weight representation

PQ2_0 stores 128 weights as one FP16 scale and 32 two-bit-code bytes: **34 bytes per group**. A packed code decodes as `(q - 1) * scale`; the audited file contains only q = 0, 1, 2. Reject unexpected q = 3 during repacking.

Implement two equal-footprint candidates:

- Raw PQ2 bytes indexing a four-weight activation table.
- BS-LUT4, recoding four ternary digits into one base-3 byte in 0–80.

Both consume 34 bytes per group including the scale: **2.125 bits/weight**. BS-LUT4 reduces the lookup domain, not weight-file size. Select the faster complete path by measurement.

### 6.2 Loader API and invariants

Provide blocks for selecting a local package, reading indexed metadata, loading a ternary chunk, loading a dense chunk, and reporting status/valid counts. A ternary request identifies tensor, starting group, group count, destination lists, and request ID. It returns copied counts and errors. A dense request identifies an element range.

Copy packed bytes and original FP16 words into ordinary numeric Scratch lists. Storage-only F32 conversion is permitted. Retain the destination list object expected by the VM/compiler. No neural operation belongs in these blocks.

Use whole-group requests. For nominal budgets of 64 KiB, 256 KiB, 1 MiB, and 4 MiB, request `floor(budget / 34)` complete groups; record actual bytes. If a lower-level reader splits records, it must carry the remainder explicitly. Preserve partial row sums across tiles and apply each group's scale exactly once.

Specify EOF, invalid offsets, short reads, cancellation, stale requests, and buffer ownership. Verify returned counts before indexing: an invalid Scratch list access can silently become numeric zero. Reuse bounded buffers; add a second buffer only after measuring its memory and actual overlap benefit.

### 6.3 Responsive execution

Use a **non-warp driver** that calls a bounded warp tile, returns, waits outside warp, checks cancellation/progress, and continues. Compiled `wait (0)` inside warp is not a reliable yield at the audited pin.

Begin with a 50 ms target for uninterrupted compute/copy tiles and a 250 ms cancellation-response target on the tested machine. These are engineering goals to verify and tune. Record worst uninterrupted intervals, compiler settings, and warp-timer mode; do not assume one Scratch script uses every CPU core.

## 7. Implementation milestones

### M0 — Freeze provenance, arithmetic, and fixture generation

**Work**

- Turn the audited file identity and tensor inventory into reproducible lock/inspection commands.
- Freeze runtime source, binary/build hash, backend, threads, seed, cache types, and attention settings. Start CPU-only, one thread, F32 caches, flash attention off, greedy decoding.
- Record requested and effective context: the audited runner rounded a request of 128 to 256. Keep the application's initial logical limit at 128.
- Build an offline fixture harness accepting explicit token IDs. Export operator inputs, transformed activations, quantized activations, layer/state checkpoints, and logits from the pinned reference. Instrument a debug build where the public interface does not expose these values.
- Write the numerical contract and fixture/error schemas. Validate deterministic repeat runs.

**Deliverables:** lock files, full tensor manifest, reference harness, compact golden fixtures, numerical specification.

**Exit:** another developer can reproduce the same model identity, discrete fixtures, and documented native outputs. No unresolved tensor orientation or arithmetic-path ambiguity.

### M1 — Implement lossless packaging and I/O-only transport

**Work**

- Implement the narrow GGUF inspector and raw/BS-LUT4 packer; preserve source scale bits, F32 values, and transform metadata.
- Reverse-check every packed ternary group and scale against the source file.
- Implement the loader API with bounded, group-aligned transfers and reusable lists.
- Test first/last groups, split rows, EOF, short reads, cancellation, and buffer reuse.
- Measure file read time and list-copy time independently for each chunk budget.

**Deliverables:** package specification, packer, loader source, transport tests and initial measurements.

**Exit:** complete lossless round trip; exact valid lengths; no stale/overwritten data; source review confirms storage-only behavior.

### M2 — Build Scratch numerical primitives

**Work**

- Establish reproducible `.sb3` generation and a small test project containing visible arithmetic blocks.
- Implement FP16 decoding, selected F32 rounding, Q8_K quantization, scalar ternary dot, dense F32 dot, RMSNorm/L2 normalization, SiLU, softmax helpers, and required trig handling.
- Verify all-zero groups, signed extremes, rounding boundaries, zero scales, subnormals, cancellation-heavy sums, and actual model rows.
- Prove an early text-handling slice: exact case-sensitive keys, UTF-16/surrogate handling, and UTF-8 byte conversion within the execution boundary. Keep the full tokenizer for M7.

**Deliverables:** primitive test `.sb3`, block sources, results, initial tokenizer compatibility proof.

**Exit:** exact discrete fixtures and frozen numerical tolerances pass in the actual compiled TurboWarp runtime. No hidden arithmetic in the loader.

### M3 — Compare kernels and decide full-engine viability

**Work**

- Compare scalar unpack, raw-byte LUT4, and BS-LUT4 using identical Q8_K inputs and scale rules.
- Exercise real rows and full representative tensors at widths 2,048 and 6,144. Use supplied transformed input vectors until Scratch transforms are integrated in M4.
- Measure complete vocabulary-head scanning, not just a few rows.
- Run 1M/10M contribution microbenchmarks and sustained repeated streaming; separate LUT setup, quantization, transfer, arithmetic, scale handling, and scheduling.
- Build a shape-weighted full-token estimate and prefill estimates for 1, 16, and 128 prompt tokens. Include secondary work and uncertainty explicitly.

**Deliverables:** benchmark `.sb3`, raw results and conditions, kernel comparison, written proceed/redesign decision.

**Exit:** numerical gates pass; measured end-to-end transport/kernel results support the declared demonstration budget; memory and cancellation stay bounded. If the projected decode exceeds five minutes/token, pause full integration and redesign within the Scratch execution boundary. A long prefill estimate must be reported independently even if decode passes.

### M4 — Integrate transforms, embeddings, and the full output head

**Work**

- Implement normalized block-512 Hadamard and explicit signs in Scratch.
- Reproduce tensor-specific folded-weight paths, inverse embedding lookup, tied-output behavior, and final normalization.
- Integrate transform → activation quantization → matvec in the native order.
- Scan all 248,320 vocabulary rows and implement deterministic greedy selection with a documented tie rule.

**Exit:** complete real projections, embedding lookups, and output logits/top-k agree with native fixtures. Full-head latency and memory are measured. No vocabulary pruning in the correctness baseline.

### M5 — Validate one complete layer of each type

**Work**

- Implement full gated attention: Q/query-gate split, K/V, Q/K normalization, RoPE, causal masking, softmax, KV cache, gate, and output projection.
- Implement Gated DeltaNet: projection paths, causal convolution, normalization, gates/decay, recurrent update, readout, and output projection.
- Integrate each layer's normalization, MLP/SwiGLU, and residual paths.
- Compare complete hidden vectors and state/cache updates at positions 1, 2, 16, and 64, with reset/replay checks.

**Exit:** both layer types and their persistent state pass the numerical contract over successive tokens. Record state drift and uninterrupted tile durations.

### M6 — Assemble the 24-layer token-ID decoder

**Work**

- Dispatch the actual 18-linear/6-full layer schedule from the manifest.
- Carry recurrent, convolution, and KV state correctly across prefill and decode; implement explicit reset and the 128-token bound.
- Compare layer checkpoints, full logits, and several subsequent greedy tokens for fixed token-ID prompts.
- Measure actual time to first token, decode latency, and sustained memory at this stage; replace earlier projections with measurements.

**Exit:** a full next-token result and successive stateful decode steps pass reference checks. Report at least ten diverse fixed-token fixtures, including near-tie analysis where relevant.

### M7 — Complete text input and multi-token generation

**Work**

- Implement exact byte BPE: `qwen35` pretokenization, byte mapping, merge ordering, case-sensitive lookup, special-token behavior, and the locked chat template.
- Match token IDs for leading spaces, CRLF, case changes, combining marks, CJK, emoji, and literal special-token text.
- Implement streaming UTF-8 detokenization across token boundaries, EOS 248046 handling, token limits, reset, progress, and cancellation.
- Keep greedy decoding as the default. Any sampling feature must run in Scratch and have its own deterministic RNG tests.

**Exit:** text fixtures yield the exact reference token IDs; generated token bytes reconstruct the expected text; stateful generation obeys EOS and the context bound.

### M8 — Optimize, review, and release

**Work**

- Optimize only against the established correctness suite. Consider LUT reuse, scale caching, tile size, prefill reuse, and later codecs separately.
- Run an adversarial code review focused on hidden inference, numerical mismatches, transport bounds, state lifecycle, and unsupported performance claims.
- Verify clean offline launch and file reselection on the intended Windows/Linux environment, with the exact runtime and extension-loading procedure.
- Publish `.sb3`, block sources/generator, loader source/hash, package tooling, fixtures, raw benchmarks, setup instructions, and limitations. A TurboWarp Packager application is an optional separate artifact.

**Exit:** the distribution is reproducible, independently inspectable, and carries measured correctness, latency, and memory evidence.

## 8. Performance and memory gates

### 8.1 Workload baseline

|Work|Count|
|---|---:|
|PQ2 contributions per decode|1,880,096,768|
|LUT4 lookups per decode|470,024,192|
|Output-head LUT4 lookups|127,139,840|
|128-token prefill LUT4 lookups, final head only|44,016,336,896|
|Pattern-plus-scale list assignments per decode|484,712,448|
|Logical binary64 writes for those assignments|Approximately 3.61 GiB|

At hypothetical 10–20 million lookups/s, lookup work alone implies 23.5–47 seconds/decode and 36.7–73.4 minutes for a 128-token prefill. These are projections excluding transport and other work. Do not infer prompt usability from a decode-only gate.

For every run, log machine/build IDs, prompt length, cache state, warm/cold status, chunk size, compiler settings, repetitions, median/worst timing, file MB/s, list items/s, LUT setup, matvec time, memory peaks, and cancellation latency.

Use the original decode classifications as project policy: under 30 seconds/token is a stretch goal; 30–120 seconds a demonstration target; two–five minutes research-only; above five minutes a redesign trigger. Classify time to first token separately and publish it for each prompt length. No classification is a forecast.

### 8.2 Logical Number payloads

|Object|Approximate binary64 payload|
|---|---:|
|Recurrent state across 18 layers|36 MiB|
|KV cache at 128 tokens|6 MiB|
|Four-slot convolution cache|3.375 MiB|
|These persistent caches combined|45.375 MiB|
|LUT4 for width 2,048 / 6,144|0.316 / 0.949 MiB|
|Full logits vector|1.895 MiB|
|Optional decoded scale cache|112.063 MiB|
|One 4 MiB-budget materialized pattern/scale buffer|31.06 MiB|

These exclude list/container overhead, model data, spare array capacity, monitors, GC, and serialization. Measure actual browser-process memory separately. Target below 6 GB initially; stop and redesign repeated peaks above 8 GB on the 16 GB target. Hide large list monitors and never expand the whole model into Scratch lists.

Standard Scratch's guarded append/insert limit is 200,000 items; the audited TurboWarp paths do not impose that cap. Target TurboWarp explicitly. Large lists still require capacity testing.

## 9. First implementation work order

1. Create the repository skeleton and lock the audited model/runtime inputs.
2. Reproduce the inventory and native discrete/operator fixtures with documented commands.
3. Freeze the numerical contract and native-Q8_K rounding tests.
4. Package a tiny actual tensor tile in both raw PQ2 and BS-LUT4 forms; prove complete reverse equivalence.
5. Build the I/O-only loader and Scratch FP16/Q8_K/scalar-dot vertical slice.
6. Make that slice pass compiled TurboWarp goldens and the loader-disable isolation test.
7. Expand to representative tensors, both LUT kernels, and the full output-head benchmark.
8. Complete the M3 decision record before starting full layer integration.

If a gate fails, diagnose and correct that gate. A partial layer or kernel demonstration may still be published with its scope stated accurately. Preserve the all-Scratch inference boundary through every fallback.

## 10. Primary implementation references

- [Pinned model and model card](https://huggingface.co/prism-ml/Ternary-Bonsai-2-2B-gguf/blob/a6032a5089ac27e96be3a80c7b42d1ad0bd89887/README.md)
- [Pinned Prism runtime release](https://github.com/PrismML-Eng/llama.cpp/releases/tag/prism-b10777-e3de715)
- [Pinned quantization implementation](https://github.com/PrismML-Eng/llama.cpp/blob/e3de715222b2667c54a267441159a5729f11155e/ggml/src/ggml-quants.c)
- [Pinned CPU type traits](https://github.com/PrismML-Eng/llama.cpp/blob/e3de715222b2667c54a267441159a5729f11155e/ggml/src/ggml-cpu/ggml-cpu.c)
- [Pinned Qwen3.5 model graph](https://github.com/PrismML-Eng/llama.cpp/blob/e3de715222b2667c54a267441159a5729f11155e/src/models/qwen35.cpp)
- [Pinned Hadamard graph paths](https://github.com/PrismML-Eng/llama.cpp/blob/e3de715222b2667c54a267441159a5729f11155e/src/llama-graph.cpp)
- [Audited TurboWarp compiler](https://github.com/TurboWarp/scratch-vm/blob/0cb82afe62506f3d9f2ff849c95593f0473b5d2c/src/compiler/jsgen.js)
- [TurboWarp unsandboxed extensions](https://docs.turbowarp.org/development/extensions/unsandboxed)
- [TurboWarp warp timer](https://docs.turbowarp.org/warp-timer)

This plan incorporates the model, numerical, transport, and Scratch/TurboWarp audit performed in [the BonsaiScratch audit session](https://app.devin.ai/sessions/4885f4ab7b494c668ae333c2f963de28).
