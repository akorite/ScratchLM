# Numerical contract

The canonical reference is PrismML `llama.cpp` commit `e3de715222b2667c54a267441159a5729f11155e`, CPU backend, one thread, with PQ2_0 weights and Q8_K activations. The model and runtime locks identify the inputs. CPU feature selection and build flags also belong in every fixture manifest: the x86 SIMD dot kernel can accumulate floating-point products in a different order from the scalar kernel.

## Stored weights

Each PQ2_0 group contains a little-endian FP16 scale followed by 32 bytes. Each byte holds four consecutive two-bit codes, least-significant pair first. A code represents `code - 1` times the scale. The model audit reports codes 0, 1 and 2; the packer must reject code 3 rather than silently discard information. Preserve the original FP16 scale bits and the bytes of F32 tensors.

Scratch must decode FP16 scales, including signed zero and subnormals. An exponent of 31 represents a nonfinite value and must produce an explicit unsupported-scale status rather than relying on Scratch's NaN coercions.

## Q8_K activations

The reference quantizer processes groups of 256 F32 values. It scans in order and keeps the signed value with strictly largest absolute magnitude; equal-magnitude later values do not replace it. An all-zero group has zero scale and zero integer payload.

For a nonzero group, the reference computes `iscale = -127.f / max`, rounds each F32 product `iscale * x[j]` with `nearest_int`, caps it at 127, and stores the scale as the F32 reciprocal `1.f / iscale`. The pinned `nearest_int` uses the F32 magic-bias operation; with the standard round-to-nearest-even floating-point environment this is ties-to-even. Scratch's `round` block is not equivalent. The stored integer payload and activation scale bits are exact acceptance checks.

Each activation group spans two PQ2_0 groups. The generic native kernel accumulates integer products within each 128-weight group, then applies that group's FP16 scale and the activation scale. The SIMD kernel's accumulation order may differ. Float comparisons require separately calibrated, frozen per-operator tolerances and explicit absolute, relative and RMS errors; a matching top-1 token alone does not establish numerical equivalence.

The resident scalar and LUT diagnostics round each scale product, scaled integer subtotal, and accumulation to F32, matching the generic CPU kernel's operation boundaries. Five actual `blk.3.attn_q.weight` rows are checked exactly against that generic ordering and diagnostically against this host's SIMD reference. Raw-byte LUT4 uses 256 entries per four activations (4-trit stride); BS-LUT4 uses 81 (4-trit stride); BS-LUT5 uses 243 (5-trit stride, 25 full groups plus 1 padded 3-trit tail per 128-weight group). These checks do not establish a tolerance for other tensors or the SIMD path.

For the captured one-thread CPU build on the recorded Xeon host, the frozen *projection-specific* gates are: `blk.3.attn_q.weight`, maximum absolute error ≤2e-6 and RMS error ≤4e-7; `blk.0.ffn_down.weight`, maximum absolute error ≤2e-7 and RMS error ≤4e-8. These bounds were set from the complete token-1 calibration tensors in `benchmarks/results/tensors-xeon.json`, before checking the stateful token-321 holdout. Record maximum relative error and cosine separately; relative error is unstable near zero and is not used alone as a gate. These bounds are not a blanket tolerance for other tensors, operators, hosts, or runtime builds.

The primitive project's mode 10 accepts radians and converts to degrees before Scratch `sin`/`cos`; the audited compiler rounds those trig results to ten decimal places. Modes 4–8 implement diagnostic dense dot, RMS normalization with the locked model's F32 epsilon (approximately 1e-6), L2 normalization, SiLU, and softmax. Their tests use mathematical examples, not frozen native-operator tolerances.

Modes 12–13 implement the normalized block-512 Sylvester-Walsh-Hadamard transform with explicit ±1 signs (pre-rotation in mode 12 for forward folded weights, post-rotation in mode 13 for inverse embedding rows), applying F32 rounding to each normalized output element. Inputs must have positive length divisible by 512 and matching sign-vector length when signs are provided; partial blocks return `invalid dimensions`. Against the captured Xeon one-thread native pairs (`fixtures/native/m4/transforms/`), the frozen transform tolerances are maximum absolute error ≤1e-6 and RMS error ≤3e-7 for widths 2048 and 6144.

Modes 14–15 implement the inverse embedding row lookup from `token_embd.weight` across `raw`, `bslut4`, and `bslut5` packages (dequantize 16 PQ2_0 groups → block-512 Hadamard → post-rotation `prism.hadamard.signs.2048` multiply); mode 16 implements final `output_norm` (mode-5 RMS norm with the model's locked F32 epsilon followed by elementwise F32-rounded `output_norm.weight` multiply). Against the captured M4 goldens (`fixtures/native/m4/embedding/` and `head/`), the frozen tolerances are maximum absolute error ≤1e-6 and RMS error ≤3e-7 for embedding rows (tokens 1, 321, 248046) and maximum absolute error ≤1e-6 and RMS error ≤2e-7 for `output_norm`.

Mode 41 computes one complete Gated DeltaNet layer (layer 0): weighted RMS norm, the folded fused QKV projection, the depthwise kernel-4 convolution over the 3-tap circular buffer with SiLU, the joint 32-head L2 normalization of q and k, the dense alpha/beta projections, the state decay `exp(ssm_a * softplus(alpha + dt_bias))`, the delta-rule update of the transposed 128x128 per-head state, the gated RMS norm with its SiLU gate, the folded output projection, and the shared residual/post-norm/SwiGLU tail. The convolution is `out[c] = sum_t taps[t][c] * weight[t][c]` with the weight stored channel-major; the fused QKV and gate weights are folded, so both projections are preceded by the mode-12 rotation.

Against `fixtures/native/m5/deltanet/`, the compiled VM reproduces `l_out-0` with maximum absolute error 1.8e-7 at position 1, 8.9e-8 at position 2 and 3.6e-7 at position 16, and the written recurrent state with 3.0e-7 / 1.2e-7 / 1.2e-7. The frozen layer tolerance is maximum absolute error <=1e-6 and RMS <=1e-7. Positions after the first need the previous token's state and convolution taps: the taps come from the captured `conv_input` for that position, because the captured `conv_state_update` is a ring-buffer snapshot rather than a chronological history. `tests/scratch/test_deltanet.py` runs all of these with `BONSAI_LAYER_STATE` naming the captured state directory.

Mode 42 runs the locked 24-layer schedule (18 Gated DeltaNet, 6 full-attention), addressing each layer's recurrent state, convolution history and KV cells through per-layer base offsets, then the final `output_norm`, the inverse `token_embd` rotation applied as signs-then-Hadamard, the full 248320-row head scan and greedy selection.

Against the captured chain the layers reproduce `l_out-0`..`l_out-17` within 1e-4 but drift after that: `l_out-17` (after 18 layers) has maximum absolute error 1.8e-3 and `l_out-23` 5.6e-2. Each layer matches its native boundary to about 1e-7 when given the native input; the drift comes from Q8_K activation quantization flipping on inputs that differ in the last bits, so it is not removable by local arithmetic changes.

The decoded result still tracks the reference: for the first token of the locked prompt the top-5 tokens and their order are identical, the greedy token is 11 (native 11), the margin is 0.427 (native 0.438), and the full logit vector differs by maximum absolute error 0.11 and RMS 0.023. `tests/scratch/test_deltanet.py::test_decoder_predicts_the_native_next_token` asserts the scan length, the top-1 token and those logit bounds.

The native M5 fixture records layer 0 and layer 3 inputs and intermediates, all 24 layer outputs, and recurrent, convolution and strided KV state at positions 1, 2, 16 and 64. Full state and logits are external; the manifest records their checksums and state probes. Convolution history is captured after its copy operation, not from the pre-copy view. Reproduce into new directories with the pinned build:

```sh
python3 tools/reference/reference.py layer --model "$WORK/model.gguf" --build "$WORK/reference-build" \
  --output "$WORK/m5-fixture" --state-dir "$WORK/m5-state"
```

Compare the `sections` checksums and probes, rather than the manifest's machine-specific paths and build metadata. These fixtures do not establish Scratch layer tolerances.

Diagnostic mode 30 accepts a zero-based position followed by one or more 256-element heads and applies 64-dimensional partial RoPE with base 1e7. It rounds the recurrent angle and sine/cosine to F32, matching the native graph's angle progression; unrotated dimensions are copied unchanged. On this host, the captured layer-3 query vectors at positions 2, 16 and 64 agree within 1e-6 maximum absolute error. This is not a complete attention-layer tolerance.

Mode 40 computes one complete full-attention layer (layer 3) in Scratch: weighted RMS norm, Q with its 2,048 gate channels, per-head q/k norm, partial RoPE, causal softmax over the KV cache, sigmoid output gate, output projection, residual, post-attention norm, SwiGLU MLP and residual. Every PQ2 projection is Hadamard-folded in the package, so the layer applies the mode-12 forward rotation with `prism.hadamard.signs.2048` (and `.6144` before the down projection) before each matmul. The KV cache is appended token-major in Scratch, while the captured native V cache is `[head][dim][cell]`; convert when seeding.

Against `fixtures/native/m5/attention/`, the compiled VM reproduces `l_out-3` with maximum absolute error 3.6e-7 and RMS 3.9e-8 at position 1, 6.5e-5 / 1.4e-5 at position 2 (two-token softmax mixture), and 8.9e-8 / 2.0e-8 at position 64. These measurements freeze the layer tolerance at maximum absolute error <=1e-4 and RMS <=2e-5; the position-2 residual comes from accumulating the fp32 score dots in a different order than the native SIMD kernel. Reproduce with:

```sh
BONSAI_LAYER_STATE="$WORK/m5-state" python3 -m pytest -q tests/scratch/test_attention.py
```

Diagnostic mode 31 performs stable causal softmax over the requested prefix; modes 32 and 33 apply SiLU gating and residual addition; modes 34 and 35 apply per-head weighted RMS normalization with and without a SiLU gate. These modes reject malformed dimensions and round outputs to F32. Compiled-VM checks against the captured position-2 layer intermediates use a maximum absolute error of 1e-6; this is not an end-to-end layer tolerance.

The separate `--text` proof uses named costumes across 256 hidden sprites because Scratch string comparisons are case-insensitive. It does not switch the stage backdrop. Compiled-VM tests cover case, UTF-16 surrogate pairs and unpaired-surrogate replacement, UTF-8 boundaries, CJK, and combining marks. Each code unit invokes 256 searches via broadcast-and-wait. On the Xeon host, a single no-renderer run took 58.50 seconds to load and execute and peaked at 3,393,016 KiB RSS. Browser renderer memory, target hardware, and production tokenizer suitability remain unverified.

## Boundaries and validation

The extension may only select files, read metadata and copy storage values into Scratch lists. It must not quantize, decode neural weights, construct activation tables or perform model arithmetic. Native tools may perform these operations only offline to produce reference fixtures and analyze results. After loading a resident tile, run its arithmetic in Scratch with the loader disabled.

Scratch uses binary64 Numbers for ordinary arithmetic, so an F32 value in a list is not by itself evidence that intermediate calculations round as F32. Apply and verify F32 rounding at the contract's operation boundaries. Record compiler build and settings and assert that the tested Scratch scripts actually compiled.

The original plan is `docs/bonsaiscratch-plan-v1.1.md`. Per-operator tolerances for `blk.3.attn_q.weight`, `blk.0.ffn_down.weight`, Hadamard-v2 transforms, inverse embedding lookup, and `output_norm`, plus end-to-end head greedy selection on the Xeon host, are established above; 24-layer decoder integration and per-layer recurrent/KV tolerances remain unestablished.
