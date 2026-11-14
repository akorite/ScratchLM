# Numerical contract

The canonical reference is PrismML `llama.cpp` commit `e3de715222b2667c54a267441159a5729f11155e`, CPU backend, one thread, with PQ2_0 weights and Q8_K activations. The model and runtime locks identify the inputs. CPU feature selection and build flags also belong in every fixture manifest: the x86 SIMD dot kernel can accumulate floating-point products in a different order from the scalar kernel.

## Stored weights

Each PQ2_0 group contains a little-endian FP16 scale followed by 32 bytes. Each byte holds four consecutive two-bit codes, least-significant pair first. A code represents `code - 1` times the scale. The model audit reports codes 0, 1 and 2; the packer must reject code 3 rather than silently discard information. Preserve the original FP16 scale bits and the bytes of F32 tensors.

Scratch must decode FP16 scales, including signed zero and subnormals. An exponent of 31 represents a nonfinite value and must produce an explicit unsupported-scale status rather than relying on Scratch's NaN coercions.

## Q8_K activations

The reference quantizer processes groups of 256 F32 values. It scans in order and keeps the signed value with strictly largest absolute magnitude; equal-magnitude later values do not replace it. An all-zero group has zero scale and zero integer payload.

For a nonzero group, the reference computes `iscale = -127.f / max`, rounds each F32 product `iscale * x[j]` with `nearest_int`, caps it at 127, and stores the scale as the F32 reciprocal `1.f / iscale`. The pinned `nearest_int` uses the F32 magic-bias operation; with the standard round-to-nearest-even floating-point environment this is ties-to-even. Scratch's `round` block is not equivalent. The stored integer payload and activation scale bits are exact acceptance checks.

Each activation group spans two PQ2_0 groups. The generic native kernel accumulates integer products within each 128-weight group, then applies that group's FP16 scale and the activation scale. The SIMD kernel's accumulation order may differ. Float comparisons require separately calibrated, frozen per-operator tolerances and explicit absolute, relative and RMS errors; a matching top-1 token alone does not establish numerical equivalence.

The resident scalar and LUT4 diagnostics round each scale product, scaled integer subtotal, and accumulation to F32, matching the generic CPU kernel's operation boundaries. Five actual `blk.3.attn_q.weight` rows are checked exactly against that generic ordering and diagnostically against this host's SIMD reference. Raw-byte LUT4 uses 256 entries per four activations; BS-LUT4 uses 81. These checks do not establish a tolerance for other tensors or the SIMD path.

For the captured one-thread CPU build on the recorded Xeon host, the frozen *projection-specific* gates are: `blk.3.attn_q.weight`, maximum absolute error ≤2e-6 and RMS error ≤4e-7; `blk.0.ffn_down.weight`, maximum absolute error ≤2e-7 and RMS error ≤4e-8. These bounds were set from the complete token-1 calibration tensors in `benchmarks/results/tensors-xeon.json`, before checking the stateful token-321 holdout. Record maximum relative error and cosine separately; relative error is unstable near zero and is not used alone as a gate. These bounds are not a blanket tolerance for other tensors, operators, hosts, or runtime builds.

The primitive project's mode 10 accepts radians and converts to degrees before Scratch `sin`/`cos`; the audited compiler rounds those trig results to ten decimal places. Modes 4–8 implement diagnostic dense dot, RMS normalization with the locked model's F32 epsilon (approximately 1e-6), L2 normalization, SiLU, and softmax. Their tests use mathematical examples, not frozen native-operator tolerances.

Modes 12–13 implement the normalized block-512 Sylvester-Walsh-Hadamard transform with explicit ±1 signs (pre-rotation in mode 12 for forward folded weights, post-rotation in mode 13 for inverse embedding rows), applying F32 rounding to each normalized output element. Inputs must have positive length divisible by 512 and matching sign-vector length when signs are provided; partial blocks return `invalid dimensions`. Against the captured Xeon one-thread native pairs (`fixtures/native/m4/transforms/`), the frozen transform tolerances are maximum absolute error ≤1e-6 and RMS error ≤3e-7 for widths 2048 and 6144.

Modes 14–15 implement the inverse embedding row lookup from `token_embd.weight` across `raw`, `bslut4`, and `bslut5` packages (dequantize 16 PQ2_0 groups → block-512 Hadamard → post-rotation `prism.hadamard.signs.2048` multiply); mode 16 implements final `output_norm` (mode-5 RMS norm with the model's locked F32 epsilon followed by elementwise F32-rounded `output_norm.weight` multiply). Against the captured M4 goldens (`fixtures/native/m4/embedding/` and `head/`), the frozen tolerances are maximum absolute error ≤1e-6 and RMS error ≤3e-7 for embedding rows (tokens 1, 321, 248046) and maximum absolute error ≤1e-6 and RMS error ≤2e-7 for `output_norm`.

The separate `--text` proof uses named costumes across 256 hidden sprites because Scratch string comparisons are case-insensitive. It does not switch the stage backdrop. Compiled-VM tests cover case, UTF-16 surrogate pairs and unpaired-surrogate replacement, UTF-8 boundaries, CJK, and combining marks. Each code unit invokes 256 searches via broadcast-and-wait. On the Xeon host, a single no-renderer run took 58.50 seconds to load and execute and peaked at 3,393,016 KiB RSS. Browser renderer memory, target hardware, and production tokenizer suitability remain unverified.

## Boundaries and validation

The extension may only select files, read metadata and copy storage values into Scratch lists. It must not quantize, decode neural weights, construct activation tables or perform model arithmetic. Native tools may perform these operations only offline to produce reference fixtures and analyze results. After loading a resident tile, run its arithmetic in Scratch with the loader disabled.

Scratch uses binary64 Numbers for ordinary arithmetic, so an F32 value in a list is not by itself evidence that intermediate calculations round as F32. Apply and verify F32 rounding at the contract's operation boundaries. Record compiler build and settings and assert that the tested Scratch scripts actually compiled.

The original plan is `docs/bonsaiscratch-plan-v1.1.md`. Per-operator tolerances and full-token compiled-runtime measurements have not yet been established.
