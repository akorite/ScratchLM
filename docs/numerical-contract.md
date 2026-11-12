# Numerical contract

The canonical reference is PrismML `llama.cpp` commit `e3de715222b2667c54a267441159a5729f11155e`, CPU backend, one thread, with PQ2_0 weights and Q8_K activations. The model and runtime locks identify the inputs. CPU feature selection and build flags also belong in every fixture manifest: the x86 SIMD dot kernel can accumulate floating-point products in a different order from the scalar kernel.

## Stored weights

Each PQ2_0 group contains a little-endian FP16 scale followed by 32 bytes. Each byte holds four consecutive two-bit codes, least-significant pair first. A code represents `code - 1` times the scale. The model audit reports codes 0, 1 and 2; the packer must reject code 3 rather than silently discard information. Preserve the original FP16 scale bits and the bytes of F32 tensors.

Scratch must decode FP16 scales, including signed zero and subnormals. An exponent of 31 represents a nonfinite value and must produce an explicit unsupported-scale status rather than relying on Scratch's NaN coercions.

## Q8_K activations

The reference quantizer processes groups of 256 F32 values. It scans in order and keeps the signed value with strictly largest absolute magnitude; equal-magnitude later values do not replace it. An all-zero group has zero scale and zero integer payload.

For a nonzero group, the reference computes `iscale = -127.f / max`, rounds each F32 product `iscale * x[j]` with `nearest_int`, caps it at 127, and stores the scale as the F32 reciprocal `1.f / iscale`. The pinned `nearest_int` uses the F32 magic-bias operation; with the standard round-to-nearest-even floating-point environment this is ties-to-even. Scratch's `round` block is not equivalent. The stored integer payload and activation scale bits are exact acceptance checks.

Each activation group spans two PQ2_0 groups. The generic native kernel accumulates integer products within each 128-weight group, then applies that group's FP16 scale and the activation scale. The SIMD kernel's accumulation order may differ. Float comparisons require separately calibrated, frozen per-operator tolerances and explicit absolute, relative and RMS errors; a matching top-1 token alone does not establish numerical equivalence.

## Boundaries and validation

The extension may only select files, read metadata and copy storage values into Scratch lists. It must not quantize, decode neural weights, construct activation tables or perform model arithmetic. Native tools may perform these operations only offline to produce reference fixtures and analyze results. After loading a resident tile, run its arithmetic in Scratch with the loader disabled.

Scratch uses binary64 Numbers for ordinary arithmetic, so an F32 value in a list is not by itself evidence that intermediate calculations round as F32. Apply and verify F32 rounding at the contract's operation boundaries. Record compiler build and settings and assert that the tested Scratch scripts actually compiled.

The original plan is `docs/bonsaiscratch-plan-v1.1.md`. Per-operator tolerances and compiled-runtime measurements have not yet been established.
