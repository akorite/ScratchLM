# Remaining levers for the Scratch decoder

Every entry is a candidate optimization found in the literature or in the
pinned runtime, with the measurement that decides it on this host. The steady
state entering the pass: one decode step 33.7 s, first token 408 s, 17.9 ns per
weight streamed, 1.88 G weights per step, 537 MB package over the storage
extension (15.9 MB/s).

Verdicts are *measured* when a harness here produced the number and *inferred*
when they follow from the cost model and published data.

## Wall: skipped zero weights (ternary sparsity)

Ternary weights are 32.6% zero in this package (BitNet-class models measure
29.7–51.5%, arXiv 2609.16338). Skipping them costs more than it saves: the inner
loop spends about 18.8 ns per four-weight group, a skip test costs a presence
read, a compare and a branch (~11 ns), so break-even needs 58.5% of *groups*
entirely zero. With independent zeros that means z⁴ ≥ 0.585, i.e. z ≈ 86–89%.
No published ternary checkpoint reaches that, and vendors agree: BITCOS needs
z > 0.375 just for storage parity and shows no gain on bandwidth-rich parts;
T-MAC absorbs zeros into the lookup at zero marginal cost and never branches;
bitnet.cpp ships dense stream formats. Zeros already pay here as *bytes saved in
the package*, not blocks saved in the interpreter, and the step is
interpreter-bound (15.9 MB/s of a 537 MB package), so no traffic reduction
reaches the critical path.

## Wall: speculative decoding and early exit

Verification of p draft tokens re-reads the weight stream once and costs about
1.56× a step at p = 2 with this split (22.9 s projections, 8.0 s recurrence,
3.5 s head), so break-even needs an acceptance rate above 0.70 and +2% needs
0.73. Medusa needs trained heads (d·V = 508.6 M, 27% of the package); EAGLE
needs a trained layer (2.2%); self-speculation re-streams a quarter of the
weights for a 0.96× result; prompt lookup has no datastore in a 10-token
prompt. Early exit and mixture-of-depths genuinely skip layers but change the
output, which the numerical contract forbids. Exactness is free here only where
the speedup is not.

## Wall: exact pruning of the 248,320-way head

Chunked branch-and-bound over the head's 8 chunks of 256 dimensions cannot pay:
the winning logit is around 4.9σ with σ = √nnz, while the remaining upper bound
after consuming a fraction f is about (1−f)·0.8d, so nothing can be pruned
until roughly 84% of the dimensions are consumed. Measured on the real shapes
with the package's own 32.6% zero rate: 93% of the multiply-accumulates still
execute, saving about 0.2 s (0.6% of the step) before the ~15.6 M extra blocks
and a mandatory chunk-major repack of the 135 MB head. Tighter side data does
not change it: the two-count signed bound equals the one-count bound to 0.1%,
and an L2-product bound is worse than doing nothing.

## Wall: scalar GEMV micro-optimization

Every native trick targets an effect this runtime does not have. Multiple
accumulators break a latency chain this loop does not have (the body issues ~50×
more work per iteration than the chain's latency) and each extra accumulator
adds a variable load and store per unroll slot. Software pipelining and
prefetching are dispatcher wins, and the TurboWarp compiler has already inlined
blocks into one JS function with hoisted variable references. Larger LUT groups
(g = 8 means 3⁸ = 6,561 entries) need about 5,500 rows to amortise their build
against a 448-row tile. Cross-matrix fusion is a *layout* decision that the
package already takes (`.attn_qkv.weight` is one tensor); packed bytes of
distinct matrices are disjoint, so only the activation LUT is shareable, worth
about 0.6%.

## Wall: straight-line projection bodies (KTD1)

The plan's floor of "one packed read, one lookup, one add per group byte" assumed
the two cursor updates were pure overhead. They are not the only per-byte work:
with a runtime base index, every read needs its own `add`, so two bytes per
accumulator write still costs seven blocks per byte against five now, and the
pairing of two lookup chains *re-associates the group sum*, which the numerical
contract forbids. Measured on the real step (two alternating pairs, 448-row
tile): 36.1 s median before, 43.2 s after, inside a 16% spread between identical
builds. Reverted.

## Open: keep the hot cursors type-provably numeric

The compiled VM lowers `item of` to a raw array read only while the index is
provably numeric, and it emits `(t = x) === t ? t : 0` guards around every
variable read it cannot prove. Two probes on the pinned compiler: `change cur by
243` emits one add with no guards, while `change cur by <variable>` emits two
guards; arithmetic-initialised cursors stay numeric, so the variable increment
is the only poison. The group body's LUT cursor advanced by `BS lut domain`, a
variable, so every byte paid two guards plus a guarded index.

Fix: pad every LUT chunk to 256 entries (a literal, codec-independent stride)
and advance the cursor by `256`. Memory grows by the padding — 26 × 13 unused
slots per 128-weight group, about 5%. The stride is identical for the raw,
`bslut4` and `bslut5` codecs, so no per-codec kernel and no exit from the
bytecode cliff is needed.

## Wall: batched prompt tokens in one weight pass

Projecting K tokens through one weight read amortizes only the packed read
(0.44 of the pass), so the modelled ceiling was 1.3–1.5× on prompt time.

Built and measured instead. First measurement was invalid: modes 48/49 were
missing from the harness loader allowlist, so the stub nulled the package after
one request and the repeated stages timed failure paths. With modes 48/49/50 on
the allowlist, outputs verified (2048 numeric values per token) and a startup
baseline (mode 50, 2.16 s) subtracted:

| Run | Wall | Per unit |
|---|---|---|
| four single stages | 4.66 s | 0.63 s per token |
| four paired stages (8 tokens) | 15.3 s | 1.64 s per token |

The paired path is **0.38×**: 2.6× slower per token. A two-token pass costs
3.3 s against 1.3 s for two single passes. The packed read is only 0.44 of the
pass while the second accumulator adds a lookup and an add per group, the pair
kernel builds a second LUT and q8 stream per site, and the per-token staging
copies lists around every projection. Batching prefill is a regression on this
runtime; the paired stage stays a verified primitive that the decoder does not
call.

