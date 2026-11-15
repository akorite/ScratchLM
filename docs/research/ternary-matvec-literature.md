# Ternary / 2-bit matvec literature → Scratch mapping

Environment model assumed throughout: TurboWarp warp mode; every Scratch block ≈
one interpreted/compiled-JS step; `item n of list` is an O(1) array read; list
writes cost at least a read plus bounds/store; arithmetic on binary64 is real but
cheap — **the meter runs on block steps and list traffic, not FLOPs**. So the
question for every technique below is not "does it reduce multiplies" (multiply
is already free-ish) but "does it reduce *block steps and list accesses per
weight*".

Measured baseline to beat: ~3–4 block steps per weight (read w, read a, combine,
loop) → a 2560×2560 matvec ≈ 6.5M weights ≈ ~20–26M steps.

---

## 1. bitnet.cpp TL1 / TL2 kernels — the direct ancestor (arXiv 2410.16144)

**Trick.** Precompute, for each activation pair/triple, the dot product with
*every possible weight pattern*, then scan an index stream:
- TL1: 2 trits → 4-bit index into a 3²=9-entry table. Per 2 weights: 1 lookup + 1 add.
- TL2: 3 trits → sign bit + 4-bit index (27 patterns, mirror-consolidated to 13
  positive-sign entries + negation). Per 3 weights: 1 lookup + 1 add (+ sign fixup).
- I2_S: plain 2-bit unpack-then-multiply baseline (the "do nothing clever" kernel).

**Why fast.** Multiplies vanish entirely; per-group work is a single table read
and an accumulate. TL2 is recommended precisely for memory/bandwidth-limited
environments — it trades compute for fewer weight-stream reads.

**Scratch transfer: ★★★★★ — this is THE template, and we can push it harder.**
- The table is a Scratch list; lookup is `item idx of LUT` — O(1), cheap.
- TL1/TL2 stop at g=2–3 because tables live in *registers* (int16 LUT, SIMD
  gather). Scratch has no register pressure: table lives in a list either way,
  so g=4 (81 entries), g=5 (243), g=6 (729) are all equally addressable. The
  constraint shifts from register budget to *table-build cost* — see §8.
- Mirror consolidation (TL2's sign-bit trick) is **not worth it in Scratch**:
  halving a ~124k-entry build saves ~60k writes once, but costs a compare +
  conditional negate *per lookup* (×M·K/g ≈ 1.3M) forever. The math: negated
  pattern p has index (3^g − 1 − p) under {0,1,2} trit encoding, so it *could*
  be done — verdict: skip unless builds dominate.
- bitnet.cpp proves the whole thing is numerically lossless vs FP32 GEMV — the
  LUT entries are exact sums of ±a_k, no approximation anywhere.

## 2. T-MAC (arXiv 2407.00088) — generalization and the crucial loop-order insight

**Trick.** Any n-bit weight matrix = n one-bit matrices (A·W = Σᵢ 2ⁱ A·Wᵢ).
Precompute activation × all 2^g bit patterns per group → LUT; weights become an
index stream; scan = lookup + accumulate per bit-plane.

**Why fast.** Eliminates dequantization *and* all multiplies; work scales
linearly in weight bit-width. 30 tok/s single-core for BitNet-3B on M2-Ultra.

**Scratch transfer: ★★★★ for its systems ideas, ★ for its bit-plane decomposition.**
- For *ternary* weights, skip the bit decomposition entirely: a g-trit group is
  natively a single index into a 3^g table — one lookup, one add, done. T-MAC's
  bit-plane approach would need 2 binary lookups + subtract per group; strictly
  worse than the native trit index (TL1/TL2's approach).
- **The transferable gold is "LUT-centric data layout": loop K first, not M.**
  The LUT is built once per activation vector and shared across all M output
  rows — T-MAC reorders axes and tiles M large specifically to amortize table
  build over maximum rows. In Scratch terms: build all K/g tables first, *then*
  scan rows. Same idea, same payoff — see cost model §8.
- Also free: offline weight permutation/interleaving (reorder weights so the
  index stream is sequential — irrelevant for us since index lists are already
  sequential) and table quantization (quantize LUT entries to int8 — pointless
  in binary64, skip).
- Key negative result they confirm: LUT-GEMM-class kernels *lost* to dequantize
  on GPUs because tables couldn't live in fast memory. The whole trick only
  pays when the table sits in your fastest storage. In Scratch, the list IS
  both storage and the only addressable memory, so this caveat is automatically
  satisfied — but it means `item of` latency is our single most important
  primitive (flag to VMInternals).

## 3. LUT-GEMM (arXiv 2206.09557, ICLR'24, Naver)

**Trick.** GPU kernel: for each activation tile, build a LUT of all partial dot
products against the quantized codebook in shared memory; replace the
dequant-GEMM with gather-accumulate. Origin of the "BCQ/binary-code" family.

**Why fast.** Removes the dequant pass; one LUT build amortized over many
weight rows of the tile.

**Scratch transfer: ★★★ conceptual confirmation.**
- Same skeleton as TL1/T-MAC: build table over activation groups, scan indices.
- Its cautionary note: LUT only wins if table access is the cheapest memory op
  in the system and the build is amortized over a big row-count. Both true for
  us (M ≥ 2560 everywhere; `item of` ≈ same cost as any list read).
- Supports non-uniform quantizers via index→value LUT — for us ternary is
  uniform {-1,0,1}·scale, so irrelevant; scale multiplies the accumulator once
  per row (negligible: M multiplies per matvec).

## 4. FLUTE (arXiv 2407.10960, EMNLP-Findings'24)

**Trick.** GPU LUT kernel for non-uniform (learned-codebook) quantized weights:
*offline* restructuring of the weight matrix so unpacking needs almost no bit
manipulation, plus LUT duplication across shared-memory banks to fight
bandwidth conflicts.

**Scratch transfer: ★★★ — one idea worth stealing.**
- "Restructure weights offline to make the hot loop dumb": our equivalent is
  precomputing the **index stream offline** — a list where item j of row m is
  the base-3 index of 5 trits. Then the inner loop needs zero unpacking math:
  `acc + item (item j of IDX_m) of LUT` ≈ 2 list reads + 1 add per 5 weights.
- Packing can happen either in the extension at file-decode time (defensible as
  file-format I/O, but worth flagging the "all math in Scratch" boundary) or in
  Scratch once at load (~one pass over the weights, amortized over all decode
  steps — seconds, once).
- Its GPU-specific parts (bank conflicts, register-heavy duplication) don't map.

## 5. llama.cpp TQ1_0 / TQ2_0 + the unsigned-offset trick (PR #8151, compilade)

**Trick.**
- TQ1_0: 5 trits/byte via base-3 packing (3⁵=243≤256), digits extracted with a
  multiply-by-reciprocal + mask trick instead of division. 1.6875 bpw.
- TQ2_0: 2 bits/trit, stored as unsigned {0,1,2}; shift+mask unpack. 2.0625 bpw.
- The `maddubs` trick: represent weights unsigned u = w+1, compute Σ a·u, then
  subtract Σa once per block. Turns signed ternary multiplies into unsigned×int8
  dot products for free.

**Why fast.** On AVX2, 2× faster than Q8_0; unpacking is pure shifts/masks.

**Scratch transfer: ★★½ — one genuinely useful algebraic detail.**
- Packing formats themselves are irrelevant (we don't have bytes or bit ops —
  `floor`/`mod` on binary64 is our only extraction, ~2 steps each).
- **The unsigned-offset trick is worth keeping in the pocket**: with trits
  encoded {0,1,2}, every group's LUT entry is a plain sum of a subset ± nothing;
  the −Σa correction can be folded into the LUT build (pattern (1,1,1,1,1) IS
  Σa over the group). In practice the balanced {−1,0,+1} LUT already handles
  this for free — the trick mainly tells us we can choose whichever trit→index
  encoding makes index computation cheapest, since the LUT absorbs the offset.
- Div-free trit extraction (multiply + mask ≈ floor(x/243)·stuff) matters only
  if we ever unpack indices in Scratch — avoid by precomputed index lists.

## 6. BITCOS (arXiv 2609.16338, Intel, Sept 2026) — exploit the zeros

**Trick.** Real ternary models are 29–51% *zeros* (BitNet-2B-4T: 42.2%, Bonsai
1.7B: 39.9%). Store a 1-bit presence bitmap + sign bits only for nonzeros →
2−z bits/weight, beating 5-trit packing for z>0.375. Kernels stream the bitmap
and only "do work" for nonzero lanes.

**Scratch transfer: ★★★½ — as a fallback/companion, not the main event.**
- The Scratch-native version is a CSR-style **index-list matvec**: for each row,
  a sorted list of positions where w=+1 and one where w=−1; inner loop is
  `acc + item (item i of pos_idx) of act` — ~3 steps per *nonzero* weight.
  At z≈0.42 that cuts steps to ~58% of dense (≈1.7×), with no table build at
  all. Zero setup cost, trivially visible in Scratch.
- But grouped LUT beats it whenever z < ~0.8: LUT at g=5 costs ~3 steps per 5
  weights = 0.6 steps/weight regardless of sparsity ≈ same as 40% sparsity CSR,
  and LUT wins outright if the model is denser. At z=0.42 they're roughly tied
  on steps; LUT also eliminates the position-list read (denser stream).
- Hybrid possibility: skip all-zero groups. With z=0.42, P(all-zero 5-group)
  = 0.42⁵ ≈ 1.3% — the branch costs more than it saves. Not worth it.
- If our actual Bonsai model is >60% zeros somewhere, CSR deserves a benchmark;
  otherwise LUT is the better default. (The bitmap layout itself doesn't map —
  no bit ops.)

## 7. Activation-quantization-free approaches — what "free" means here

The literature splits into two families:
- **W-only quantized, FP16 activations (mpGEMM):** T-MAC, LUT-GEMM, FLUTE,
  bitnet.cpp TL1/TL2. These are exactly our regime — Scratch activations are
  binary64, strictly *more* precise than their fp16, and every LUT entry is an
  exact binary64 sum. Zero accuracy concerns.
- **Both-quantized (activation LUT / codebook):** MADDNESS/aQML, LUT-NN,
  DeepGEMM (the CNN one, arXiv 1805.XXXX-era), BitNet a4.8 (4-bit activations
  + sparsification for int8 SIMD). **Do not transfer:** their entire gain comes
  from quantizing activations to feed int8/PQ hardware. In Scratch a binary64
  multiply is already cheap; there's no int8 fast-path to unlock. BitNet a4.8's
  activation sparsification (top-k on activations) is the one interesting bit —
  dynamic sparsity on the activation side could drop ~50% of LUT lookups *if*
  a cheap max-filter exists, but thresholding costs a full pass over K per
  matvec, eating the gain. Park it.

## 8. The Scratch-native synthesis: trit-group LUT matvec ("ScratchTL")

The literature converges on one scheme, which we can run at a bigger group size
than anyone on real hardware because our LUT has no register/SMEM budget:

**Setup (once per matvec, per token):**
- Weights pre-packed offline: index list IDX per row, item j = base-3 index of
  trits 5j..5j+4 (or 6j.. for g=6). Optionally several group-indices per
  binary64 slot (243⁶ = 2.06e14 < 2⁵³) — saves index reads at ~2 steps per
  floor/mod extraction; marginal, measure first.
- LUT build, shared across ALL rows: flat list of (K/g)·3^g binary64 entries.
  Build by DP with **zero multiplies**: LUT[p] = LUT[⌊p/3⌋] + c(p mod 3)·a_last,
  c∈{−1,0,+1} → one read, one conditional add, one write per entry.

**Scan:** for each row m (accumulator in a variable, NOT a list slot —
keep writes out of the inner loop):
  for each group j: acc += item (j·3^g + item j of IDX_m) of LUT
→ ~3 block steps per g weights; one list write per row total.

**Cost model** (units = block steps; dense baseline ≈ 3–4·M·K):

| g | 3^g | build ≈ 3·(K/g)·3^g | scan ≈ 3·M·K/g | total, M=K=2560 | speedup vs ~4·M·K dense |
|---|-----|---------------------|----------------|------------------|--------------------------|
| 4 | 81  | ~155k               | ~4.9M          | ~5.1M            | ~2.0× |
| 5 | 243 | ~370k               | ~3.9M          | ~4.3M            | ~2.4–4.5× |
| 6 | 729 | ~930k               | ~3.3M          | ~4.2M            | ~2.5–4.5× |
| 7 | 2187| ~2.4M               | ~2.8M          | ~5.2M            | worse |

For tall/wide FFN matvecs (M·K ≈ 17.7M, e.g. 6912×2560): g=6 → ~9.8M steps vs
~70M dense ≈ **5–7×**. Rule: g=5 for near-square, g=6 when M/K ≥ ~2.5; LUT
build is the same currency as the scan, so the bigger M is relative to K, the
bigger g pays. List memory at g=6, K=2560: 427×729 ≈ 311k entries — fits in one
TurboWarp list (no practical cap), though splitting LUTs per k-block keeps them
smaller if list-index cost grows with length (VMInternals to confirm `item of`
is truly O(1) at 300k+ entries).

**Why this wins on our specific bottleneck:** every per-weight list touch in
the dense loop becomes one LUT *read* amortized over g weights, and every
accumulator write becomes one variable update. List writes drop from O(M·K) to
O(K·3^g/g + M); block steps drop by ~g× on the scan.

**Runner-up (zero-setup option):** CSR index-list matvec (§6) — ~1.7× fewer
steps at 42% zeros, no build. Worth a head-to-head microbenchmark because it's
30 lines of Scratch and needs no offline repacking.

## 9. Summary ranking

| Technique | Core trick | Scratch fit |
|---|---|---|
| bitnet.cpp TL1/TL2 | 3^g-entry activation LUT + index scan | ★★★★★ template; push g to 5–6 |
| T-MAC | bit-plane LUT; K-first layout; amortize build | ★★★★ steal loop order, skip bit-planes |
| LUT-GEMM | shared-mem LUT mpGEMM | ★★★ confirms + warns: table must be cheap to read |
| FLUTE | offline weight restructuring | ★★★ precomputed index lists |
| llama.cpp TQ1_0/TQ2_0 | 5-trits/byte; unsigned+offset | ★★ packing n/a; offset trick minor |
| BITCOS / sparse | presence bitmap; CSR gather-add | ★★½ ~1.7× at 42% zeros, no build |
| MADDNESS/LUT-NN/a4.8 | quantize activations too | ★ nothing to gain in binary64 |

## Sources
- T-MAC: arXiv:2407.00088, github.com/microsoft/T-MAC
- bitnet.cpp kernels (I2_S/TL1/TL2): arXiv:2410.16144, github.com/microsoft/BitNet
- LUT-GEMM: arXiv:2206.09557 (ICLR 2024)
- FLUTE: arXiv:2407.10960 (EMNLP Findings 2024)
- TQ1_0/TQ2_0: llama.cpp PR #8151; compilade.net/blog/ternary-packing;
  ggml-org/llama.cpp Discussion #17393
- BITCOS: arXiv:2609.16338 (Intel, zero-density measurements of 29 ternary models)
- BitNet b1.58: arXiv:2402.17764; BitNet a4.8: arXiv:2411.04965
- MADDNESS / LUT-NN / DeepGEMM lineage: cited in T-MAC §2.4

## Open questions for VMInternals / ScratchSpeedTech
1. Is `item n of list` flat O(1) at 100k–300k+ entries, and what's its step cost
   vs a variable read? (Determines whether flat mega-LUT or per-k-block LUTs.)
2. Cost asymmetry read vs write vs `add to list`/`replace item` in warp mode —
   LUT build is write-heavy; if writes are ~2× reads, prefer g=5 over 6.
3. `letter i of string` + `unicode of` vs `item of` for the index stream:
   243 fits a byte; two indices (59049) fit one UTF-16 code unit. If string
   indexing is faster than list indexing, pack indices as chars.
4. Whether an `if` in the warp-compiled inner loop is cheap enough to make
   mirror-consolidated (half-size) LUTs or zero-group skips viable.
