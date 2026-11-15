# BonsaiScratch

BonsaiScratch investigates visible Scratch-block inference with the locked PrismML Ternary Bonsai 2 2B model in TurboWarp. The [implementation plan](docs/bonsaiscratch-plan-v1.1.md) defines the numerical contract, execution boundary, milestones and acceptance gates. Native code in this repository is for offline reference fixtures only; the deployed project must perform model math in Scratch blocks.

## Release set

| Artifact | Where | Notes |
|---|---|---|
| Prompt-to-text project | **`dist/bonsai-prompt-to-text.sb3`** | the runnable release: 256 lookup sprites, modes 22-24, ask-and-wait input, `Bonsai` speaker, storage extension declared |
| Primitive project | `dist/fp16-q8-scalar.sb3` | diagnostic modes 0-42, no text pipeline |
| Hashes | `fixtures/artifact-checksums.json` | both projects, checked by `tests/test_artifacts.py` |
| Storage extension | `extension/loader.js` | storage-only I/O; contains no model arithmetic |
| Model packages | not in Git | built from the locked GGUF with `tools/package_model.py` (see below) |
| Native expectations | `fixtures/native/` | layer, head, tokenizer and prompt fixtures with recorded tolerances |

Reproducing the project and the prompt path from a fresh checkout needs only the locked GGUF and the pinned TurboWarp VM checkout:

```sh
WORK="$HOME/.cache/bonsaiscratch"
python3 tools/inspect_gguf.py "$WORK/model.gguf" "$WORK/model-manifest.json"
TMPDIR="$WORK/build-tmp" python3 tools/package_model.py "$WORK/model.gguf" "$WORK/model-bslut5-tokenizer.bsp" --codec bslut5
python3 tools/generate_project.py dist/fp16-q8-scalar.sb3
python3 tools/generate_project.py dist/bonsai-prompt-to-text.sb3 --text --benchmark
```

Packages are large; the packer writes through `TMPDIR`, so point it at a disk with at least 1 GB free rather than a small tmpfs. `fixtures/package-checksums.json` records the byte counts and hashes of the three codecs, and the packer is deterministic: two builds into fresh files produce the same hash. The `bslut5` and `bslut4` packages embed the tokenizer's pair-merge table as `prism.bpe.*` tensors, which is what makes the Scratch tokenizer possible without any JavaScript tokenization.

Measured limits on the recorded Xeon host, Node VM without a renderer: **33.7 s per decode step** (448-row tiles, three runs, 17.9 ns per weight) and **408 s to first token for a 10-token prompt** (466 s for a 12-token one), down from 41.8 s and 723.8 s before the per-site projection kernels and the smaller tile, at **4.2 GiB peak RSS**, with a **128-token bound clamped to the remaining KV cells** (a ten-token prompt may generate 118). Each projection call site now has its own procedure because V8 refuses to optimise a compiled Scratch procedure past roughly 61 kB of bytecode; the layer bodies were over that and ran unoptimised. Typing a new prompt clears the token list, the KV cells, the convolution history and the recurrent state, so one session can serve several prompts.

Release tests: `tests/scratch/test_e2e_prompts.py` runs three typed prompts through the compiled VM and compares the rendered text, the templated tokens and the generated ids with `fixtures/native/m5/e2e-prompts.json`, which `tools/e2e_fixtures.py` captures from the pinned build; the same file checks the KV clamp and the second-prompt reset. `tests/scratch/test_tokenizer.py` covers the fixture classes and a character split across two tokens. The 16 GB Ryzen-class target and a browser renderer were not measured. Prompt prefill re-streams every weight per token, so long prompts scale linearly; batched prefill was built and measured at 0.85× per token, so it is not wired in (see `docs/levers.md`).

## Model identity and inventory

The model is not stored in Git. Download the locked file and check its SHA-256 and size before using it:

```sh
WORK="$HOME/.cache/bonsaiscratch"
mkdir -p "$WORK"
curl -fL 'https://huggingface.co/prism-ml/Ternary-Bonsai-2-2B-gguf/resolve/a6032a5089ac27e96be3a80c7b42d1ad0bd89887/Ternary-Bonsai-2-2B-PQ2_0.gguf' -o "$WORK/model.gguf"
python3 tools/inspect_gguf.py "$WORK/model.gguf" "$WORK/model-manifest.json"
```

The inspector checks `model-lock.json` by default, streams the GGUF metadata into JSON, and records each tensor's dimensions, type, offsets and size. The generated manifest contains the complete tokenizer metadata and can be large. `fixtures/tensor-manifest.json` is the compact tensor-only inventory of the locked model. `--no-lock` is intended for synthetic or explicitly untrusted files, not model verification.

Create `raw`, `bslut4`, or `bslut5` packages with the commands in [the model format](docs/model-format.md). Each build verifies every group and tensor against the source before publishing the package; checksums for the locked model are in `fixtures/package-checksums.json`.

## Offline native reference

The native harness builds the exact pinned PrismML source and records its binary hash and compiler settings. Its input is an explicit list of token IDs, not a prompt passed through a tokenizer. Download the source archive separately and extract it outside this repository:

```sh
WORK="$HOME/.cache/bonsaiscratch"
mkdir -p "$WORK"
curl -fL 'https://github.com/PrismML-Eng/llama.cpp/archive/e3de715222b2667c54a267441159a5729f11155e.tar.gz' -o "$WORK/prism-source.tar.gz"
tar -xzf "$WORK/prism-source.tar.gz" -C "$WORK"
python3 tools/reference/reference.py build --archive "$WORK/prism-source.tar.gz" --source "$WORK/llama.cpp-e3de715222b2667c54a267441159a5729f11155e" --build "$WORK/reference-build"
python3 tools/reference/reference.py run --model "$WORK/model.gguf" --build "$WORK/reference-build" --output "$WORK/fixture" --tokens 1 --context 128
```

`run --help` and `m4 --help` describe graph capture, M4 transform/embedding/head fixtures, and successive greedy decode. A fixture includes logits, opaque native sequence-state snapshots and build metadata. With a capture filter, the harness separately records selected graph tensors and the actual CPU Q8_K staging for matching PQ2_0 matrix operations. Graph-derived Q8_K files are separate reference quantizations, not staging buffers. Callback inputs are read after the node executes and can have been overwritten by an in-place operation; capture the producer node separately when pre-operation input is required. See [the numerical contract](docs/numerical-contract.md) before comparing fixtures.

## Local loader

The [storage-only extension](docs/extension.md) loads a locally selected package into existing TurboWarp lists. Host-driven read and copy measurements are in `benchmarks/results/`; they are not compiled-kernel or target-hardware benchmarks.

## Scratch primitive slice

`dist/fp16-q8-scalar.sb3` is a partial test project, not a full decoder. Generate it with `python3 tools/generate_project.py dist/fp16-q8-scalar.sb3`. The default mode 0 decodes FP16 words; mode 1 quantizes Q8_K activation groups; mode 2 computes one scalar ternary row; mode 3 rounds F32; modes 4–8 exercise dense dot, RMS/L2 normalization, SiLU and softmax; mode 9 uses raw-byte, BS-LUT4, or BS-LUT5 lookup; mode 10 converts radians to Scratch's degree-based sine/cosine; modes 12–13 run normalized block-512 Hadamard transforms with explicit signs (forward/inverse); modes 14–16 run inverse embedding lookup (resident/loader) and final `output_norm`; mode 18 runs greedy selection with lower-ID tie breaking; mode 40 runs one complete full-attention decoder layer against the layer-loader package. Inspect the Stage variables and lists.

The compiled VM integration tests insert the real loader block into a test copy, load a real model row, disable further loader reads, and execute the resident Scratch arithmetic. They use the audited VM source pin; this does not pin a deployed TurboWarp application build.

Generate the separate text proof with `python3 tools/generate_project.py dist/text-proof.sb3 --text`. It uses 256 hidden sprites for exact UTF-16 code-unit lookup; see the numerical contract for its substantial measured startup and memory cost. `python3 tools/generate_project.py dist/head-benchmark.sb3 --benchmark` generates the streaming head project (640-row default tiles, 250 fps pacing): mode 11 scans all 248,320 rows from a transformed input, while mode 17 runs the end-to-end M4 head (`output_norm` → signs+Hadamard → Q8_K → all 248,320 rows → top-8 and greedy token); load the storage extension and select a package first.

For full-logit head validation, generate an external native reference using `tools/reference/reference.py run` with token `1`, context `8`, and `--capture result`; pass its `manifest.json` and `logits-0.f32` to `benchmarks/head.py` using `--native-manifest` and `--native-logits` alongside `--m4-manifest fixtures/native/m4/head/manifest.json`. The benchmark verifies provenance and compares every logit; the native vector is not stored in Git.

## Prompt to text

The tokenizer and detokenizer run in Scratch blocks. `python3 tools/generate_project.py dist/text-proof.sb3 --text --benchmark` adds three modes for a selected package: mode 22 turns the prompt in `BS prompt` into token ids in `BS prompt tokens`, mode 23 maps `BS out tokens` back to text in `BS text`, and mode 24 tokenizes the prompt, then broadcasts one decode step per token (`BS max tokens`, default 8) and shows the growing text on the `Bonsai` sprite.

The pre-tokenizer reproduces the pinned runtime's own splitter. Its code point classes come from `fixtures/unicode/qwen35-classes.json`, derived by `tools/unicode_classes.py` from `src/unicode-data.cpp` at the pinned revision rather than from the host's Unicode build, because newer assignments would split prompts differently. The pair merge table is packed into the package as `prism.bpe.*` tensors by `tools/package_model.py`. Ten fixture classes match the pinned tokenizer exactly in the compiled VM; see `tests/scratch/test_tokenizer.py`. Each decode step is one Scratch run, so a prompt plus generated tokens takes minutes on this host (128-token bound, clamped by the 128-cell KV cache).

## Running the demo

1. Open `dist/bonsai-prompt-to-text.sb3` in TurboWarp.
2. Add the storage extension (`extension/loader.js`) and click **select Bonsai package**, choosing the `bslut5` package built above.
3. Press the green flag. An ask box takes the prompt; the `Bonsai` sprite says the generated text as each token lands.

One flag press runs the whole generation on the Scratch side (the loop broadcasts one decode step per token). Expect about 12 minutes before the first token and about a minute per further token on the recorded Xeon host, with 4.3-4.8 GiB peak memory in the Node VM; the browser renderer adds to that. A new prompt resets the sequence state, and generation stops at `<|im_end|>`, at the 128-token bound, or when the prompt has filled the 128-cell KV cache.

## Decoder benchmark

`python3 benchmarks/decoder.py --output benchmarks/results/decoder-xeon.json` runs two prompts end to end at the production configuration, one flag press per decode step, and rejects a run whose first generated token differs from the pinned native build's choice. It records prompt tokens, wall and CPU time, per-step time, peak RSS and the rendered text; the current Xeon numbers are in that file and summarized in the decision record.

## Tests

```sh
python3 -m pytest -q tests/
node --test tests/transport/*.test.js
BONSAI_DECODE_MATRIX=1 python3 -m pytest -q tests/scratch/test_deltanet.py::DeltaNetStageTests::test_decoder_matches_the_native_fixture_matrix
```

The full suite takes about 90 minutes on the recorded host because three typed prompts run through the compiled VM end to end; the ten-token decode matrix adds about 8 minutes when the environment variable is set. Set `BONSAI_FIXTURE_TMP` to a disk-backed directory: several tests write multi-hundred-megabyte temporaries.

Full model files and large generated manifests and fixtures belong outside Git. Current development measurements on the Intel Xeon host do not establish performance or memory use on the plan's Ryzen-class 16 GB target.

The [M3/M4 decision record](docs/m3-decision.md) summarizes the speed redesign, `bslut5` adoption, projected decode/prefill costs, and measured M4 end-to-end head results on the Xeon host.

## Running it in TurboWarp (browser or desktop)

The `.sb3` files in `dist/` do not contain the model. You need the package first, then the
loader extension, then the package selected at runtime.

### 1. Fetch the locked GGUF

```sh
export WORK=~/.cache/bonsaiscratch
mkdir -p "$WORK/build-tmp"
# prism-ml/Ternary-Bonsai-2-2B-gguf at revision a6032a5089ac27e96be3a80c7b42d1ad0bd89887
# file: Ternary-Bonsai-2-2B-PQ2_0.gguf  (model-lock.json records the hash)
```

Check `model-lock.json` for the pinned repository, revision, filename and SHA-256, and copy
or download that file to `"$WORK/model.gguf"`. The packaging step refuses a model whose
identity does not match the lock.

### 2. Build the package (~537 MB)

```sh
TMPDIR="$WORK/build-tmp" python3 tools/package_model.py \
  "$WORK/model.gguf" "$WORK/model-bslut5-tokenizer.bsp" --codec bslut5
```

It validates the GGUF, reads the package back, reverses every PQ2_0 group and compares the
result byte-for-byte with the source, then prints the package SHA-256. Keep that checksum.
The package is not committed and must not be uploaded to Git.

### 3. Load the extension

```sh
python3 -m http.server 8000     # from the repo root
```

In TurboWarp, open Extensions → Custom Extension, load
`http://localhost:8000/extension/loader.js` (or the file itself), and tick
**"Run extension without sandbox"**. An unsandboxed extension has full page access: only
load source you trust. This works in the browser editor as well as the desktop app.

### 4. Load the project and the package

Open `dist/bonsai-prompt-to-text.sb3`. The updated loader opens the **Bonsai chat**
window automatically; use **Open Bonsai chat** to reopen it after closing it.
Click **Select model**, choose the `.bsp` you built, and wait for **Ready**.
A failed selection displays its error in the status area.

### 5. Run

Type in **Message Bonsai** and click **Send** (or Enter; Shift+Enter inserts a new
line). Responses appear in the conversation as tokens arrive. **Stop** interrupts
generation and keeps any text already generated. Each send starts a fresh model
context; earlier messages in the conversation are display-only and stay in this tab.
Without the browser chat UI, the green flag still opens the native Scratch prompt.

The status shows real VM phases: tokenization, embedding, the current **layer / 24**
for the current prompt/generation token, vocabulary rows scored during head scanning,
and response-token count with elapsed time. The progress bar is phase-local, not an
estimated overall percentage or countdown. Keep the tab foreground: background-tab
throttling slows the scheduler. Model math still runs entirely in Scratch blocks.

Measured memory was 4.2–4.8 GiB peak RSS in the headless Node VM; browser memory and
full-generation performance have not been benchmarked. The updated chat was
smoke-tested in TurboWarp web with a real package through live layer progress and Stop.
For an optional browser smoke check, see `tests/scratch/check-chat-ui.js`; it sends
one short prompt and stops after observing a real layer.
