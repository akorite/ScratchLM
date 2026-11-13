# BonsaiScratch

BonsaiScratch investigates visible Scratch-block inference with the locked PrismML Ternary Bonsai 2 2B model in TurboWarp. The [implementation plan](docs/bonsaiscratch-plan-v1.1.md) defines the numerical contract, execution boundary, milestones and acceptance gates. Native code in this repository is for offline reference fixtures only; the deployed project must perform model math in Scratch blocks.

## Model identity and inventory

The model is not stored in Git. Download the locked file and check its SHA-256 and size before using it:

```sh
WORK="$HOME/.cache/bonsaiscratch"
mkdir -p "$WORK"
curl -fL 'https://huggingface.co/prism-ml/Ternary-Bonsai-2-2B-gguf/resolve/a6032a5089ac27e96be3a80c7b42d1ad0bd89887/Ternary-Bonsai-2-2B-PQ2_0.gguf' -o "$WORK/model.gguf"
python3 tools/inspect_gguf.py "$WORK/model.gguf" "$WORK/model-manifest.json"
```

The inspector checks `model-lock.json` by default, streams the GGUF metadata into JSON, and records each tensor's dimensions, type, offsets and size. The generated manifest contains the complete tokenizer metadata and can be large. `fixtures/tensor-manifest.json` is the compact tensor-only inventory of the locked model. `--no-lock` is intended for synthetic or explicitly untrusted files, not model verification.

Create raw and BS-LUT4 packages with the commands in [the model format](docs/model-format.md). Each build verifies every group and tensor against the source before publishing the package; checksums for the locked model are in `fixtures/package-checksums.json`.

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

`run --help` describes graph capture and successive greedy decode. A fixture includes logits, opaque native sequence-state snapshots and build metadata. With a capture filter, the harness separately records selected graph tensors and the actual CPU Q8_K staging for matching PQ2_0 matrix operations. Graph-derived Q8_K files are separate reference quantizations, not staging buffers. Callback inputs are read after the node executes and can have been overwritten by an in-place operation; capture the producer node separately when pre-operation input is required. See [the numerical contract](docs/numerical-contract.md) before comparing fixtures.

## Local loader

The [storage-only extension](docs/extension.md) loads a locally selected package into existing TurboWarp lists. Host-driven read and copy measurements are in `benchmarks/results/`; they are not compiled-kernel or target-hardware benchmarks.

## Scratch primitive slice

`dist/fp16-q8-scalar.sb3` is a partial test project, not a decoder. Generate it with `python3 tools/generate_project.py dist/fp16-q8-scalar.sb3`. The default mode 0 decodes FP16 words; mode 1 quantizes Q8_K activation groups; mode 2 computes one scalar ternary row; mode 3 rounds F32; modes 4–8 exercise dense dot, RMS/L2 normalization, SiLU and softmax; mode 9 uses raw-byte or BS-LUT4 lookup; mode 10 converts radians to Scratch's degree-based sine/cosine. Inspect the Stage variables and lists. Modes 2 and 9 require an activation and a resident model row; they are not an interactive inference workflow.

The compiled VM integration tests insert the real loader block into a test copy, load a real model row, disable further loader reads, and execute the resident Scratch arithmetic. They use the audited VM source pin; this does not pin a deployed TurboWarp application build.

Generate the separate text proof with `python3 tools/generate_project.py dist/text-proof.sb3 --text`. It uses 256 hidden sprites for exact UTF-16 code-unit lookup; see the numerical contract for its substantial measured startup and memory cost. `python3 tools/generate_project.py dist/head-benchmark.sb3 --benchmark` generates the streaming head benchmark with the captured activation embedded; load the storage extension and select a package first. Its output is a diagnostic projection, not model logits.

## Tests

```sh
python3 -m unittest discover -s tests -v
node --test tests/transport/*.test.js
```

Full model files and large generated manifests and fixtures belong outside Git. Current development measurements on the Intel Xeon host do not establish performance or memory use on the plan's Ryzen-class 16 GB target.

The [M3 decision record](docs/m3-decision.md) summarizes the kernel comparison, projected decode/prefill costs, and the conditional redesign verdict that gates M4.
