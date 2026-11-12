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

`run --help` describes graph capture and successive greedy decode. A fixture includes logits, native sequence-state snapshots and build metadata. The optional graph callback records selected available tensors; its Q8_K files are reference quantizations of captured F32 tensors, not intercepted internal staging buffers. See [the numerical contract](docs/numerical-contract.md) before comparing fixtures.

## Tests

```sh
python3 -m unittest discover -s tests -v
```

Full model files and large generated manifests and fixtures belong outside Git. Current development measurements on the Intel Xeon host do not establish performance or memory use on the plan's Ryzen-class 16 GB target.
