# Model package format

A package is a single local file. Integers and tensor storage use little-endian byte order. The first eight bytes are `BNSPKG1\0`, followed by an unsigned 64-bit JSON byte length, the UTF-8 JSON manifest, and the payload. Tensor offsets in the manifest are byte offsets from the start of the payload, not the start of the file.

The manifest identifies schema `bonsaiscratch-package/v1`, the source model size and SHA-256, codec, endianness, GGUF alignment, complete typed GGUF metadata, payload length, and each tensor's name, GGML dimensions, source offset, package offset, type and byte length. Dimension zero is the contiguous input dimension. The manifest also identifies the unmodified original GGUF header stored at the start of the payload, preserving the exact tokenizer, transform and metadata bytes even where JSON cannot preserve float bit patterns.

F32 tensor payload bytes are unchanged. A PQ2_0 group is always an original two-byte FP16 scale followed by 32 code bytes for 128 weights. The raw codec retains each original two-bit byte. The `bslut4` codec replaces a raw byte's four codes `q0..q3` with `q0 + 3*q1 + 9*q2 + 27*q3`; valid values are 0 through 80. The `bslut5` codec replaces the 32 code bytes with 26 index bytes followed by 6 required-zero spare bytes: byte j (0 ≤ j ≤ 24) is the base-3 index of fields 5j..5j+4, `t5j + 3*t5j+1 + 9*t5j+2 + 27*t5j+3 + 81*t5j+4` ∈ [0,242], and byte 25 is the tail index `t125 + 3*t126 + 9*t127 + 108` ∈ [108,134] (pad fields 1,1). All codecs use 34 bytes per group. Code 3 is rejected, including for raw packages.

`tools/package_model.py` checks the model identity against `model-lock.json`, validates GGUF structure, constructs the package, reads it back, reverses every PQ2_0 group and compares every original tensor byte and the source header. It verifies the source hash again and prints the SHA-256 of the complete package. It refuses to overwrite an existing output. Keep the printed checksum alongside any distributed package.

```sh
WORK="$HOME/.cache/bonsaiscratch"
python3 tools/package_model.py "$WORK/model.gguf" "$WORK/model-raw.bsp" --codec raw
python3 tools/package_model.py "$WORK/model.gguf" "$WORK/model-bslut4.bsp" --codec bslut4
```

Neither package is committed to Git. The source GGUF and both generated packages occupy approximately 1.5 GB combined, excluding temporary files and the JSON manifests embedded in the packages.
