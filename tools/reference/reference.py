#!/usr/bin/env python3
"""Offline pinned native fixtures. Run with --help for build/run/quantize options."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ARCHIVE_SHA256 = "e0dbe5fde916a15a93cf11fe6467b320bf23b6b60fca772d3c136fdf880fda70"
REVISION = "e3de715222b2667c54a267441159a5729f11155e"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact(path):
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": digest(path)}


def token_ids(text):
    parts = text.split(",")
    if not parts or any(not p.isascii() or not p.isdecimal() or int(p) >= 248320 for p in parts):
        raise ValueError("tokens must be comma-separated IDs in [0, 248320)")
    return [int(p) for p in parts]


def verify_model(path, identity):
    if path.stat().st_size != identity["size_bytes"] or digest(path) != identity["sha256"]:
        raise ValueError("model size or SHA-256 does not match model-lock.json")


def verify_source(source, archive):
    if digest(archive) != ARCHIVE_SHA256:
        raise ValueError("source archive SHA-256 does not match the pinned upstream archive")
    prefix = "llama.cpp-" + REVISION + "/"
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if member.isdir() and member.name == prefix[:-1]:
                continue
            if not member.name.startswith(prefix):
                raise ValueError("unexpected archive prefix")
            rel = Path(member.name[len(prefix):])
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError("unsafe source archive path")
            path = source / rel
            if member.isfile():
                f = tar.extractfile(member)
                if path.is_symlink() or not path.is_file() or hashlib.sha256(f.read()).hexdigest() != digest(path):
                    raise ValueError(f"source differs from pinned archive: {rel}")
            elif member.issym():
                if not path.is_symlink() or os.readlink(path) != member.linkname:
                    raise ValueError(f"source symlink differs from pinned archive: {rel}")


def capture_manifest(root):
    captures = []
    with (root / "captures.tsv").open(encoding="utf-8") as f:
        for line in f:
            step, node, operation, role, name, dtype, shape, strides, payload, q8 = line.rstrip("\n").split("\t")
            captures.append({"step": int(step), "node": node, "operation": operation, "role": role,
                             "name": name, "type": dtype, "shape": [int(x) for x in shape.split(",")],
                             "strides_bytes": [int(x) for x in strides.split(",")],
                             "payload": artifact(root / payload), "q8_k": None if q8 == "-" else artifact(root / q8),
                             "q8_k_unavailable": "not contiguous F32 with complete, finite, non-overflowing Q8_K groups" if q8 == "-" else None})
    return captures

def staging_manifest(root):
    captures = []
    with (root / "staging.tsv").open(encoding="utf-8") as f:
        for line in f:
            step, node, operation, weight, activation, elements, payload = line.rstrip("\n").split("\t")
            elements = int(elements)
            data = artifact(root / payload)
            if elements <= 0 or elements % 256 or data["bytes"] != elements // 256 * 260:
                raise ValueError("invalid Q8_K staging payload size")
            captures.append({"step": int(step), "node": node, "operation": operation,
                             "weight": weight, "activation": activation, "elements": elements,
                             "format": "q8_k-d-qs", "payload": data})
    return captures


def instrument_source(path):
    text = path.read_text(encoding="utf-8")
    declaration = "void ggml_compute_forward_mul_mat(\n"
    point = "    ggml_barrier(params->threadpool);\n\n#if GGML_USE_LLAMAFILE\n"
    if text.count(declaration) != 1 or text.count(point) != 1:
        raise ValueError("pinned CPU source does not match instrumentation anchors")
    text = text.replace(declaration,
        "extern void bonsai_capture_q8k(const struct ggml_tensor *, const void *, int64_t);\n\n" + declaration)
    text = text.replace(point, "    ggml_barrier(params->threadpool);\n"
        "    if (ith == 0 && src0->type == GGML_TYPE_PQ2_0 && src1->type == GGML_TYPE_F32 && vec_dot_type == GGML_TYPE_Q8_K) {\n"
        "        bonsai_capture_q8k(dst, params->wdata, ne10 * ne11 * ne12 * ne13);\n"
        "    }\n\n#if GGML_USE_LLAMAFILE\n")
    path.write_text(text, encoding="utf-8")


def run_command(args):
    subprocess.run([str(x) for x in args], check=True)


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build(args):
    source, archive, dest = args.source.resolve(), args.archive.resolve(), args.build.resolve()
    if args.jobs < 1:
        raise ValueError("build jobs must be positive")
    verify_source(source, archive)
    dest.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(dest, filter="data")
    instrumented = dest / ("llama.cpp-" + REVISION)
    verify_source(instrumented, archive)
    cpu_source = instrumented / "ggml/src/ggml-cpu/ggml-cpu.c"
    pristine_cpu_sha256 = digest(cpu_source)
    instrument_source(cpu_source)
    instrumented_cpu_sha256 = digest(cpu_source)
    configure = ["cmake", "-S", HERE, "-B", dest, "-DLLAMA_SOURCE=" + str(instrumented), "-DCMAKE_BUILD_TYPE=Release"]
    run_command(configure)
    run_command(["cmake", "--build", dest, "--target", "bonsai-reference", "--parallel", str(args.jobs)])
    binary = dest / "bonsai-reference"
    cache = (dest / "CMakeCache.txt").read_text(encoding="utf-8")
    commands = json.loads((dest / "compile_commands.json").read_text(encoding="utf-8"))
    compilers = {}
    for line in cache.splitlines():
        if line.startswith(("CMAKE_C_COMPILER:FILEPATH=", "CMAKE_CXX_COMPILER:FILEPATH=")):
            key, compiler = line.split("=", 1)
            compilers[key] = subprocess.run([compiler, "--version"], capture_output=True, text=True, check=True).stdout
    meta = {"revision": REVISION, "source_archive_sha256": ARCHIVE_SHA256, "binary": artifact(binary),
            "instrumentation": {"source": "ggml/src/ggml-cpu/ggml-cpu.c",
                                "pristine_sha256": pristine_cpu_sha256,
                                "instrumented_sha256": instrumented_cpu_sha256,
                                "method": "reference.py instrument_source"},
            "harness_sources": {name: digest(HERE / name) for name in ("native.cpp", "CMakeLists.txt", "reference.py")},
            "configure": [str(x) for x in configure], "cmake_cache": cache, "compile_commands": commands,
            "environment": {key: os.environ.get(key) for key in ("CC", "CXX", "CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS")},
            "compilers": compilers, "build_machine": machine()}
    write_json(dest / "reference-build.json", meta)


def checked_build(path):
    meta = json.loads((path / "reference-build.json").read_text(encoding="utf-8"))
    binary = path / "bonsai-reference"
    if meta["revision"] != REVISION or meta["source_archive_sha256"] != ARCHIVE_SHA256 or artifact(binary) != meta["binary"]:
        raise ValueError("native build identity or binary hash mismatch")
    return binary.resolve(), meta


def machine():
    cpu = Path("/proc/cpuinfo")
    return {"platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(),
            "byteorder": sys.byteorder, "cpuinfo": cpu.read_text(encoding="utf-8") if cpu.exists() else None}


def run_fixture(args):
    tokens = token_ids(args.tokens)
    if args.context < 1 or args.context > 128 or args.generate < 0 or len(tokens) + args.generate > args.context:
        raise ValueError("tokens plus generated tokens must fit requested context, between 1 and 128")
    if sys.byteorder != "little":
        raise ValueError("fixture binary format requires a little-endian host")
    identity = json.loads((ROOT / "model-lock.json").read_text(encoding="utf-8"))
    verify_model(args.model, identity)
    binary, build_meta = checked_build(args.build)
    if "instrumentation" not in build_meta:
        raise ValueError("native build lacks CPU staging instrumentation; rebuild the reference")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    run_command([binary, args.model.resolve(), out, args.context, args.generate, args.capture or "", *tokens])
    settings = json.loads((ROOT / "runtime-lock.json").read_text(encoding="utf-8"))
    native = (out / "native.tsv").read_text(encoding="utf-8").splitlines()
    effective = int(native[0].split("\t")[1])
    vocab = int(native[1].split("\t")[1])
    steps = []
    for row in native[2:]:
        label, index, token, top1 = row.split("\t")
        if label != "step":
            raise ValueError("malformed native output")
        i = int(index)
        steps.append({"index": i, "token": int(token), "greedy_next": int(top1),
                      "logits": artifact(out / f"logits-{i}.f32"), "state": artifact(out / f"state-{i}.bin")})
    manifest = {"schema": "bonsai-native-reference/v1", "model": identity, "runtime": settings,
                "build": build_meta, "machine": machine(), "requested_context": args.context,
                "effective_context": effective, "vocabulary_size": vocab, "input_tokens": tokens,
                "generated_tokens_requested": args.generate, "steps": steps, "capture_filter": args.capture,
                "captures": capture_manifest(out), "cpu_q8_k_staging": staging_manifest(out),
                "capture_notes": "Graph callback Q8_K files are separately re-quantized, not CPU staging. CPU staging payloads contain each native block's float32 d and 256 int8 qs, with bsums omitted because zero groups may leave them uninitialized. Captures cover selected PQ2_0 MUL_MAT nodes converting F32 to Q8_K, only when a capture filter is supplied; Q8_0 fallback, prequantized inputs and other operators are excluded. State is opaque native sequence serialization.",
                "seed_note": "Greedy argmax is deterministic and uses no RNG; locked seed is recorded, not consumed.",
                "endianness": "little", "logits_format": "IEEE-754 binary32"}
    write_json(out / "manifest.json", manifest)


def quantize(args):
    if sys.byteorder != "little":
        raise ValueError("fixture binary format requires a little-endian host")
    binary, build_meta = checked_build(args.build)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    run_command([binary, "--quantize", args.input.resolve(), out / "values.q8_k"])
    write_json(out / "manifest.json", {"schema": "bonsai-native-q8-k/v1", "build": build_meta,
               "machine": machine(), "input": artifact(args.input), "output": artifact(out / "values.q8_k"),
               "format": "native block_q8_K: float32 d, 256 int8 qs, 16 int16 bsums; little-endian",
               "implementation": "quantize_row_q8_K_ref",
               "zero_groups": "bsums are zero-initialized; the pinned quantizer leaves them untouched"})

def project(args):
    if sys.byteorder != "little":
        raise ValueError("fixture binary format requires a little-endian host")
    identity = json.loads((ROOT / "model-lock.json").read_text(encoding="utf-8"))
    verify_model(args.model, identity)
    binary, build_meta = checked_build(args.build)
    tensors = json.loads((ROOT / "fixtures/tensor-manifest.json").read_text(encoding="utf-8"))["tensors"]
    tensor = next((t for t in tensors if t["name"] == args.tensor), None)
    if not tensor or tensor["type_id"] != 142 or len(tensor["dimensions"]) != 2:
        raise ValueError("tensor must name a locked two-dimensional PQ2_0 tensor")
    width, rows = tensor["dimensions"]
    if args.input.stat().st_size != width * 4:
        raise ValueError("activation size does not match tensor width")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    run_command([binary, "--project-pq2", args.model.resolve(), tensor["offset"], rows,
                 args.input.resolve(), out / "projection.f32"])
    result = artifact(out / "projection.f32")
    if result["bytes"] != rows * 4:
        raise ValueError("native projection output has wrong length")
    write_json(out / "manifest.json", {"schema": "bonsai-native-pq2-projection/v1", "model": identity,
               "build": build_meta, "machine": machine(), "tensor": tensor, "input": artifact(args.input),
               "output": result, "quantizer": "quantize_row_q8_K_ref", "dot": "CPU PQ2_0 x Q8_K type trait",
               "format": "little-endian IEEE-754 binary32", "note": "Projection only; not full model logits."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build", help="verify archive and source, then build the CPU-only native runner")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--build", type=Path, required=True, help="new build directory")
    p.add_argument("--jobs", type=int, default=1)
    p.set_defaults(func=build)
    p = sub.add_parser("run", help="produce explicit-token logits, sequence states, and optional graph captures")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="new fixture directory")
    p.add_argument("--tokens", required=True, help="comma-separated token IDs; no tokenization or implicit BOS")
    p.add_argument("--context", type=int, default=128)
    p.add_argument("--generate", type=int, default=0, help="number of greedy tokens to decode after the prompt")
    p.add_argument("--capture", help="substring of graph node name to capture, including its available inputs")
    p.set_defaults(func=run_fixture)
    p = sub.add_parser("quantize", help="quantize a raw finite little-endian F32 file with the pinned Q8_K implementation")
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(func=quantize)
    p = sub.add_parser("project", help="project a raw F32 activation through a locked PQ2_0 tensor")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--tensor", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(func=project)
    args = parser.parse_args()
    try:
        args.func(args)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
