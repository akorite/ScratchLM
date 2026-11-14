#!/usr/bin/env python3
"""Offline pinned native fixtures. Run with --help for build/run/quantize options."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import struct
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

def gguf_metadata(path):
    """Minimal GGUFv3 KV reader for the prism.hadamard.* metadata arrays."""
    scalars = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
               6: "<f", 7: "<B", 10: "<Q", 11: "<q", 12: "<d"}
    with path.open("rb") as f:
        head = f.read(24)
        if len(head) != 24 or head[:4] != b"GGUF":
            raise ValueError("model is not a GGUF file")
        version, _, kv_count = struct.unpack("<IQQ", head[4:24])
        if version != 3:
            raise ValueError(f"unsupported GGUF version {version}")

        def read_str():
            (n,) = struct.unpack("<Q", f.read(8))
            return f.read(n).decode("utf-8")

        def read_value(kind):
            if kind == 8:
                return read_str()
            if kind == 9:
                elem, count = struct.unpack("<IQ", f.read(12))
                return [read_value(elem) for _ in range(count)] if elem == 8 else \
                    list(struct.unpack("<" + scalars[elem][1] * count, f.read(count * struct.calcsize(scalars[elem]))))
            return struct.unpack(scalars[kind], f.read(struct.calcsize(scalars[kind])))[0]

        meta = {}
        for _ in range(kv_count):
            key = read_str()
            (kind,) = struct.unpack("<I", f.read(4))
            meta[key] = read_value(kind)
        return meta


def copy_artifact(dst_dir, src_dir, src_artifact, name):
    shutil.copyfile(src_dir / src_artifact["file"], dst_dir / name)
    return artifact(dst_dir / name)

def f32_values(path):
    data = path.read_bytes()
    return list(struct.unpack("<" + "f" * (len(data) // 4), data))


def m4(args):
    if sys.byteorder != "little":
        raise ValueError("fixture binary format requires a little-endian host")
    identity = json.loads((ROOT / "model-lock.json").read_text(encoding="utf-8"))
    verify_model(args.model, identity)
    binary, build_meta = checked_build(args.build)
    settings = json.loads((ROOT / "runtime-lock.json").read_text(encoding="utf-8"))
    prism = {k: v for k, v in gguf_metadata(args.model).items() if k.startswith("prism.hadamard.")}
    if prism.get("prism.hadamard.transform") != "normalized-sylvester-walsh-hadamard" \
            or prism.get("prism.hadamard.sign_mode") != "explicit":
        raise ValueError("model does not carry the pinned Hadamard-v2 explicit-sign metadata")
    block_size = prism["prism.hadamard.block_size"]
    sign_widths = prism["prism.hadamard.sign_widths"]
    sign_values = prism["prism.hadamard.sign_values"]
    weights = prism["prism.hadamard.weight_names"]
    inverses = prism["prism.hadamard.inverse_weight_names"]
    tensors = {t["name"]: t for t in json.loads((ROOT / "fixtures/tensor-manifest.json").read_text())["tensors"]}
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    work = out / "native-runs"
    runs = {}
    # Graph tensors are shared across steps, so each embedding token needs its
    # own run; the "result" and first "hadamard" runs reuse token[0] for the
    # norm pair and head activation.
    runs["hadamard-0"] = work / "hadamard-0"
    runs["hadamard-0"].mkdir(parents=True)
    run_command([binary, args.model.resolve(), runs["hadamard-0"], args.context, 0, "hadamard", str(args.tokens[0])])
    for i, token in enumerate(args.tokens[1:], 1):
        dest = work / f"hadamard-{i}"
        dest.mkdir(parents=True)
        run_command([binary, args.model.resolve(), dest, args.context, 0, "hadamard", str(token)])
        runs[f"hadamard-{i}"] = dest
    runs["result"] = work / "result"
    runs["result"].mkdir(parents=True)
    run_command([binary, args.model.resolve(), runs["result"], args.context, 0, "result", str(args.tokens[0])])
    runs["pre_norm"] = work / "pre_norm"
    runs["pre_norm"].mkdir(parents=True)
    run_command([binary, args.model.resolve(), runs["pre_norm"], args.context, 0, "l_out-23", str(args.tokens[0])])
    caps = capture_manifest(runs["hadamard-0"])
    provenance = {"model": identity, "runtime": settings, "build": build_meta, "machine": machine(),
                  "requested_context": args.context, "input_tokens": args.tokens,
                  "capture_method": "native-capture",
                  "capture_note": "bonsai-reference --capture hadamard|result on the pinned CPU graph; greedy decode is deterministic (locked seed is recorded, not consumed)",
                  "seed": settings["seed"], "endianness": "little", "format": "IEEE-754 binary32"}

    trans = out / "transforms"
    trans.mkdir()
    emb = out / "embedding"
    emb.mkdir()
    head_dir = out / "head"
    head_dir.mkdir()

    # The TSV lists each rotation MUL_MAT as an "out" row followed by its
    # chain rows (s0 rotation, s1 reshape, s1.0 sign-mul or producer, s1.0.x).
    chains, current = [], None
    for c in caps:
        if c["role"] == "out" and c["operation"] == "MUL_MAT":
            current = {"step": c["step"], "node": c["node"], "out": c}
            chains.append(current)
        elif current is not None:
            current.setdefault("members", []).append(c)
    for chain in chains:
        members = {c["role"]: c for c in chain.pop("members", [])}
        if "s1" not in members or "s1.0" not in members:
            chain["skip"] = True
            continue
        chain["rotation_out"] = copy_artifact(trans, runs["hadamard-0"], chain["out"]["payload"], chain["node"] + "-out.f32")
        if "s0" in members:
            copy_artifact(trans, runs["hadamard-0"], members["s0"]["payload"], "prism.hadamard.512.f32")
        chain["rotation"] = artifact(trans / "prism.hadamard.512.f32")
        signed = members.get("s1.0")
        if signed and signed["operation"] == "MUL":
            # input = sign-multiplied activation (s1.0); pre_sign = the
            # producer before signs (s1.0.0). The sign tensor is shared across
            # chains; any captured row naming it can seed the canonical file.
            leaf = members.get("s1.0.0", signed)
            chain["pre_sign"] = copy_artifact(trans, runs["hadamard-0"], leaf["payload"], chain["node"] + "-pre-sign.f32")
            chain["sign_width"] = signed["shape"][0]
            name = f"prism.hadamard.signs.{chain['sign_width']}.f32"
            if not (trans / name).exists():
                row = members.get("s1.0.1") or next(
                    (r for r in caps if r["step"] == chain["step"]
                     and r["name"] == f"prism.hadamard.signs.{chain['sign_width']}"), None)
                if row:
                    copy_artifact(trans, runs["hadamard-0"], row["payload"], name)
            chain["signs"] = artifact(trans / name)
            chain["input"] = copy_artifact(trans, runs["hadamard-0"], signed["payload"], chain["node"] + "-input.f32")
        else:
            leaf, chain["pre_sign"], chain["sign_width"], chain["signs"] = signed, None, None, None
            chain["input"] = copy_artifact(trans, runs["hadamard-0"], leaf["payload"], chain["node"] + "-input.f32")
        chain["input_leaf"] = {"name": leaf["name"], "operation": leaf["operation"]}
        chain["blocks"] = members["s1"]["shape"][1]
        chain["activation_width"] = chain["out"]["shape"][0] * chain["out"]["shape"][1]
        del chain["out"]

    chains = [c for c in chains if not c.get("skip")]
    for c in chains:
        c.pop("skip", None)

    # Captured sign tensors must match model metadata sign_values bit-for-bit.
    offset = 0
    for width in sign_widths:
        got = f32_values(trans / f"prism.hadamard.signs.{width}.f32")
        if got != [float(v) for v in sign_values[offset:offset + width]]:
            raise ValueError(f"captured prism.hadamard.signs.{width} differs from model metadata sign_values")
        offset += width

    
    leaf_kind = {"attn_norm": "attn-qkv-gate", "attn_post_norm": "attn-out-ssm",
                 "final_output": "ssm-out-permute", "ffn_swiglu": "ffn-down",
                 "result_norm": "head", "model.input_embed": "inverse", "norm": "head",
                 "GET_ROWS": "inverse-lookup"}
    for chain in chains:
        chain["kind"] = leaf_kind.get(chain["input_leaf"]["name"].split("-")[0], "forward")
        if chain["input_leaf"]["operation"] == "GET_ROWS":
            chain["kind"] = "inverse-lookup"

    # Folded weights consume transforms built on these activation leaves:
    # attn_norm-N feeds attn_qkv/attn_gate, final_output-N feeds ssm_out,
    # attn_post_norm-N feeds ffn_gate/ffn_up, ffn_swiglu-N feeds ffn_down.
    leaf_family = {"attn_qkv": "attn_norm", "attn_gate": "attn_norm", "ssm_out": "final_output",
                   "ffn_gate": "attn_post_norm", "ffn_up": "attn_post_norm", "ffn_down": "ffn_swiglu"}
    associations = []
    for name in weights:
        t = tensors[name]
        stem = name.removesuffix(".weight")
        family, layer = stem.split(".", 2)[-1], stem.split(".")[1]
        leaf = leaf_family.get(family)
        sample = next((c["node"] for c in chains
                       if c["signs"] and c["input_leaf"]["name"] == f"{leaf}-{layer}"), None)
        associations.append({"weight": name, "role": "forward", "input_width": t["dimensions"][0],
                             "output_width": t["dimensions"][1], "block_size": block_size,
                             "sign_width": t["dimensions"][0] if t["dimensions"][0] in sign_widths else None,
                             "input_leaf": f"{leaf}-{layer}" if leaf else None, "sample_chain": sample})
    for name in inverses:
        t = tensors[name]
        associations.append({"weight": name, "role": "inverse-lookup-and-head", "input_width": t["dimensions"][0],
                             "output_width": t["dimensions"][1], "block_size": block_size,
                             "sign_width": t["dimensions"][0], "chain_kind": "inverse-lookup+head"})

    write_json(trans / "manifest.json",
               {"schema": "bonsai-native-m4-transforms/v1", **provenance,
                "transform": prism["prism.hadamard.transform"], "axis": prism["prism.hadamard.axis"],
                "sign_mode": "explicit", "block_size": block_size,
                "rotation": artifact(trans / "prism.hadamard.512.f32"),
                "sign_vectors": {str(w): artifact(trans / f"prism.hadamard.signs.{w}.f32") for w in sign_widths},
                "pairs": chains, "associations": associations,
                "sign_provenance": "sign files are the captured prism.hadamard.signs.<width> tensors, verified equal to model metadata sign_values",
                "folded_weight_note": "folded weight payloads are the locked PQ2_0 tensors in fixtures/tensor-manifest.json; pairs capture the activation-side transform only"})

    rows = []
    for i, tid in enumerate(args.tokens):
        run_caps = caps if i == 0 else capture_manifest(runs[f"hadamard-{i}"])
        run_dir = runs[f"hadamard-{i}"]
        inv = next((c for c in run_caps if c["role"] == "s1.0" and c["operation"] == "GET_ROWS"), None)
        rot = next((c for c in run_caps if c["role"] == "out" and c["operation"] == "MUL_MAT"
                    and c["shape"][0] == block_size), None)
        post = next((c for c in run_caps if c["name"] == "model.input_embed"), None)
        if inv and rot and post:
            rows.append({"token_id": tid,
                         "raw_row": copy_artifact(emb, run_dir, inv["payload"], f"token-{tid}-raw.f32"),
                         "inverse_out": copy_artifact(emb, run_dir, rot["payload"], f"token-{tid}-inverse.f32"),
                         "post_sign_row": copy_artifact(emb, run_dir, post["payload"], f"token-{tid}-row.f32"),
                         "chain": rot["node"]})
    write_json(emb / "manifest.json",
               {"schema": "bonsai-native-m4-embedding/v1", **provenance, "tensor": tensors["token_embd.weight"],
                "rows": rows,
                "lookup_note": "raw_row is the get_rows output in the rotated basis; the head consumes that basis directly while the embedding lookup applies rotation then signs to produce post_sign_row (model.input_embed)"})

    result_caps = capture_manifest(runs["result"])
    norm_out = next(c for c in result_caps if c["name"] == "result_norm" and c["role"] == "out")
    pre_norm_caps = capture_manifest(runs["pre_norm"])
    norm_in = next(c for c in pre_norm_caps if c["name"] == "l_out-23" and c["role"] == "out")
    logits = f32_values(runs["result"] / "logits-0.f32")
    order = sorted(range(len(logits)), key=lambda i: -logits[i])
    top = [{"token": i, "logit": logits[i]} for i in order[:8]]
    head_chain = next(c for c in chains if c["input_leaf"]["name"] == "result_norm")
    write_json(head_dir / "manifest.json",
               {"schema": "bonsai-native-m4-head/v1", **provenance,
                "pre_norm": copy_artifact(head_dir, runs["pre_norm"], norm_in["payload"], "output-norm-input.f32"),
                "post_norm": copy_artifact(head_dir, runs["result"], norm_out["payload"], "output-norm-output.f32"),
                "transformed_activation": copy_artifact(head_dir, trans, head_chain["rotation_out"], "head-input-transformed.f32"),
                "head_chain": head_chain["node"], "weight": tensors["token_embd.weight"],
                "vocabulary_size": len(logits), "top_k": top, "greedy_token": top[0]["token"],
                "margin_top1_top2": top[0]["logit"] - top[1]["logit"],
                "tie_rule": "native argmax returns the lower token index on exact ties (std::max_element)"})
    write_json(out / "manifest.json",
               {"schema": "bonsai-native-m4/v1", **provenance,
                "fixtures": {"transforms": "transforms/manifest.json", "embedding": "embedding/manifest.json",
                             "head": "head/manifest.json"},
                "tied_output": {"tensor": "token_embd.weight",
                                "note": "exactly one token_embd.weight tensor serves the embedding lookup, the inverse lookup transform, and the output head; no output.weight exists"},
                "chains_captured": len(chains)})
    shutil.rmtree(work)


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
    p = sub.add_parser("m4", help="capture M4 fixtures: Hadamard chains, embedding rows, output_norm, head goldens")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="new fixture directory")
    p.add_argument("--tokens", type=lambda s: token_ids(s), default=[1, 321, 248046],
                   help="comma-separated token IDs for embedding rows")
    p.add_argument("--context", type=int, default=8)
    p.set_defaults(func=m4)
    args = parser.parse_args()
    try:
        args.func(args)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
