import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tarfile
import tempfile
import unittest

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "tools/reference/reference.py"
spec = importlib.util.spec_from_file_location("reference", SCRIPT)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


class ReferenceTests(unittest.TestCase):
    def test_explicit_token_parser_rejects_bad_ids(self):
        self.assertEqual(reference.token_ids("1,248046,0"), [1, 248046, 0])
        for value in ("", "1,,2", "-1", "248320", "1.0"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                reference.token_ids(value)

    def test_model_identity_checks_size_and_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "model"
            path.write_bytes(b"abc")
            identity = {"size_bytes": 3, "sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"}
            reference.verify_model(path, identity)
            for bad in ({**identity, "size_bytes": 4}, {**identity, "sha256": "0" * 64}):
                with self.assertRaises(ValueError):
                    reference.verify_model(path, bad)

    def test_source_verification_rejects_modified_extracted_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            source = root / "source"
            source.mkdir()
            file = source / "source.c"
            file.write_bytes(b"original")
            archive = root / "source.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                tar.add(source, arcname="llama.cpp-" + reference.REVISION)
            old = reference.ARCHIVE_SHA256
            try:
                reference.ARCHIVE_SHA256 = reference.digest(archive)
                reference.verify_source(source, archive)
                file.write_bytes(b"modified")
                with self.assertRaises(ValueError):
                    reference.verify_source(source, archive)
            finally:
                reference.ARCHIVE_SHA256 = old

    def test_manifest_indexes_payload_checksums(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "logits.f32").write_bytes(b"abc")
            (root / "captures.tsv").write_text("0\tnode\tMUL_MAT\tout\tactivation\tF32\t4,1,1,1\t4,16,16,16\tlogits.f32\t-\n")
            captures = reference.capture_manifest(root)
            self.assertEqual(captures[0]["shape"], [4, 1, 1, 1])
            self.assertEqual(captures[0]["payload"]["sha256"], "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
            self.assertIsNone(captures[0]["q8_k"])
            (root / "logits.f32").unlink()
            with self.assertRaises(FileNotFoundError):
                reference.capture_manifest(root)

    def test_staging_manifest_records_exact_payload_and_operation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "stage-0.bin").write_bytes(b"\x00" * 260)
            (root / "staging.tsv").write_text("3\tlayer.1\tMUL_MAT\tweight\tinput\t256\tstage-0.bin\n")
            capture = reference.staging_manifest(root)[0]
            self.assertEqual((capture["step"], capture["node"], capture["weight"], capture["activation"]),
                             (3, "layer.1", "weight", "input"))
            self.assertEqual(capture["elements"], 256)
            self.assertEqual(capture["payload"]["bytes"], 260)
            self.assertEqual(capture["format"], "q8_k-d-qs")
            (root / "stage-0.bin").write_bytes(b"\x00" * 259)
            with self.assertRaises(ValueError):
                reference.staging_manifest(root)

    @unittest.skipUnless(os.environ.get('BONSAI_NATIVE_BUILD') and os.environ.get('BONSAI_MODEL_PATH'),
                         'set BONSAI_NATIVE_BUILD and BONSAI_MODEL_PATH')
    def test_real_cpu_staging_matches_locked_projection(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get('BONSAI_FIXTURE_TMP')) as directory:
            output = pathlib.Path(directory) / 'capture'
            result = subprocess.run([sys.executable, str(SCRIPT), 'run', '--model', os.environ['BONSAI_MODEL_PATH'],
                                     '--build', os.environ['BONSAI_NATIVE_BUILD'], '--output', str(output),
                                     '--tokens', '1', '--context', '128', '--capture', 'Qcur_full'],
                                    capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((output / 'manifest.json').read_text())
            capture = next(c for c in manifest['cpu_q8_k_staging'] if c['weight'] == 'blk.3.attn_q.weight')
            expected = SCRIPT.parents[2] / 'fixtures/native/layer3-q-staging.q8_k'
            self.assertEqual((output / capture['payload']['file']).read_bytes(), expected.read_bytes())
            self.assertEqual(manifest['steps'][0]['greedy_next'], 321)

    @unittest.skipUnless(os.environ.get('BONSAI_NATIVE_BUILD') and os.environ.get('BONSAI_MODEL_PATH'),
                         'set BONSAI_NATIVE_BUILD and BONSAI_MODEL_PATH')
    def test_native_pq2_projection_matches_captured_rows(self):
        root = SCRIPT.parents[2]
        with tempfile.TemporaryDirectory(dir=os.environ.get('BONSAI_FIXTURE_TMP')) as temp:
            output = pathlib.Path(temp) / 'rows.f32'
            binary = pathlib.Path(os.environ['BONSAI_NATIVE_BUILD']) / 'bonsai-reference'
            result = subprocess.run([str(binary), '--project-pq2', os.environ['BONSAI_MODEL_PATH'], '195420256',
                                     '5', str(root / 'fixtures/native/layer3-q-input.f32'), str(output)],
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            expected = (root / 'fixtures/native/layer3-q-first5.f32').read_bytes()
            self.assertEqual(output.read_bytes(), expected)



M4 = pathlib.Path(__file__).resolve().parents[1] / "fixtures/native/m4"


def f32(path):
    import struct
    data = path.read_bytes()
    return list(struct.unpack("<" + "f" * (len(data) // 4), data))


class M4FixtureTests(unittest.TestCase):
    @unittest.skipUnless(M4.is_dir(), "run tools/reference/reference.py m4 first")
    def test_manifest_provenance_and_shapes(self):
        top = json.loads((M4 / "manifest.json").read_text())
        self.assertEqual(top["schema"], "bonsai-native-m4/v1")
        self.assertEqual(top["capture_method"], "native-capture")
        for key in ("model", "build", "runtime", "machine", "seed", "input_tokens"):
            self.assertIn(key, top)
        trans = json.loads((M4 / "transforms/manifest.json").read_text())
        self.assertEqual(len(trans["associations"]), 151)
        self.assertEqual(sum(1 for a in trans["associations"] if a["role"] == "forward"), 150)
        widths = {p["activation_width"] for p in trans["pairs"] if p["signs"]}
        self.assertEqual(widths, {2048, 6144})
        emb = json.loads((M4 / "embedding/manifest.json").read_text())
        self.assertEqual([r["token_id"] for r in emb["rows"]], [1, 321, 248046])
        head = json.loads((M4 / "head/manifest.json").read_text())
        self.assertNotEqual(head["pre_norm"]["sha256"], head["post_norm"]["sha256"])
        self.assertGreater(head["margin_top1_top2"], 0)
        self.assertEqual(head["greedy_token"], head["top_k"][0]["token"])
    @unittest.skipUnless(M4.is_dir(), "run tools/reference/reference.py m4 first")
    def test_transform_pairs_satisfy_documented_math(self):
        trans = json.loads((M4 / "transforms/manifest.json").read_text())
        rot = f32(M4 / "transforms/prism.hadamard.512.f32")
        self.assertEqual(len(rot), 512 * 512)
        signs = {w: f32(M4 / f"transforms/prism.hadamard.signs.{w}.f32") for w in (2048, 6144)}
        import math
        for pair in trans["pairs"]:
            x = f32(M4 / "transforms" / pair["input"]["file"])
            y = f32(M4 / "transforms" / pair["rotation_out"]["file"])
            self.assertEqual(len(x), len(y))
            err = 0.0
            for b in range(len(x) // 512):
                for r in range(0, 512, 16):  # sample 32 rows per block
                    acc = sum(rot[r * 512 + c] * x[b * 512 + c] for c in range(512))
                    err = max(err, abs(acc - y[b * 512 + r]))
            norm_in = math.sqrt(sum(v * v for v in x))
            norm_out = math.sqrt(sum(v * v for v in y))
            with self.subTest(pair=pair["node"]):
                self.assertLess(err, 1e-4)
                self.assertAlmostEqual(norm_out / norm_in, 1.0, places=5)
                if pair["signs"]:
                    self.assertEqual(len(signs[pair["sign_width"]]), pair["activation_width"])
                    self.assertTrue(all(abs(v) == 1.0 for v in signs[pair["sign_width"]]))

    @unittest.skipUnless(M4.is_dir(), "run tools/reference/reference.py m4 first")
    def test_embedding_rows_and_tied_output(self):
        top = json.loads((M4 / "manifest.json").read_text())
        self.assertEqual(top["tied_output"]["tensor"], "token_embd.weight")
        emb = json.loads((M4 / "embedding/manifest.json").read_text())
        trans = json.loads((M4 / "transforms/manifest.json").read_text())
        rot = f32(M4 / "transforms/prism.hadamard.512.f32")
        sgn = f32(M4 / "transforms/prism.hadamard.signs.2048.f32")
        for row in emb["rows"]:
            raw = f32(M4 / "embedding" / row["raw_row"]["file"])
            post = f32(M4 / "embedding" / row["post_sign_row"]["file"])
            for b in range(4):
                for r in range(0, 512, 64):
                    acc = sum(rot[r * 512 + c] * raw[b * 512 + c] for c in range(512))
                    self.assertAlmostEqual(acc * sgn[b * 512 + r], post[b * 512 + r], places=5)
        assoc = trans["associations"]
        embd = [a for a in assoc if a["weight"] == "token_embd.weight"]
        self.assertEqual(len(embd), 1)
        self.assertEqual(embd[0]["role"], "inverse-lookup-and-head")

    def test_gguf_metadata_reader(self):
        import struct as st
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "t.gguf"
            kv = b""
            name = b"prism.hadamard.sign_widths"
            kv += st.pack("<Q", len(name)) + name + st.pack("<I", 9)
            kv += st.pack("<I", 5) + st.pack("<Q", 2) + st.pack("<2i", 4, 8)
            path.write_bytes(b"GGUF" + st.pack("<IQQ", 3, 0, 1) + kv)
            self.assertEqual(reference.gguf_metadata(path)["prism.hadamard.sign_widths"], [4, 8])
            with self.assertRaises(OSError):
                reference.gguf_metadata(pathlib.Path(d) / "missing.gguf")



    def test_cli_rejects_invalid_tokens_before_native_execution(self):
        result = subprocess.run([sys.executable, str(SCRIPT), "run", "--model", "missing", "--build", "missing", "--output", "missing", "--tokens", "-1"], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("token", result.stderr.lower())


if __name__ == "__main__":
    unittest.main()
