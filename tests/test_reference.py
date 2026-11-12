import importlib.util
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

    def test_cli_rejects_invalid_tokens_before_native_execution(self):
        result = subprocess.run([sys.executable, str(SCRIPT), "run", "--model", "missing", "--build", "missing", "--output", "missing", "--tokens", "-1"], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("token", result.stderr.lower())


if __name__ == "__main__":
    unittest.main()
