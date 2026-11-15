import hashlib
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / 'tools' / 'inspect_gguf.py'


def s(value):
    value = value.encode('utf-8')
    return struct.pack('<Q', len(value)) + value


def fixture():
    metadata = (
        s('general.alignment') + struct.pack('<II', 4, 32)
        + s('tokenizer.ggml.tokens') + struct.pack('<IIQ', 9, 8, 2) + s('Hello') + s('世界')
        + s('tokenizer.ggml.add_bos_token') + struct.pack('<IB', 7, 1)
        + s('special.float') + struct.pack('<If', 6, float('inf'))
    )
    tensors = (
        s('weights') + struct.pack('<IQQIQ', 2, 128, 2, 142, 0)
        + s('norm') + struct.pack('<IQIQ', 1, 3, 0, 96)
    )
    header = b'GGUF' + struct.pack('<IQQ', 3, 2, 4) + metadata + tensors
    return header + bytes((-len(header)) % 32) + bytes(68) + bytes(28) + bytes(12)


class InspectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.model = Path(self.temp.name) / 'model.gguf'
        self.output = Path(self.temp.name) / 'manifest.json'

    def run_cli(self, data, *args):
        self.model.write_bytes(data)
        return subprocess.run([sys.executable, str(SCRIPT), str(self.model), str(self.output), *args], capture_output=True, text=True)

    def test_inventory_and_deterministic_output(self):
        data = fixture()
        result = self.run_cli(data, '--no-lock')
        self.assertEqual(result.returncode, 0, result.stderr)
        first = self.output.read_bytes()
        manifest = json.loads(first)
        self.assertEqual(manifest['source']['sha256'], hashlib.sha256(data).hexdigest())
        self.assertEqual(manifest['source']['size_bytes'], len(data))
        self.assertEqual(manifest['metadata']['tokenizer.ggml.tokens']['value'], ['Hello', '世界'])
        self.assertIs(manifest['metadata']['tokenizer.ggml.add_bos_token']['value'], True)
        self.assertEqual(manifest['metadata']['special.float']['value'], 'Infinity')
        weights, norm = manifest['tensors']
        self.assertEqual((weights['type_id'], weights['dimensions'], weights['size_bytes']), (142, [128, 2], 68))
        self.assertEqual((norm['type_id'], norm['dimensions'], norm['size_bytes']), (0, [3], 12))
        self.assertEqual(norm['offset'], weights['offset'] + 96)
        self.assertEqual(weights['offset'] % 32, 0)
        self.assertEqual(self.run_cli(data, '--no-lock').returncode, 0)
        self.assertEqual(first, self.output.read_bytes())

    def test_lock_is_enforced_by_default(self):
        result = self.run_cli(fixture())
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.output.exists())

    def test_checks_hash_even_when_size_matches(self):
        data = fixture()
        lock = Path(self.temp.name) / 'lock.json'
        lock.write_text(json.dumps({'size_bytes': len(data), 'sha256': '0' * 64}))
        result = self.run_cli(data, '--lock', str(lock))
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.output.exists())
        lock.write_text(json.dumps({'size_bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}))
        result = self.run_cli(data, '--lock', str(lock))
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_rejects_malformed_files_without_publishing_output(self):
        original = fixture()
        cases = [
            b'bad!' + original[4:],
            original[:20],
            original[:-1],
            original.replace(struct.pack('<IQQIQ', 2, 128, 2, 142, 0), struct.pack('<IQQIQ', 2, 128, 2, 143, 0), 1),
            original.replace(struct.pack('<IQQIQ', 2, 128, 2, 142, 0), struct.pack('<IQQIQ', 2, 127, 2, 142, 0), 1),
            original.replace(s('norm') + struct.pack('<IQIQ', 1, 3, 0, 96), s('norm') + struct.pack('<IQIQ', 1, 3, 0, 32), 1),
            original.replace(s('general.alignment') + struct.pack('<II', 4, 32), s('general.alignment') + struct.pack('<II', 4, 3), 1),
        ]
        for data in cases:
            with self.subTest(data=data[:24]):
                result = self.run_cli(data, '--no-lock')
                self.assertNotEqual(result.returncode, 0, result.stderr)
                self.assertFalse(self.output.exists())

    def test_rejects_truncated_or_invalid_metadata(self):
        prefix = b'GGUF' + struct.pack('<IQQ', 3, 0, 1)
        cases = [
            prefix + s('x') + struct.pack('<I', 99),
            prefix + s('x') + struct.pack('<IQ', 8, 1000),
            prefix + s('x') + struct.pack('<IB', 7, 2),
            prefix + s('x') + struct.pack('<IIQ', 9, 8, 2) + s('only one'),
        ]
        for data in cases:
            with self.subTest(data=data):
                self.assertNotEqual(self.run_cli(data, '--no-lock').returncode, 0)
                self.assertFalse(self.output.exists())

    def test_large_metadata_string_crosses_utf8_chunk_boundary(self):
        text = 'a' * (1024 * 1024 - 1) + '世界'
        metadata = s('large') + struct.pack('<I', 8) + s(text)
        header = b'GGUF' + struct.pack('<IQQ', 3, 0, 1) + metadata
        data = header + bytes((-len(header)) % 32)
        result = self.run_cli(data, '--no-lock')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(self.output.read_text())['metadata']['large']['value'], text)

    def test_failed_inspection_preserves_existing_output(self):
        self.output.write_text('previous')
        result = self.run_cli(fixture()[:-1], '--no-lock')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.output.read_text(), 'previous')

    def test_output_must_not_overwrite_input(self):
        data = fixture()
        self.model.write_bytes(data)
        result = subprocess.run([sys.executable, str(SCRIPT), str(self.model), str(self.model), '--no-lock'], capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.model.read_bytes(), data)


if __name__ == '__main__':
    unittest.main()
