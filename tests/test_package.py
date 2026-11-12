import hashlib
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from test_inspection import fixture

SCRIPT = Path(__file__).resolve().parents[1] / 'tools' / 'package_model.py'


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / 'model.gguf'
        data = bytearray(fixture())
        self.start = len(data) - 108
        data[self.start:self.start + 34] = b'\x00\x3c' + bytes([0x24]) * 32
        data[self.start + 34:self.start + 68] = b'\x01\x00' + bytes([0xaa]) * 32
        data[self.start + 96:self.start + 108] = bytes.fromhex('0100c07f0000803f00000080')
        self.data = bytes(data)
        self.model.write_bytes(self.data)
        self.lock = self.root / 'lock.json'
        self.lock.write_text(json.dumps({'size_bytes': len(self.data), 'sha256': hashlib.sha256(self.data).hexdigest()}))

    def package(self, codec, output):
        return subprocess.run([sys.executable, str(SCRIPT), str(self.model), str(output), '--codec', codec, '--lock', str(self.lock)], capture_output=True, text=True)

    def unpack(self, path):
        data = path.read_bytes()
        self.assertEqual(data[:8], b'BNSPKG1\0')
        size = struct.unpack('<Q', data[8:16])[0]
        return json.loads(data[16:16 + size]), data[16 + size:]

    def test_raw_and_base3_preserve_real_payloads(self):
        for codec in ('raw', 'bslut4'):
            with self.subTest(codec=codec):
                path = self.root / codec
                result = self.package(codec, path)
                self.assertEqual(result.returncode, 0, result.stderr)
                metadata, payload = self.unpack(path)
                self.assertEqual(metadata['codec'], codec)
                self.assertEqual(metadata['source']['sha256'], hashlib.sha256(self.data).hexdigest())
                self.assertEqual(metadata['metadata']['tokenizer.ggml.tokens']['value'], ['Hello', '世界'])
                weights, norm = metadata['tensors']
                w = payload[weights['offset']:weights['offset'] + weights['length']]
                f = payload[norm['offset']:norm['offset'] + norm['length']]
                self.assertEqual(f, self.data[self.start + 96:self.start + 108])
                self.assertEqual(w[:2], b'\x00\x3c')
                self.assertEqual(w[34:36], b'\x01\x00')
                if codec == 'raw':
                    self.assertEqual(w, self.data[self.start:self.start + 68])
                else:
                    self.assertEqual(w[2:34], bytes([21]) * 32)
                    self.assertEqual(w[36:68], bytes([80]) * 32)
                again = self.root / (codec + '-again')
                self.assertEqual(self.package(codec, again).returncode, 0)
                self.assertEqual(path.read_bytes(), again.read_bytes())

    def test_all_81_four_trit_patterns(self):
        name = b'weights'
        tensor = struct.pack('<Q', len(name)) + name + struct.pack('<IQQIQ', 2, 128, 3, 142, 0)
        header = b'GGUF' + struct.pack('<IQQ', 3, 1, 0) + tensor
        header += bytes((-len(header)) % 32)
        raw = [((n % 3) | ((n // 3 % 3) << 2) | ((n // 9 % 3) << 4) | ((n // 27) << 6)) for n in range(81)]
        raw.extend([0] * (96 - len(raw)))
        model = header + b''.join(b'\0\0' + bytes(raw[i:i + 32]) for i in range(0, 96, 32))
        self.model.write_bytes(model)
        self.lock.write_text(json.dumps({'size_bytes': len(model), 'sha256': hashlib.sha256(model).hexdigest()}))
        output = self.root / 'all-patterns'
        result = self.package('bslut4', output)
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest, payload = self.unpack(output)
        tensor = manifest['tensors'][0]
        packed = payload[tensor['offset']:tensor['offset'] + tensor['length']]
        values = b''.join(packed[i + 2:i + 34] for i in range(0, len(packed), 34))
        self.assertEqual(values[:81], bytes(range(81)))

    def test_reserved_code_fails_without_creating_package(self):
        bad = bytearray(self.data)
        bad[self.start + 2] = 3
        self.model.write_bytes(bad)
        self.lock.write_text(json.dumps({'size_bytes': len(bad), 'sha256': hashlib.sha256(bad).hexdigest()}))
        for codec in ('raw', 'bslut4'):
            path = self.root / codec
            self.assertNotEqual(self.package(codec, path).returncode, 0)
            self.assertFalse(path.exists())

    def test_mismatched_source_and_existing_output_are_preserved(self):
        output = self.root / 'package'
        output.write_bytes(b'prior')
        self.assertNotEqual(self.package('raw', output).returncode, 0)
        self.assertEqual(output.read_bytes(), b'prior')
        self.model.write_bytes(self.data[:-1])
        missing = self.root / 'missing'
        self.assertNotEqual(self.package('raw', missing).returncode, 0)
        self.assertFalse(missing.exists())


if __name__ == '__main__':
    unittest.main()
