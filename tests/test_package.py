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
                token_info = metadata['tokenizer_arrays']['tokenizer.ggml.tokens']
                self.assertEqual(token_info['count'], 2)
                self.assertEqual(token_info['sha256'], hashlib.sha256(b'["Hello","\xe4\xb8\x96\xe7\x95\x8c"]').hexdigest())
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

    def test_bslut5_records_carry_index_bytes_and_round_trip(self):
        path = self.root / 'bslut5'
        result = self.package('bslut5', path)
        self.assertEqual(result.returncode, 0, result.stderr)
        metadata, payload = self.unpack(path)
        self.assertEqual(metadata['codec'], 'bslut5')
        weights = metadata['tensors'][0]
        w = payload[weights['offset']:weights['offset'] + weights['length']]
        for record in (w[:34], w[34:68]):
            indices = record[2:28]
            self.assertEqual(record[28:], bytes(6))
            self.assertTrue(all(index <= 242 for index in indices[:25]))
            self.assertTrue(108 <= indices[25] <= 134)
        # 0x24 = fields (0,1,2,0); 5-field groups misalign with 4-field code bytes
        # -> index period 4: 21,88,191,63; tail fields (1,2,0) -> 1+6+108 = 115
        self.assertEqual(list(w[2:27]), [21, 88, 191, 63] * 6 + [21])
        self.assertEqual(w[27], 115)
        self.assertEqual(w[36:61], bytes([242]) * 25)
        self.assertEqual(w[61], 134)

    def test_bslut5_all_patterns_and_reserved_code(self):
        name = b'weights'
        tensor = struct.pack('<Q', len(name)) + name + struct.pack('<IQQIQ', 2, 128, 3, 142, 0)
        header = b'GGUF' + struct.pack('<IQQ', 3, 1, 0) + tensor
        header += bytes((-len(header)) % 32)
        raw = [((n % 3) | ((n // 3 % 3) << 2) | ((n // 9 % 3) << 4) | ((n // 27) << 6)) for n in range(81)]
        raw.extend([0] * (96 - len(raw)))
        model = header + b''.join(b'\0\0' + bytes(raw[i:i + 32]) for i in range(0, 96, 32))
        self.model.write_bytes(model)
        self.lock.write_text(json.dumps({'size_bytes': len(model), 'sha256': hashlib.sha256(model).hexdigest()}))
        output = self.root / 'all-patterns-5'
        self.assertEqual(self.package('bslut5', output).returncode, 0)
        manifest, payload = self.unpack(output)
        packed = payload[manifest['tensors'][0]['offset']:]
        # record 0 holds trit patterns 0..127: group g index = pattern ids summed in base 3
        indices = packed[2:28]
        self.assertEqual(indices[0], 81)   # trits 0-4 = (0,0,0,0, field0 of byte 1 = 1) -> 81
        self.assertTrue(108 <= indices[25] <= 134)
        bad = bytearray(model)
        bad[-34 + 2] = 3  # field value 3 inside a PQ2 record
        self.model.write_bytes(bytes(bad))
        self.lock.write_text(json.dumps({'size_bytes': len(bad), 'sha256': hashlib.sha256(bad).hexdigest()}))
        self.assertNotEqual(self.package('bslut5', self.root / 'bad5').returncode, 0)
        self.assertFalse((self.root / 'bad5').exists())

    def test_mismatched_source_and_existing_output_are_preserved(self):
        output = self.root / 'package'
        output.write_bytes(b'prior')
        self.assertNotEqual(self.package('raw', output).returncode, 0)
        self.assertEqual(output.read_bytes(), b'prior')
        self.model.write_bytes(self.data[:-1])
        missing = self.root / 'missing'
        self.assertNotEqual(self.package('raw', missing).returncode, 0)
        self.assertFalse(missing.exists())

    def test_bslut5_full_243_and_decode_rejections(self):
        sys.path.insert(0, str(SCRIPT.parent))
        import package_model
        # Exercise all 243 5-trit patterns in aligned (g=0) and unaligned (g=1, 24) positions plus all 27 tail patterns.
        for pat in range(243):
            trits = [1] * 128
            x = pat
            for i in range(5):
                trits[i] = x % 3
                x //= 3
            x = pat
            for i in range(5):
                trits[5 + i] = x % 3
                x //= 3
            x = pat
            for i in range(5):
                trits[120 + i] = x % 3
                x //= 3
            t = pat % 27
            trits[125], trits[126], trits[127] = t % 3, (t // 3) % 3, t // 9
            raw = bytearray(b'\x00\x3c')
            for j in range(32):
                raw.append(trits[4 * j] | (trits[4 * j + 1] << 2) | (trits[4 * j + 2] << 4) | (trits[4 * j + 3] << 6))
            enc = package_model.encode5(bytes(raw))
            self.assertEqual(package_model.decode5(enc), bytes(raw))
            self.assertEqual(package_model.recode5(bytes(raw)), enc)
            self.assertEqual(package_model.decode5_chunk(enc), bytes(raw))
        good = bytearray(package_model.encode5(b'\x00\x3c' + bytes([0x55] * 32)))
        for bad_pos, bad_val in [(28, 1), (33, 1), (2, 243), (26, 255), (27, 107), (27, 135)]:
            corrupt = bytearray(good)
            corrupt[bad_pos] = bad_val
            with self.assertRaises(package_model.InvalidGGUF):
                package_model.decode5(bytes(corrupt))

    def test_bpe_tables_emitted_as_dense_tensors(self):
        def kv_str_array(key, values):
            k = key.encode()
            body = b''.join(struct.pack('<Q', len(v.encode())) + v.encode() for v in values)
            return struct.pack('<Q', len(k)) + k + struct.pack('<IIQ', 9, 8, len(values)) + body
        name = b'w'
        tensor = struct.pack('<Q', len(name)) + name + struct.pack('<IQQIQ', 2, 128, 1, 142, 0)
        import sys
        sys.path.insert(0, str(SCRIPT.parent))
        from tokenizer_tables import bytes_to_unicode
        alphabet = bytes_to_unicode()
        tokens = [alphabet[b] for b in range(256)] + ['ab', 'ba']
        kvs = kv_str_array('tokenizer.ggml.tokens', tokens) + kv_str_array('tokenizer.ggml.merges', ['a b'])
        header = b'GGUF' + struct.pack('<IQQ', 3, 1, 2) + kvs + tensor
        header += bytes((-len(header)) % 32)
        model = header + b'\x00\x3c' + bytes([0x55] * 32)
        self.model.write_bytes(model)
        self.lock.write_text(json.dumps({'size_bytes': len(model), 'sha256': hashlib.sha256(model).hexdigest()}))
        out = self.root / 'bpe.bsp'
        result = self.package('raw', out)
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest, payload = self.unpack(out)
        by_name = {t['name']: t for t in manifest['tensors']}
        read = lambda name: struct.unpack(f"<{by_name[name]['dimensions'][0]}f",
                                          payload[by_name[name]['offset']:][:by_name[name]['length']])
        self.assertEqual(by_name['prism.bpe.byte']['dimensions'], [256])
        self.assertEqual(read('prism.bpe.byte')[0x61], 0x61)
        self.assertEqual(list(read('prism.bpe.left')), [0x61])
        self.assertEqual(list(read('prism.bpe.right')), [0x62])
        self.assertEqual(list(read('prism.bpe.result')), [256])
        self.assertEqual(list(read('prism.bpe.rank')), [0])

    def test_hadamard_signs_emitted_as_dense_f32_tensors(self):
        def kv_i32_arr(key, vals):
            k = key.encode()
            return struct.pack('<Q', len(k)) + k + struct.pack('<IIQ', 9, 5, len(vals)) + struct.pack(f'<{len(vals)}i', *vals)
        def kv_f32_arr(key, vals):
            k = key.encode()
            return struct.pack('<Q', len(k)) + k + struct.pack('<IIQ', 9, 6, len(vals)) + struct.pack(f'<{len(vals)}f', *vals)
        name = b'w'
        tensor = struct.pack('<Q', len(name)) + name + struct.pack('<IQQIQ', 2, 128, 1, 142, 0)
        kvs = kv_i32_arr('prism.hadamard.sign_widths', [4, 8]) + kv_f32_arr('prism.hadamard.sign_values', [1.0, -1.0] * 6)
        header = b'GGUF' + struct.pack('<IQQ', 3, 1, 2) + kvs + tensor
        header += bytes((-len(header)) % 32)
        model = header + b'\x00\x3c' + bytes([0x55] * 32)
        self.model.write_bytes(model)
        self.lock.write_text(json.dumps({'size_bytes': len(model), 'sha256': hashlib.sha256(model).hexdigest()}))
        out = self.root / 'with-signs.bsp'
        res = self.package('bslut5', out)
        self.assertEqual(res.returncode, 0, res.stderr)
        manifest, payload = self.unpack(out)
        by_name = {t['name']: t for t in manifest['tensors']}
        self.assertEqual(by_name['prism.hadamard.signs.4']['type_id'], 0)
        self.assertEqual(by_name['prism.hadamard.signs.4']['dimensions'], [4])
        s4 = struct.unpack('<4f', payload[by_name['prism.hadamard.signs.4']['offset']:][:16])
        self.assertEqual(s4, (1.0, -1.0, 1.0, -1.0))
        s8 = struct.unpack('<8f', payload[by_name['prism.hadamard.signs.8']['offset']:][:32])
        self.assertEqual(s8, (1.0, -1.0) * 4)


if __name__ == '__main__':
    unittest.main()
