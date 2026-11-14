#!/usr/bin/env python3
"""Build a verified, single-file BonsaiScratch model package."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import tempfile

from inspect_gguf import CHUNK, DEFAULT_LOCK, InvalidGGUF, inspect

MAGIC = b'BNSPKG1\0'
GROUP = 34
GROUPS_PER_CHUNK = CHUNK // GROUP


def _tables():
    encode = bytearray(256)
    invalid = bytearray(256)
    decode = bytearray(256)
    for byte in range(256):
        digits = [(byte >> (2 * i)) & 3 for i in range(4)]
        invalid[byte] = 1 if 3 in digits else 0
        if not invalid[byte]:
            code = digits[0] + 3 * digits[1] + 9 * digits[2] + 27 * digits[3]
            encode[byte] = code
            decode[code] = byte
    return bytes(encode), bytes(decode), bytes(invalid)


ENCODE, DECODE, INVALID = _tables()

def _tables5():
    t5 = bytearray([255]) * 1024
    ttail = bytearray([255]) * 64
    for key in range(1024):
        d = [(key >> (2 * i)) & 3 for i in range(5)]
        if 3 not in d:
            t5[key] = d[0] + 3 * d[1] + 9 * d[2] + 27 * d[3] + 81 * d[4]
    for key in range(64):
        d = [(key >> (2 * i)) & 3 for i in range(3)]
        if 3 not in d:
            ttail[key] = d[0] + 3 * d[1] + 9 * d[2] + 108
    d_low4, d_t4, d_low3, d_hi2, d_low2, d_hi3, d_t0, d_hi4 = ([bytearray(243) for _ in range(8)])
    for idx in range(243):
        x = idx
        trits = []
        for _ in range(5):
            trits.append(x % 3)
            x //= 3
        t0, t1, t2, t3, t4 = trits
        d_low4[idx] = t0 | (t1 << 2) | (t2 << 4) | (t3 << 6)
        d_t4[idx] = t4
        d_low3[idx] = t0 | (t1 << 2) | (t2 << 4)
        d_hi2[idx] = t3 | (t4 << 2)
        d_low2[idx] = t0 | (t1 << 2)
        d_hi3[idx] = t2 | (t3 << 2) | (t4 << 4)
        d_t0[idx] = t0
        d_hi4[idx] = t1 | (t2 << 2) | (t3 << 4) | (t4 << 6)
    d_tail = bytearray(256)
    for idx in range(108, 135):
        x = idx - 108
        d_tail[idx] = (x % 3) | ((x // 3 % 3) << 2) | ((x // 9) << 4)
    return (bytes(t5), bytes(ttail),
            bytes(d_low4), bytes(d_t4), bytes(d_low3), bytes(d_hi2),
            bytes(d_low2), bytes(d_hi3), bytes(d_t0), bytes(d_hi4), bytes(d_tail))


(T5_BASE, T5_TAIL,
 D5_LOW4, D5_T4, D5_LOW3, D5_HI2, D5_LOW2, D5_HI3, D5_T0, D5_HI4, D5_TAIL) = _tables5()


def encode5(record):
    """Recode one 34-byte PQ2_0 record as 26 base-3 index bytes plus spare."""
    out = bytearray(34)
    out[0] = record[0]
    out[1] = record[1]
    t5, ttail = T5_BASE, T5_TAIL
    j = 2
    for k in range(2, 32, 5):
        b0, b1, b2, b3, b4 = record[k], record[k + 1], record[k + 2], record[k + 3], record[k + 4]
        i0 = t5[b0 | ((b1 & 3) << 8)]
        i1 = t5[(b1 >> 2) | ((b2 & 15) << 6)]
        i2 = t5[(b2 >> 4) | ((b3 & 63) << 4)]
        i3 = t5[(b3 >> 6) | (b4 << 2)]
        if i0 == 255 or i1 == 255 or i2 == 255 or i3 == 255:
            raise InvalidGGUF('PQ2_0 contains reserved code 3')
        out[j], out[j + 1], out[j + 2], out[j + 3] = i0, i1, i2, i3
        j += 4
    b30, b31 = record[32], record[33]
    i24 = t5[b30 | ((b31 & 3) << 8)]
    i25 = ttail[b31 >> 2]
    if i24 == 255 or i25 == 255:
        raise InvalidGGUF('PQ2_0 contains reserved code 3')
    out[26] = i24
    out[27] = i25
    return bytes(out)


def decode5(record):
    """Invert encode5; rejects out-of-range indices, tail range, nonzero spare."""
    if record[28] or record[29] or record[30] or record[31] or record[32] or record[33]:
        raise InvalidGGUF('invalid bslut5 record')
    for k in range(2, 27):
        if record[k] > 242:
            raise InvalidGGUF('invalid bslut5 record')
    if not 108 <= record[27] <= 134:
        raise InvalidGGUF('invalid bslut5 record')
    out = bytearray(34)
    out[0] = record[0]
    out[1] = record[1]
    low4, t4, low3, hi2, low2, hi3, t0, hi4 = (
        D5_LOW4, D5_T4, D5_LOW3, D5_HI2, D5_LOW2, D5_HI3, D5_T0, D5_HI4)
    j = 2
    for k in range(2, 26, 4):
        i0, i1, i2, i3 = record[k], record[k + 1], record[k + 2], record[k + 3]
        out[j] = low4[i0]
        out[j + 1] = t4[i0] | (low3[i1] << 2)
        out[j + 2] = hi2[i1] | (low2[i2] << 4)
        out[j + 3] = hi3[i2] | (t0[i3] << 6)
        out[j + 4] = hi4[i3]
        j += 5
    i24, i25 = record[26], record[27]
    out[32] = low4[i24]
    out[33] = t4[i24] | (D5_TAIL[i25] << 2)
    return bytes(out)


def recode(data, table, validate=False):
    result = bytearray(data)
    for offset in range(0, len(result), GROUP):
        values = result[offset + 2:offset + GROUP]
        if validate and 1 in values.translate(INVALID):
            raise InvalidGGUF('PQ2_0 contains reserved code 3')
        result[offset + 2:offset + GROUP] = values.translate(table)
    return result


def recode5(data):
    result = bytearray(data)
    t5, ttail = T5_BASE, T5_TAIL
    for o in range(0, len(result), GROUP):
        j = o + 2
        for k in range(o + 2, o + 32, 5):
            b0, b1, b2, b3, b4 = result[k], result[k + 1], result[k + 2], result[k + 3], result[k + 4]
            i0 = t5[b0 | ((b1 & 3) << 8)]
            i1 = t5[(b1 >> 2) | ((b2 & 15) << 6)]
            i2 = t5[(b2 >> 4) | ((b3 & 63) << 4)]
            i3 = t5[(b3 >> 6) | (b4 << 2)]
            if i0 == 255 or i1 == 255 or i2 == 255 or i3 == 255:
                raise InvalidGGUF('PQ2_0 contains reserved code 3')
            result[j], result[j + 1], result[j + 2], result[j + 3] = i0, i1, i2, i3
            j += 4
        b30, b31 = result[o + 32], result[o + 33]
        i24 = t5[b30 | ((b31 & 3) << 8)]
        i25 = ttail[b31 >> 2]
        if i24 == 255 or i25 == 255:
            raise InvalidGGUF('PQ2_0 contains reserved code 3')
        result[o + 26] = i24
        result[o + 27] = i25
        result[o + 28] = result[o + 29] = result[o + 30] = result[o + 31] = result[o + 32] = result[o + 33] = 0
    return result


def decode5_chunk(data):
    result = bytearray(data)
    low4, t4, low3, hi2, low2, hi3, t0, hi4, dtail = (
        D5_LOW4, D5_T4, D5_LOW3, D5_HI2, D5_LOW2, D5_HI3, D5_T0, D5_HI4, D5_TAIL)
    for o in range(0, len(result), GROUP):
        if result[o + 28] or result[o + 29] or result[o + 30] or result[o + 31] or result[o + 32] or result[o + 33]:
            raise InvalidGGUF('invalid bslut5 record')
        for k in range(o + 2, o + 27):
            if result[k] > 242:
                raise InvalidGGUF('invalid bslut5 record')
        if not 108 <= result[o + 27] <= 134:
            raise InvalidGGUF('invalid bslut5 record')
        # Read indices before overwriting the 32-byte payload in place
        idx = bytes(result[o + 2:o + 28])
        j = o + 2
        for k in range(0, 24, 4):
            i0, i1, i2, i3 = idx[k], idx[k + 1], idx[k + 2], idx[k + 3]
            result[j] = low4[i0]
            result[j + 1] = t4[i0] | (low3[i1] << 2)
            result[j + 2] = hi2[i1] | (low2[i2] << 4)
            result[j + 3] = hi3[i2] | (t0[i3] << 6)
            result[j + 4] = hi4[i3]
            j += 5
        i24, i25 = idx[24], idx[25]
        result[o + 32] = low4[i24]
        result[o + 33] = t4[i24] | (dtail[i25] << 2)
    return result

def digest(file):
    h = hashlib.sha256()
    file.seek(0)
    for chunk in iter(lambda: file.read(CHUNK), b''):
        h.update(chunk)
    return h.hexdigest()


def copy_exact(source, target, length, codec=None):
    while length:
        size = min(length, GROUPS_PER_CHUNK * GROUP if codec else CHUNK)
        data = source.read(size)
        if len(data) != size:
            raise InvalidGGUF('short source read')
        if codec == 'bslut5':
            data = recode5(data)
        elif codec:
            data = recode(data, ENCODE if codec == 'bslut4' else bytes(range(256)), True)
        target.write(data)
        length -= size


def package(model, output, codec, lock_path):
    if output.exists():
        raise FileExistsError(f'output already exists: {output}')
    lock = json.loads(lock_path.read_text(encoding='utf-8'))
    with model.open('rb') as source, tempfile.TemporaryFile(mode='w+', encoding='utf-8') as metadata:
        size = os.fstat(source.fileno()).st_size
        source_hash = digest(source)
        if size != lock['size_bytes'] or source_hash != lock['sha256']:
            raise InvalidGGUF('model identity does not match lock')
        source.seek(0)
        alignment, header_size, tensors = inspect(source, size, metadata)
        metadata.seek(0)
        values = json.load(metadata)
        payload_offset = header_size
        packed_tensors = []
        for tensor in tensors:
            packed_tensors.append({'name': tensor['name'], 'type_id': tensor['type_id'],
                                   'dimensions': tensor['dimensions'], 'source_offset': tensor['offset'],
                                   'offset': payload_offset, 'length': tensor['size_bytes']})
            payload_offset += tensor['size_bytes']
        extra_tensors = []
        if 'prism.hadamard.sign_widths' in values and 'prism.hadamard.sign_values' in values:
            widths = values['prism.hadamard.sign_widths']['value']
            sign_vals = values['prism.hadamard.sign_values']['value']
            if sum(widths) != len(sign_vals) or any(v not in (-1, 1, -1.0, 1.0) for v in sign_vals):
                raise InvalidGGUF('invalid prism.hadamard sign metadata')
            pos = 0
            for width in widths:
                chunk = sign_vals[pos:pos + width]
                pos += width
                blob = struct.pack(f'<{width}f', *[float(v) for v in chunk])
                name = f'prism.hadamard.signs.{width}'
                packed_tensors.append({'name': name, 'type_id': 0, 'dimensions': [width],
                                       'source_offset': None, 'offset': payload_offset, 'length': len(blob)})
                extra_tensors.append((name, blob))
                payload_offset += len(blob)
        manifest = {'schema': 'bonsaiscratch-package/v1', 'codec': codec, 'endianness': 'little',
                    'source': {'size_bytes': size, 'sha256': source_hash}, 'alignment': alignment,
                    'gguf_header': {'offset': 0, 'length': header_size}, 'metadata': values,
                    'tensors': packed_tensors, 'payload_length': payload_offset}
        header = json.dumps(manifest, ensure_ascii=True, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
        fd, temporary = tempfile.mkstemp(prefix='.bonsai-package-', dir=output.parent)
        try:
            with os.fdopen(fd, 'w+b') as target:
                target.write(MAGIC + struct.pack('<Q', len(header)) + header)
                payload_start = target.tell()
                source.seek(0)
                copy_exact(source, target, header_size)
                for tensor in tensors:
                    source.seek(tensor['offset'])
                    copy_exact(source, target, tensor['size_bytes'], codec if tensor['type_id'] == 142 else None)
                for _, blob in extra_tensors:
                    target.write(blob)
                target.flush()
                target.seek(payload_start)
                source.seek(0)
                remaining = header_size
                while remaining:
                    length = min(remaining, CHUNK)
                    if source.read(length) != target.read(length):
                        raise InvalidGGUF('GGUF header round-trip mismatch')
                    remaining -= length
                for tensor in tensors:
                    source.seek(tensor['offset'])
                    remaining = tensor['size_bytes']
                    while remaining:
                        is_pq = tensor['type_id'] == 142
                        length = min(remaining, GROUPS_PER_CHUNK * GROUP if is_pq else CHUNK)
                        original, packed = source.read(length), target.read(length)
                        if is_pq and codec == 'bslut4':
                            packed = recode(packed, DECODE)
                        elif is_pq and codec == 'bslut5':
                            packed = decode5_chunk(packed)
                        if len(original) != length or original != packed:
                            raise InvalidGGUF(f"tensor round-trip mismatch: {tensor['name']}")
                        remaining -= length
                for name, blob in extra_tensors:
                    if target.read(len(blob)) != blob:
                        raise InvalidGGUF(f'dense sign round-trip mismatch: {name}')
                if digest(source) != source_hash:
                    raise InvalidGGUF('source changed during packaging')
                target.seek(0)
                package_hash = digest(target)
            os.link(temporary, output)
            print(f'{package_hash}  {output}')
        finally:
            os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--codec', required=True, choices=('raw', 'bslut4', 'bslut5'))
    parser.add_argument('--lock', type=Path, default=DEFAULT_LOCK)
    args = parser.parse_args()
    try:
        package(args.model, args.output, args.codec, args.lock)
    except (InvalidGGUF, OSError, ValueError, KeyError) as exc:
        parser.exit(1, f'error: {exc}\n')


if __name__ == '__main__':
    main()
