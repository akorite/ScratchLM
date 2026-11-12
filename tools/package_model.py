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


def recode(data, table, validate=False):
    result = bytearray(data)
    for offset in range(0, len(result), GROUP):
        values = result[offset + 2:offset + GROUP]
        if validate and 1 in values.translate(INVALID):
            raise InvalidGGUF('PQ2_0 contains reserved code 3')
        result[offset + 2:offset + GROUP] = values.translate(table)
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
        if codec:
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
                        if codec == 'bslut4' and is_pq:
                            packed = recode(packed, DECODE)
                        if len(original) != length or original != packed:
                            raise InvalidGGUF(f"tensor round-trip mismatch: {tensor['name']}")
                        remaining -= length
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
    parser.add_argument('--codec', required=True, choices=('raw', 'bslut4'))
    parser.add_argument('--lock', type=Path, default=DEFAULT_LOCK)
    args = parser.parse_args()
    try:
        package(args.model, args.output, args.codec, args.lock)
    except (InvalidGGUF, OSError, ValueError, KeyError) as exc:
        parser.exit(1, f'error: {exc}\n')


if __name__ == '__main__':
    main()
