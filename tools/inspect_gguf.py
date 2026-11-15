#!/usr/bin/env python3
"""Inventory a GGUF v3 file; by default require the repository's model lock.

Usage: python tools/inspect_gguf.py MODEL.gguf MANIFEST.json
Use --no-lock only for explicitly untrusted or synthetic GGUF files. Metadata
values retain their GGUF type IDs; nonfinite floats are strings (NaN, Infinity,
-Infinity), so the output is valid JSON. Dimensions are in GGML order: the
first dimension is contiguous and, for matrices, is the input/row width.
"""

import argparse
import codecs
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile


DEFAULT_LOCK = Path(__file__).resolve().parents[1] / 'model-lock.json'
FORMATS = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i',
           6: 'f', 7: 'B', 10: 'Q', 11: 'q', 12: 'd'}
CHUNK = 1024 * 1024


class InvalidGGUF(ValueError):
    pass


class Reader:
    def __init__(self, file, size):
        self.file = file
        self.size = size

    def read(self, count):
        if count > self.size - self.file.tell():
            raise InvalidGGUF('truncated GGUF record')
        data = self.file.read(count)
        if len(data) != count:
            raise InvalidGGUF('truncated GGUF record')
        return data

    def number(self, fmt):
        return struct.unpack('<' + fmt, self.read(struct.calcsize(fmt)))[0]

    def name(self):
        length = self.number('Q')
        # shortcut: identifiers are limited to 1 MiB; upgrade for larger GGUF keys.
        if length > CHUNK:
            raise InvalidGGUF('GGUF identifier too long')
        return self.read(length).decode('utf-8')

    def string_json(self, out):
        length = self.number('Q')
        if length > self.size - self.file.tell():
            raise InvalidGGUF('truncated GGUF string')
        decoder = codecs.getincrementaldecoder('utf-8')('strict')
        out.write('"')
        while length:
            data = self.read(min(length, CHUNK))
            length -= len(data)
            text = decoder.decode(data, final=not length)
            out.write(json.dumps(text, ensure_ascii=True)[1:-1])
        if not decoder.getstate()[0] == b'':
            raise InvalidGGUF('truncated UTF-8 string')
        out.write('"')

    def value_json(self, type_id, out):
        if type_id == 8:
            self.string_json(out)
            return None
        if type_id not in FORMATS:
            raise InvalidGGUF(f'unsupported metadata type {type_id}')
        value = self.number(FORMATS[type_id])
        if type_id == 7:
            if value not in (0, 1):
                raise InvalidGGUF('invalid boolean')
            value = bool(value)
        elif type_id in (6, 12) and not math.isfinite(value):
            value = 'NaN' if math.isnan(value) else ('Infinity' if value > 0 else '-Infinity')
        out.write(json.dumps(value, ensure_ascii=True, allow_nan=False))
        return value


def inspect(file, size, metadata_file):
    reader = Reader(file, size)
    if reader.read(4) != b'GGUF' or reader.number('I') != 3:
        raise InvalidGGUF('expected GGUF version 3')
    tensor_count, metadata_count = reader.number('Q'), reader.number('Q')
    if tensor_count > size // 24 or metadata_count > size // 12:
        raise InvalidGGUF('impossible record count')
    metadata_file.write('{')
    names = set()
    alignment = 32
    for index in range(metadata_count):
        name = reader.name()
        if not name or name in names:
            raise InvalidGGUF('empty or duplicate metadata key')
        names.add(name)
        type_id = reader.number('I')
        if index:
            metadata_file.write(',')
        metadata_file.write(json.dumps(name, ensure_ascii=True) + ':{"type_id":' + str(type_id))
        if type_id == 9:
            element_type = reader.number('I')
            count = reader.number('Q')
            if element_type not in FORMATS and element_type != 8:
                raise InvalidGGUF(f'unsupported array element type {element_type}')
            minimum = 8 if element_type == 8 else struct.calcsize(FORMATS[element_type])
            if count > (size - file.tell()) // minimum:
                raise InvalidGGUF('truncated metadata array')
            metadata_file.write(',"array_type_id":' + str(element_type) + ',"value":[')
            for i in range(count):
                if i:
                    metadata_file.write(',')
                reader.value_json(element_type, metadata_file)
            metadata_file.write(']')
            value = None
        else:
            metadata_file.write(',"value":')
            value = reader.value_json(type_id, metadata_file)
        metadata_file.write('}')
        if name == 'general.alignment':
            if type_id != 4 or not value or value & (value - 1):
                raise InvalidGGUF('invalid general.alignment')
            alignment = value
    metadata_file.write('}')

    tensors = []
    names = set()
    for _ in range(tensor_count):
        name = reader.name()
        if not name or name in names:
            raise InvalidGGUF('empty or duplicate tensor name')
        names.add(name)
        ndim = reader.number('I')
        if not 1 <= ndim <= 4:
            raise InvalidGGUF('invalid tensor dimension count')
        dimensions = [reader.number('Q') for _ in range(ndim)]
        if not all(dimensions):
            raise InvalidGGUF('zero tensor dimension')
        type_id, relative = reader.number('I'), reader.number('Q')
        elements = math.prod(dimensions)
        if type_id == 0:
            length = elements * 4
        elif type_id == 142:
            if dimensions[0] % 128:
                raise InvalidGGUF('PQ2_0 row width is not divisible by 128')
            length = elements // 128 * 34
        else:
            raise InvalidGGUF(f'unsupported tensor type {type_id}')
        if relative % alignment:
            raise InvalidGGUF('unaligned tensor offset')
        tensors.append({'name': name, 'type_id': type_id, 'dimensions': dimensions,
                        'relative_offset': relative, 'size_bytes': length})
    data_offset = (file.tell() + alignment - 1) // alignment * alignment
    if data_offset > size:
        raise InvalidGGUF('truncated tensor data alignment')
    intervals = []
    for tensor in tensors:
        start = data_offset + tensor['relative_offset']
        end = start + tensor['size_bytes']
        if end > size:
            raise InvalidGGUF('truncated tensor payload')
        tensor['offset'] = start
        intervals.append((start, end))
    intervals.sort()
    if any(right[0] < left[1] for left, right in zip(intervals, intervals[1:])):
        raise InvalidGGUF('overlapping tensor payloads')
    return alignment, data_offset, tensors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('output', type=Path)
    lock = parser.add_mutually_exclusive_group()
    lock.add_argument('--lock', type=Path, default=DEFAULT_LOCK, help='identity lock (default: repository model-lock.json)')
    lock.add_argument('--no-lock', action='store_true', help='inspect without model identity verification')
    args = parser.parse_args()
    if args.model.resolve() == args.output.resolve() or (args.output.exists() and os.path.samefile(args.model, args.output)):
        raise InvalidGGUF('output must differ from model')
    if not args.no_lock and (args.lock.resolve() == args.output.resolve() or (args.output.exists() and os.path.samefile(args.lock, args.output))):
        raise InvalidGGUF('output must differ from lock')
    with args.model.open('rb') as model, tempfile.TemporaryFile(mode='w+', encoding='utf-8') as metadata:
        size = os.fstat(model.fileno()).st_size
        if not args.no_lock:
            with args.lock.open(encoding='utf-8') as lock_file:
                expected = json.load(lock_file)
            if size != expected['size_bytes']:
                raise InvalidGGUF('model size does not match lock')
        digest = hashlib.sha256()
        for chunk in iter(lambda: model.read(CHUNK), b''):
            digest.update(chunk)
        sha256 = digest.hexdigest()
        if not args.no_lock and sha256 != expected['sha256']:
            raise InvalidGGUF('model SHA-256 does not match lock')
        model.seek(0)
        alignment, data_offset, tensors = inspect(model, size, metadata)
        fd, temporary = tempfile.mkstemp(prefix='.inspect-gguf-', dir=args.output.parent)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as out:
                out.write('{"version":3,"source":')
                json.dump({'size_bytes': size, 'sha256': sha256}, out, separators=(',', ':'))
                out.write(',"alignment":' + str(alignment) + ',"tensor_data_offset":' + str(data_offset))
                out.write(',"dimension_order":"ggml: dimension 0 is contiguous and is the matrix input/row width"')
                out.write(',"metadata":')
                metadata.seek(0)
                shutil.copyfileobj(metadata, out, CHUNK)
                out.write(',"tensors":')
                json.dump(tensors, out, ensure_ascii=True, separators=(',', ':'), allow_nan=False)
                out.write('}\n')
            os.replace(temporary, args.output)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


if __name__ == '__main__':
    try:
        main()
    except (InvalidGGUF, OSError, UnicodeError, KeyError, ValueError) as exc:
        print(f'inspect_gguf: {exc}', file=sys.stderr)
        sys.exit(1)
