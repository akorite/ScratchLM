#!/usr/bin/env python3
"""Derive the qwen35 pre-tokenizer's code point classes from the pinned runtime.

llama.cpp ships its own Unidata-derived tables (src/unicode-data.cpp). Newer
assignments in the host's Unicode build would split prompts differently, so the
classes come from that file, not from `unicodedata`. The output is a run-length
table the Scratch scanner binary-searches: each entry is [first code point, class]
and the class holds until the next entry.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

OTHER, LETTER, MARK, NUMBER, SPACE_HERE = 0, 1, 2, 3, 4
CR, LF = 5, 6


def parse(source):
    text = source.read_text()
    block = text.split('unicode_ranges_flags = {', 1)[1].split('};', 1)[0]
    ranges = [(int(a, 16), int(f, 16)) for a, f in
              re.findall(r'\{0x([0-9A-Fa-f]+),\s*0x([0-9A-Fa-f]+)\}', block)]
    whitespace = {int(x, 16) for x in
                  re.findall(r'0x([0-9A-Fa-f]+)', text.split('unicode_set_whitespace = {', 1)[1].split('};', 1)[0])}
    flags = bytearray(0x110000)
    for index in range(len(ranges) - 1):
        start, value = ranges[index]
        flags[start:ranges[index + 1][0]] = bytes([value & 0xFF]) * (ranges[index + 1][0] - start)
    return flags, whitespace


def classify(code, flags, whitespace):
    if code in whitespace:
        if code == 0x0D:
            return CR
        if code == 0x0A:
            return LF
        return SPACE_HERE
    value = flags[code]
    if value & 0x0004:
        return LETTER
    if value & 0x0010:
        return MARK
    if value & 0x0002:
        return NUMBER
    return OTHER


def runs(flags, whitespace):
    table, previous = [], None
    for code in range(0x110000):
        kind = classify(code, flags, whitespace)
        if kind != previous:
            table.append([code, kind])
            previous = kind
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='pinned llama.cpp src/unicode-data.cpp')
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    flags, whitespace = parse(args.source)
    table = runs(flags, whitespace)
    digest = hashlib.sha256(args.source.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'schema': 'bonsai-unicode-classes/v1', 'source_sha256': digest,
                                       'runs': table}, separators=(',', ':')) + '\n')
    print(f'{len(table)} runs from {args.source.name} sha256 {digest[:16]} -> {args.output}')


if __name__ == '__main__':
    main()
