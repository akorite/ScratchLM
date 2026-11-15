#!/usr/bin/env python3
"""Build the numeric BPE tables the Scratch tokenizer needs from a package.

Scratch string comparison is case-insensitive, so the byte-level pieces and
merge strings cannot be compared as text. This tool converts the package's
`tokenizer.ggml.tokens` / `tokenizer.ggml.merges` into numbers: the token id of
each single byte, and a rank-sorted list of (left, right) pair keys with the
merged token id, matching the order of the merge ranks.
"""
import argparse
import json
import struct
from pathlib import Path


def bytes_to_unicode():
    printable = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) \
        + list(range(ord('®'), ord('ÿ') + 1))
    mapping = {byte: chr(byte) for byte in printable}
    spare = 0
    for byte in range(256):
        if byte not in mapping:
            mapping[byte] = chr(256 + spare)
            spare += 1
    return mapping


def read_package(path):
    with path.open('rb') as source:
        if source.read(8) != b'BNSPKG1\0':
            raise ValueError('not a BonsaiScratch package')
        (length,) = struct.unpack('<Q', source.read(8))
        manifest = json.loads(source.read(length))
    return manifest['metadata']


def build(metadata):
    tokens = metadata['tokenizer.ggml.tokens']['value']
    merges = metadata['tokenizer.ggml.merges']['value']
    vocabulary = len(tokens)
    ids = {}
    for index, piece in enumerate(tokens):
        ids.setdefault(piece, index)
    alphabet = bytes_to_unicode()
    byte_token = []
    for byte in range(256):
        piece = alphabet[byte]
        if piece not in ids:
            raise ValueError(f'byte piece {piece!r} is missing from the vocabulary')
        byte_token.append(ids[piece])
    entries = {}
    for rank, line in enumerate(merges):
        left, right = line.split(' ')
        key = ids[left] * vocabulary + ids[right]
        merged = ids.get(left + right)
        if merged is None:
            raise ValueError(f'merge {line!r} has no vocabulary entry')
        # The first occurrence of a pair is the merge that applies.
        entries.setdefault(key, (rank, merged))
    ordered = sorted(entries.items())
    lefts = [key // vocabulary for key, _ in ordered]
    rights = [key % vocabulary for key, _ in ordered]
    return {'schema': 'bonsai-bpe-tables/v1', 'vocabulary': vocabulary,
            'byte_token': byte_token,
            'merge_keys': [key for key, _ in ordered],
            'merge_rank': [rank for _, (rank, _) in ordered],
            'merge_results': [merged for _, (_, merged) in ordered],
            'merge_left': lefts, 'merge_right': rights}



def bpe(tables, piece_token_ids):
    """Reference implementation of the same merge loop the Scratch code runs."""
    import bisect
    vocabulary = tables['vocabulary']
    keys, ranks, results = tables['merge_keys'], tables['merge_rank'], tables['merge_results']
    ids = list(piece_token_ids)
    while len(ids) > 1:
        best_rank, best_at, best_result = None, None, None
        for position in range(len(ids) - 1):
            key = ids[position] * vocabulary + ids[position + 1]
            at = bisect.bisect_left(keys, key)
            if at < len(keys) and keys[at] == key and (best_rank is None or ranks[at] < best_rank):
                best_rank, best_at, best_result = ranks[at], position, results[at]
        if best_rank is None:
            break
        ids[best_at:best_at + 2] = [best_result]
    return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('package', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    tables = build(read_package(args.package))
    args.output.write_text(json.dumps(tables, separators=(',', ':')) + '\n')
    print(f"{len(tables['merge_keys'])} merges, {len(tables['byte_token'])} byte tokens -> {args.output}")


if __name__ == '__main__':
    main()
