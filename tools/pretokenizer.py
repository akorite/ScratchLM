#!/usr/bin/env python3
"""Reference port of llama.cpp's qwen35 pre-tokenizer and BPE pass.

The Scratch tokenizer has to reproduce the pinned runtime's splits exactly, so
this module mirrors `unicode_regex_split_custom_qwen35` statement for
statement. The C++ scanner is deterministic (no backtracking), which is what
makes a block-by-block port possible; the Scratch code in `scratch/tokenizer.py`
follows the same order of tests.
"""
import bisect
import json
from pathlib import Path

OTHER, LETTER, MARK, NUMBER, SPACE, CR, LF = 0, 1, 2, 3, 4, 5, 6
CLASSES = None


def load_classes(path=None):
    """Install the pinned run-length table; without it the scanner has no classes."""
    global CLASSES
    path = path or Path(__file__).resolve().parents[1] / 'fixtures/unicode/qwen35-classes.json'
    table = json.loads(Path(path).read_text())
    CLASSES = ([row[0] for row in table['runs']], [row[1] for row in table['runs']])
    return CLASSES


def class_of(code):
    if CLASSES is None:
        load_classes()
    starts, kinds = CLASSES
    return kinds[bisect.bisect_right(starts, code) - 1]


def flags(code):
    """(letter, mark, number, whitespace, assigned) for a code point."""
    kind = class_of(code)
    return (kind == LETTER, kind == MARK, kind == NUMBER, kind in (SPACE, CR, LF), True)


def tolower(code):
    return code + 32 if 0x41 <= code <= 0x5A else code


def pretokenize(code_points):
    """Split code points exactly as the pinned qwen35 scanner does."""
    length = len(code_points)
    out_of_range = None
    get = lambda position: code_points[position] if 0 <= position < length else out_of_range
    get_flags = lambda position: flags(get(position)) if get(position) is not None else (False,) * 5
    pieces, position = [], 0
    while position < length:
        cpt = get(position)
        letter, mark, number, space, assigned = get_flags(position)
        # (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if cpt == 0x27 and position + 1 < length:
            first = tolower(get(position + 1))
            if first in (ord('s'), ord('t'), ord('m'), ord('d')):
                pieces.append(code_points[position:position + 2])
                position += 2
                continue
            if position + 2 < length:
                second = tolower(get(position + 2))
                if (first, second) in ((ord('r'), ord('e')), (ord('v'), ord('e')), (ord('l'), ord('l'))):
                    pieces.append(code_points[position:position + 3])
                    position += 3
                    continue
        # [^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+
        if not (cpt in (13, 10) or number):
            if letter or mark or get_flags(position + 1)[0] or get_flags(position + 1)[1]:
                start = position
                position += 1
                while get_flags(position)[0] or get_flags(position)[1]:
                    position += 1
                pieces.append(code_points[start:position])
                continue
        # \\p{N}
        if number:
            pieces.append(code_points[position:position + 1])
            position += 1
            continue
        #  ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*
        other_flags = get_flags(position + 1) if cpt == 0x20 else (letter, mark, number, space, assigned)
        if not (other_flags[3] or other_flags[0] or other_flags[1] or other_flags[2]) and assigned:
            start = position
            position += 1 if cpt == 0x20 else 0
            while True:
                current = get_flags(position)
                if current[3] or current[0] or current[1] or current[2] or not current[4]:
                    break
                position += 1
            while get(position) in (13, 10):
                position += 1
            pieces.append(code_points[start:position])
            continue
        # \\s*[\\r\\n]+ / \\s+(?!\\S) / \\s+
        whitespaces = 0
        last_newline_end = 0
        while get_flags(position + whitespaces)[3]:
            if get(position + whitespaces) in (13, 10):
                last_newline_end = position + whitespaces + 1
            whitespaces += 1
        if last_newline_end > 0:
            start = position
            position = last_newline_end
            pieces.append(code_points[start:position])
            continue
        if whitespaces > 1 and get(position + whitespaces) is not None:
            start = position
            position += whitespaces - 1
            pieces.append(code_points[start:position])
            continue
        if whitespaces > 0:
            start = position
            position += whitespaces
            pieces.append(code_points[start:position])
            continue
        pieces.append(code_points[position:position + 1])
        position += 1
    return pieces
