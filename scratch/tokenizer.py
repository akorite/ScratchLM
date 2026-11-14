"""Scratch pre-tokenizer and BPE, the counterpart of tools/pretokenizer.py.

`BS tokenize` mirrors llama.cpp's unicode_regex_split_custom_qwen35 test by
test: contractions, `[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+`, single numbers, the
space-prefixed punctuation run, then the three whitespace rules. Code point
classes come from the run-length table embedded in the project; the merge
table comes from the package as dense lists.

The tokenizer needs the 256 costume-lookup sprites of the text mode to turn a
`letter of` result into a code unit number, so it rides on the text build.
"""


def make(h):
    """Block bodies keyed by procedure name."""
    from scratch import embedding
    var, op, floor, item, setvar, addlist, conditional, repeat, change, length = (
        h[name] for name in ('var', 'op', 'floor', 'item', 'setvar', 'addlist',
                             'conditional', 'repeat', 'change', 'length'))
    add = lambda a, b: op('operator_add', NUM1=a, NUM2=b)
    sub = lambda a, b: op('operator_subtract', NUM1=a, NUM2=b)
    mul = lambda a, b: op('operator_multiply', NUM1=a, NUM2=b)
    div = lambda a, b: op('operator_divide', NUM1=a, NUM2=b)
    mod = lambda a, b: op('operator_mod', NUM1=a, NUM2=b)
    eq = lambda a, b: op('operator_equals', OPERAND1=a, OPERAND2=b)
    lt = lambda a, b: op('operator_lt', OPERAND1=a, OPERAND2=b)
    gt = lambda a, b: op('operator_gt', OPERAND1=a, OPERAND2=b)
    both = lambda a, b: op('operator_and', OPERAND1=a, OPERAND2=b)
    either = lambda a, b: op('operator_or', OPERAND1=a, OPERAND2=b)
    not_ = lambda value: op('operator_not', OPERAND=value)
    textlen = lambda value: op('operator_length', STRING=value)
    until = lambda condition, body: ('control_repeat_until', {'CONDITION': condition}, {}, {'SUBSTACK': body})
    call = lambda name: ('procedures_call', {}, {'PROCCODE': (name, 'true')}, {})
    cp = lambda index: item('BS cp', 'l_cp', index)
    ids = lambda index: item('BS ids', 'l_ids', index)
    cat_class = lambda index: item('BS cat class', 'l_cat_class', index)
    length_of = lambda: length('BS cp', 'l_cp')
    in_range = lambda position: both(gt(position, 0), lt(position, add(length_of(), 1)))
    is_ws = lambda value: either(either(eq(value, 4), eq(value, 5)), eq(value, 6))
    is_letter = lambda value: either(eq(value, 1), eq(value, 2))
    is_newline = lambda value: either(eq(value, 13), eq(value, 10))

    def class_at(position, name):
        """Store the class of the 1-based `position` in the variable `name`; 0 past either end."""
        return [setvar('BS class', 0),
                conditional(in_range(position), [setvar('BS cp at', cp(position)), call('BS class of')]),
                setvar(name, var('BS class'))]

    def read_unit(text, position, destination):
        letter = op('operator_letter_of', LETTER=position, STRING=text)
        return [setvar('BS text needle', op('operator_join', STRING1='u', STRING2=letter)),
                setvar('BS text result', -1),
                ('event_broadcastandwait', {'BROADCAST_INPUT': 'BS lookup'}, {}, {}),
                setvar(destination, var('BS text result'))]

    def lower_of(value, destination='BS lower'):
        return [setvar(destination, value),
                conditional(both(gt(value, 64), lt(value, 91)), [setvar(destination, add(value, 32))])]

    # --- 'BS class of': binary search the run starts, read the class ---------
    def class_of():
        lo, hi, mid = var('BS search lo'), var('BS search hi'), var('BS search mid')
        starts = lambda index: item('BS cat start', 'l_cat_start', index)
        return [
            setvar('BS search lo', 1),
            setvar('BS search hi', add(length('BS cat start', 'l_cat_start'), 1)),
            until(not_(lt(lo, hi)), [
                setvar('BS search mid', floor(div(add(lo, hi), 2))),
                conditional(gt(starts(mid), var('BS cp at')),
                            [setvar('BS search hi', mid)], [setvar('BS search lo', add(mid, 1))]),
            ]),
            setvar('BS class', cat_class(sub(var('BS search lo'), 1))),
        ]

    # --- 'BS pair rank': rank of (BS pair l, BS pair r) or 1e9 --------------
    def pair_rank():
        lo, hi, mid = var('BS search lo'), var('BS search hi'), var('BS search mid')
        scan = var('BS scan index')
        lefts = lambda index: item('BS bpe left', 'l_bpe_left', index)
        return [
            setvar('BS pair found', 1000000000), setvar('BS pair merged', 0),
            setvar('BS search lo', 1),
            setvar('BS search hi', add(length('BS bpe left', 'l_bpe_left'), 1)),
            until(not_(lt(lo, hi)), [
                setvar('BS search mid', floor(div(add(lo, hi), 2))),
                conditional(lt(lefts(mid), var('BS pair l')),
                            [setvar('BS search lo', add(mid, 1))], [setvar('BS search hi', mid)]),
            ]),
            setvar('BS scan index', var('BS search lo')),
            until(either(gt(scan, length('BS bpe left', 'l_bpe_left')), not_(eq(lefts(scan), var('BS pair l')))), [
                conditional(eq(item('BS bpe right', 'l_bpe_right', scan), var('BS pair r')), [
                    setvar('BS pair found', item('BS bpe rank', 'l_bpe_rank', scan)),
                    setvar('BS pair merged', item('BS bpe result', 'l_bpe_result', scan)),
                    setvar('BS scan index', add(length('BS bpe left', 'l_bpe_left'), 1)),
                ]),
                change('BS scan index', 1),
            ]),
        ]

    # --- 'BS encode piece': code points -> bytes -> merges -> BS prompt tokens
    def push_byte(value):
        return addlist(item('BS bpe byte', 'l_bpe_byte', add(value, 1)), 'BS ids', 'l_ids')

    def encode_piece():
        code = var('BS cp at')
        cont = lambda value: add(128, mod(value, 64))
        index, size = var('BS cp index'), var('BS ids count')
        return [
            ('data_deletealloflist', {}, {'LIST': ['BS ids', 'l_ids']}, {}),
            setvar('BS cp index', var('BS piece start')),
            until(not_(lt(index, var('BS piece end'))), [
                setvar('BS cp at', cp(index)),
                conditional(lt(code, 128), [push_byte(code)], [
                    conditional(lt(code, 2048), [
                        push_byte(add(192, floor(div(code, 64)))), push_byte(cont(code))], [
                        conditional(lt(code, 65536), [
                            push_byte(add(224, floor(div(code, 4096)))),
                            push_byte(cont(floor(div(code, 64)))), push_byte(cont(code))], [
                            push_byte(add(240, floor(div(code, 262144)))),
                            push_byte(cont(floor(div(code, 4096)))),
                            push_byte(cont(floor(div(code, 64)))), push_byte(cont(code))])])]),
                change('BS cp index', 1),
            ]),
            # merge the adjacent pair with the smallest rank until none is left
            setvar('BS ids count', length('BS ids', 'l_ids')),
            until(lt(size, 2), [
                setvar('BS best rank', 1000000000), setvar('BS merge at', 0),
                setvar('BS pair position', 1),
                until(gt(var('BS pair position'), sub(size, 1)), [
                    setvar('BS pair l', ids(var('BS pair position'))),
                    setvar('BS pair r', ids(add(var('BS pair position'), 1))),
                    call('BS pair rank'),
                    conditional(lt(var('BS pair found'), var('BS best rank')), [
                        setvar('BS best rank', var('BS pair found')),
                        setvar('BS best merged', var('BS pair merged')),
                        setvar('BS merge at', var('BS pair position')),
                    ]),
                    change('BS pair position', 1),
                ]),
                conditional(eq(var('BS merge at'), 0), [
                    setvar('BS ids count', 1),
                ], [
                    ('data_replaceitemoflist', {'INDEX': var('BS merge at'), 'ITEM': var('BS best merged')},
                     {'LIST': ['BS ids', 'l_ids']}, {}),
                    ('data_deleteoflist', {'INDEX': add(var('BS merge at'), 1)},
                     {'LIST': ['BS ids', 'l_ids']}, {}),
                    change('BS ids count', -1),
                ]),
            ]),
            setvar('BS ids index', 1),
            until(gt(var('BS ids index'), length('BS ids', 'l_ids')), [
                addlist(ids(var('BS ids index')), 'BS prompt tokens', 'l_prompt'),
                addlist(ids(var('BS ids index')), 'BS output', 'l_output'),
                change('BS ids index', 1),
            ]),
        ]

    # --- 'BS tokenize': prompt -> code points -> scanner --------------------
    def read_prompt():
        """Fill `BS cp` with the prompt's code points, combining surrogate pairs."""
        index = var('BS cp index')
        return [
            ('data_deletealloflist', {}, {'LIST': ['BS cp', 'l_cp']}, {}),
            ('data_deletealloflist', {}, {'LIST': ['BS prompt tokens', 'l_prompt']}, {}),
            setvar('BS cp index', 1),
            until(gt(index, textlen(var('BS prompt'))), [
                *read_unit(var('BS prompt'), index, 'BS text unit'),
                setvar('BS cp at', var('BS text unit')),
                conditional(both(gt(var('BS text unit'), 55295), lt(var('BS text unit'), 56320)), [
                    setvar('BS cp at', 65533),
                    conditional(lt(index, textlen(var('BS prompt'))), [
                        *read_unit(var('BS prompt'), add(index, 1), 'BS text next'),
                        conditional(both(gt(var('BS text next'), 56319), lt(var('BS text next'), 57344)), [
                            setvar('BS cp at', add(65536, add(mul(sub(var('BS text unit'), 55296), 1024),
                                                              sub(var('BS text next'), 56320)))),
                            change('BS cp index', 1),
                        ]),
                    ]),
                ], [
                    conditional(both(gt(var('BS text unit'), 56319), lt(var('BS text unit'), 57344)),
                                [setvar('BS cp at', 65533)]),
                ]),
                addlist(var('BS cp at'), 'BS cp', 'l_cp'),
                change('BS cp index', 1),
            ]),
        ]

    def tokenize():
        pos, start = var('BS pos'), var('BS piece start')
        here, ahead = var('BS class here'), var('BS class next')
        hit, scratch = var('BS hit'), var('BS class scan')
        first, second = 'BS lower', 'BS lower2'
        first_value, second_value = var(first), var(second)

        def emit(end):
            return [setvar('BS piece end', end), call('BS encode piece'),
                    setvar('BS piece start', end), setvar('BS pos', end), setvar('BS hit', 1)]

        contraction = conditional(both(eq(cp(pos), 39), lt(pos, length_of())), [
            *lower_of(cp(add(pos, 1)), first),
            conditional(either(either(eq(first_value, 115), eq(first_value, 116)),
                               either(eq(first_value, 109), eq(first_value, 100))),
                        emit(add(pos, 2)), [
                conditional(lt(pos, add(length_of(), -2)), [
                    *lower_of(cp(add(pos, 2)), second),
                    conditional(either(both(eq(first_value, 114), eq(second_value, 101)),
                                       either(both(eq(first_value, 118), eq(second_value, 101)),
                                              both(eq(first_value, 108), eq(second_value, 108)))),
                                emit(add(pos, 3))),
                ]),
            ]),
        ])

        letters = conditional(both(not_(is_newline(cp(pos))), not_(eq(here, 3))), [
            conditional(either(is_letter(here), is_letter(ahead)), [
                change('BS pos', 1),
                *class_at(pos, 'BS class scan'),
                until(not_(is_letter(scratch)), [change('BS pos', 1), *class_at(pos, 'BS class scan')]),
                setvar('BS piece end', pos), call('BS encode piece'),
                setvar('BS piece start', pos), setvar('BS hit', 1),
            ]),
        ])

        numbers = conditional(eq(here, 3), [change('BS pos', 1), *emit(pos)])

        punctuation = conditional(both(eq(scratch, 0), in_range(pos)), [
            conditional(eq(cp(pos), 32), [change('BS pos', 1)]),
            *class_at(pos, 'BS class scan'),
            until(either(not_(eq(scratch, 0)), not_(in_range(pos))), [
                change('BS pos', 1), *class_at(pos, 'BS class scan')]),
            until(not_(is_newline(cp(pos))), [change('BS pos', 1)]),
            *emit(pos),
        ])

        whitespace = [
            setvar('BS ws count', 0), setvar('BS nl end', 0),
            *class_at(pos, 'BS class scan'),
            until(not_(is_ws(scratch)), [
                conditional(is_newline(cp(add(pos, var('BS ws count')))), [setvar('BS nl end', add(add(pos, var('BS ws count')), 1))]),
                change('BS ws count', 1),
                *class_at(add(pos, var('BS ws count')), 'BS class scan'),
            ]),
            conditional(gt(var('BS nl end'), 0), [
                setvar('BS pos', var('BS nl end')), *emit(pos),
            ], [
                conditional(both(gt(var('BS ws count'), 1), in_range(add(pos, var('BS ws count')))), [
                    setvar('BS pos', sub(add(pos, var('BS ws count')), 1)), *emit(pos),
                ], [
                    conditional(gt(var('BS ws count'), 0), [
                        setvar('BS pos', add(pos, var('BS ws count'))), *emit(pos),
                    ]),
                ]),
            ]),
        ]

        single = conditional(eq(hit, 0), [change('BS pos', 1), *emit(pos)])
        tables = []
        for req, tensor, name, list_id, count in (
                ('tok-byte', 'prism.bpe.byte', 'BS bpe byte', 'l_bpe_byte', 256),
                ('tok-left', 'prism.bpe.left', 'BS bpe left', 'l_bpe_left', 247587),
                ('tok-right', 'prism.bpe.right', 'BS bpe right', 'l_bpe_right', 247587),
                ('tok-rank', 'prism.bpe.rank', 'BS bpe rank', 'l_bpe_rank', 247587),
                ('tok-result', 'prism.bpe.result', 'BS bpe result', 'l_bpe_result', 247587)):
            tables += embedding.load_dense(h, req, tensor, name, list_id, count, [])
        return [
            *tables,
            *read_prompt(), setvar('BS pos', 1), setvar('BS piece start', 1),
            until(gt(pos, length_of()), [
                setvar('BS hit', 0),
                *class_at(pos, 'BS class here'), *class_at(add(pos, 1), 'BS class next'),
                contraction,
                conditional(eq(hit, 0), [
                    letters,
                    conditional(eq(hit, 0), [
                        numbers,
                        conditional(eq(hit, 0), [
                            conditional(eq(cp(pos), 32), [setvar('BS class scan', ahead)],
                                        [setvar('BS class scan', here)]),
                            punctuation,
                            conditional(eq(hit, 0), [
                                *whitespace,
                                single,
                            ]),
                        ]),
                    ]),
                ]),
            ]),
        ]

    return {'BS class of': class_of(), 'BS pair rank': pair_rank(),
            'BS encode piece': encode_piece(), 'BS tokenize': tokenize()}
