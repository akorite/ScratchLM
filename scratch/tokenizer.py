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

    def load_strings(h_, key, name, list_id, request):
        """Copy a metadata string array into a list through the storage extension."""
        eq = lambda a, b: op('operator_equals', OPERAND1=a, OPERAND2=b)
        status = ('bonsailoader_status', {'ID': request}, {}, False)
        copied = ('bonsailoader_copied', {'ID': request}, {}, False)
        return [('data_deletealloflist', {}, {'LIST': [name, list_id]}, {}),
                ('bonsailoader_loadStrings', {'KEY': key, 'DEST': list_id, 'ID': request}, {}, {}),
                until(not_(eq(status, 'pending')), [('control_wait', {'DURATION': 0}, {}, {})])]

    # --- 'BS detokenize': token ids -> text ---------------------------------
    def detokenize():
        """Map ids through the loaded vocabulary, undo the byte alphabet, decode UTF-8."""
        index, unit = var('BS detok index'), var('BS detok unit')
        byte, code = var('BS detok byte'), var('BS detok code')
        position, cursor = var('BS detok position'), var('BS byte index')
        text = var('BS text')
        piece = var('BS detok piece')
        data = lambda at: item('BS bytes', 'l_bytes', at)
        continuation = lambda at: sub(data(at), 128)
        bytes_of = lambda: length('BS bytes', 'l_bytes')

        def inverse_alphabet():
            """The byte a byte-level character stands for; the alphabet is a fixed bijection."""
            # 0x21..0x7E, 0xA1..0xAC and 0xAE..0xFF stand for themselves; everything else
            # is written as 0x100 + the position among the remaining bytes.
            identity = either(either(both(gt(unit, 32), lt(unit, 127)), both(gt(unit, 160), lt(unit, 173))),
                              both(gt(unit, 173), lt(unit, 256)))
            return [conditional(identity, [setvar('BS detok byte', unit)], [
                conditional(eq(unit, 173), [setvar('BS detok byte', 160)], [
                    setvar('BS detok slot', sub(unit, 256)),
                    conditional(lt(var('BS detok slot'), 33), [
                        setvar('BS detok byte', var('BS detok slot'))], [
                        conditional(lt(var('BS detok slot'), 67), [
                            setvar('BS detok byte', add(127, sub(var('BS detok slot'), 33)))], [
                            setvar('BS detok byte', 173)])])])])]

        def append_text():
            return [conditional(lt(code, 65536), [
                setvar('BS text', op('operator_join', STRING1=text,
                    STRING2=item('BS units', 'l_units', add(code, 1))))], [
                setvar('BS text', op('operator_join', STRING1=text, STRING2=op('operator_join',
                    STRING1=item('BS units', 'l_units', add(add(55296, floor(div(sub(code, 65536), 1024))), 1)),
                    STRING2=item('BS units', 'l_units',
                                 add(add(56320, mod(sub(code, 65536), 1024)), 1)))))])]

        def read_piece_bytes():
            return [
                ('data_deletealloflist', {}, {'LIST': ['BS bytes', 'l_bytes']}, {}),
                setvar('BS detok index', var('BS render start')),
                until(gt(index, length('BS out tokens', 'l_out_tokens')), [
                    setvar('BS detok piece',
                           item('BS vocab', 'l_vocab',
                                add(item('BS out tokens', 'l_out_tokens', index), 1))),
                    setvar('BS detok position', 1),
                    until(gt(position, textlen(piece)), [
                        *read_unit(piece, position, 'BS detok unit'),
                        *inverse_alphabet(),
                        addlist(byte, 'BS bytes', 'l_bytes'),
                        change('BS detok position', 1)]),
                    change('BS detok index', 1)])]

        def decode_bytes():
            return [
                setvar('BS text', ''), setvar('BS byte index', 1),
                until(gt(cursor, bytes_of()), [
                    setvar('BS detok byte', data(cursor)),
                    conditional(lt(byte, 128), [
                        setvar('BS detok code', byte), change('BS byte index', 1)], [
                        conditional(lt(byte, 224), [
                            setvar('BS detok code', add(mul(sub(byte, 192), 64), continuation(add(cursor, 1)))),
                            change('BS byte index', 2)], [
                            conditional(lt(byte, 240), [
                                setvar('BS detok code', add(add(mul(sub(byte, 224), 4096),
                                    mul(continuation(add(cursor, 1)), 64)), continuation(add(cursor, 2)))),
                                change('BS byte index', 3)], [
                                setvar('BS detok code', add(add(add(
                                    mul(sub(byte, 240), 262144),
                                    mul(continuation(add(cursor, 1)), 4096)),
                                    mul(continuation(add(cursor, 2)), 64)), continuation(add(cursor, 3)))),
                                change('BS byte index', 4)])])]),
                    *append_text()])]

        return {'load': load_strings(h, 'tokenizer.ggml.tokens', 'BS vocab', 'l_vocab', 'detok-vocab'),
                'render': [*read_piece_bytes(), *decode_bytes()]}

    # --- 'BS generate': prompt -> tokens -> decode steps -> text ------------
    def generate():
        """One flag run: tokenize the prompt, then one decode broadcast per token."""
        index = var('BS generate index')
        eos = var('BS greedy token')
        save, stop = var('BS prompt saved'), var('BS stop')
        segment = lambda text: [
            setvar('BS prompt', text),
            ('procedures_call', {}, {'PROCCODE': ('BS tokenize', 'true')}, {}),
            setvar('BS generate index', 1),
            until(gt(index, length('BS prompt tokens', 'l_prompt')), [
                addlist(item('BS prompt tokens', 'l_prompt', index), 'BS out tokens', 'l_out_tokens'),
                change('BS generate index', 1)])]
        special = lambda token_id: [addlist(token_id, 'BS out tokens', 'l_out_tokens')]
        body = [
            # The locked template wraps the prompt so the model sees the turn markers.
            conditional(eq(var('BS template'), 1), [
                *special(248045), *segment('user\n'), *segment(save), *special(248046),
                *segment('\n'), *special(248045), *segment('assistant\n')],
                [*segment(save)]),
            setvar('BS prompt', save),
            ('procedures_call', {}, {'PROCCODE': ('BS detok load', 'true')}, {}),
            setvar('BS step', 0), setvar('BS generate index', 0), setvar('BS render start', 1),
            until(either(eq(stop, 1), not_(lt(index, var('BS max tokens')))), [
                ('event_broadcastandwait', {'BROADCAST_INPUT': 'BS decode step'}, {}, {}),
                change('BS generate index', 1),
                conditional(eq(eos, var('BS eos token')), [setvar('BS stop', 1)], [
                    addlist(eos, 'BS out tokens', 'l_out_tokens'),
                    # only the generated tail is shown, not the templated prompt
                    setvar('BS render start', sub(add(length('BS out tokens', 'l_out_tokens'), 1), index)),
                    ('procedures_call', {}, {'PROCCODE': ('BS detok render', 'true')}, {}),
                    ('event_broadcastandwait', {'BROADCAST_INPUT': 'BS show'}, {}, {}),
                ]),
            ]),
        ]
        guard = conditional(eq(var('BS prompt saved'), ''), [
            setvar('BS text', ''), setvar('BS stop', 1),
        ], [setvar('BS stop', 0), *body])
        return [
            conditional(eq(var('BS ask'), 1), [
                ('sensing_askandwait', {'QUESTION': 'Prompt'}, {}, {}),
                setvar('BS prompt', ('sensing_answer', {}, {}, False))]),
            setvar('BS prompt saved', var('BS prompt')),
            ('data_deletealloflist', {}, {'LIST': ['BS out tokens', 'l_out_tokens']}, {}),
            guard,
        ]

    detok = detokenize()
    return {'BS class of': class_of(), 'BS pair rank': pair_rank(),
            'BS encode piece': encode_piece(), 'BS tokenize': tokenize(),
            'BS detok load': detok['load'], 'BS detok render': detok['render'],
            'BS generate': generate()}
