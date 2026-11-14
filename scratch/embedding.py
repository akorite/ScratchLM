"""Visible Scratch inverse embedding lookup and final output_norm blocks."""

from scratch import numeric, transform


def decode_fp16_scale(h, word_expr):
    var, op, floor, item, setvar, conditional = (
        h[name] for name in ('var', 'op', 'floor', 'item', 'setvar', 'conditional'))
    add = lambda a, b: op('operator_add', NUM1=a, NUM2=b)
    mul = lambda a, b: op('operator_multiply', NUM1=a, NUM2=b)
    div = lambda a, b: op('operator_divide', NUM1=a, NUM2=b)
    mod = lambda a, b: op('operator_mod', NUM1=a, NUM2=b)
    eq = lambda a, b: op('operator_equals', OPERAND1=a, OPERAND2=b)
    exponent, mantissa, sign = map(var, ('BS exponent', 'BS mantissa', 'BS sign'))
    normal = mul(sign, mul(add(1024, mantissa), item('BS powers', 'l_powers', add(exponent, 1))))
    subnormal = mul(sign, mul(mantissa, 2**-24))
    return [
        setvar('BS word', word_expr),
        setvar('BS exponent', mod(floor(div(var('BS word'), 1024)), 32)),
        setvar('BS mantissa', mod(var('BS word'), 1024)),
        conditional(op('operator_gt', OPERAND1=var('BS word'), OPERAND2=32767),
                    [setvar('BS sign', -1)], [setvar('BS sign', 1)]),
        conditional(eq(exponent, 31), [setvar('BS dot valid', 0)], [
            conditional(eq(exponent, 0), [setvar('BS weight scale', subnormal)], [setvar('BS weight scale', normal)]),
        ]),
    ]


def dequantize_row(h):
    var, op, floor, item, setvar, addlist, conditional, repeat, change = (
        h[name] for name in ('var', 'op', 'floor', 'item', 'setvar',
                             'addlist', 'conditional', 'repeat', 'change'))
    add = lambda a, b: op('operator_add', NUM1=a, NUM2=b)
    sub = lambda a, b: op('operator_subtract', NUM1=a, NUM2=b)
    mul = lambda a, b: op('operator_multiply', NUM1=a, NUM2=b)
    div = lambda a, b: op('operator_divide', NUM1=a, NUM2=b)
    mod = lambda a, b: op('operator_mod', NUM1=a, NUM2=b)
    eq = lambda a, b: op('operator_equals', OPERAND1=a, OPERAND2=b)
    rem = var('BS lut pattern')
    base = var('BS base')
    wscale = var('BS weight scale')
    emit_step = [
        addlist(mul(sub(mod(rem, base), 1), wscale)),
        setvar('BS lut pattern', floor(div(rem, base))),
    ]
    raw_or_lut4 = [
        setvar('BS base', conditional_base := 4),
    ]
    del raw_or_lut4, conditional_base
    byte_at = item('BS packed bytes', 'l_packed', var('BS byte cur'))
    decode_32 = lambda b: [
        setvar('BS base', b),
        repeat(32, [
            setvar('BS lut pattern', byte_at),
            repeat(4, emit_step),
            change('BS byte cur', 1),
        ]),
    ]
    decode_bslut5 = [
        setvar('BS base', 3),
        repeat(25, [
            setvar('BS lut pattern', byte_at),
            repeat(5, emit_step),
            change('BS byte cur', 1),
        ]),
        setvar('BS lut pattern', sub(byte_at, 108)),
        repeat(3, emit_step),
    ]
    group_body = decode_fp16_scale(h, item('BS weight scales', 'l_scales', var('BS dot group'))) + [
        conditional(eq(var('BS dot valid'), 1), [
            setvar('BS byte cur', add(mul(sub(var('BS dot group'), 1), 32), 1)),
            conditional(eq(var('BS codec'), 0), decode_32(4), [
                conditional(eq(var('BS codec'), 1), decode_32(3), decode_bslut5),
            ]),
        ]),
        change('BS dot group', 1),
    ]
    return [
        setvar('BS dot group', 1),
        setvar('BS dot valid', 1),
        repeat(h['length']('BS weight scales', 'l_scales'), group_body),
    ]


def output_norm(h):
    var, op, item, length, setvar, repeat, change = (
        h[name] for name in ('var', 'op', 'item', 'length', 'setvar', 'repeat', 'change'))
    mul = lambda a, b: op('operator_multiply', NUM1=a, NUM2=b)
    idx = var('BS index')
    scale_weights = [
        setvar('BS index', 1),
        repeat(length('BS output', 'l_output'), [
            setvar('BS f32 input', mul(item('BS output', 'l_output', idx),
                                       item('BS norm weights', 'l_norm_weights', idx))),
            h['call_f32'](),
            transform.replace_item(idx, var('BS f32 output'), 'BS output', 'l_output'),
            change('BS index', 1),
        ]),
    ]
    return numeric.make(5, h) + scale_weights


def make(mode, h, with_benchmark=False):
    if mode not in (14, 15, 16):
        return None
    var, op, floor, length, setvar, addlist, conditional = (
        h[name] for name in ('var', 'op', 'floor', 'length', 'setvar', 'addlist', 'conditional'))
    eq = lambda a, b: op('operator_equals', OPERAND1=a, OPERAND2=b)
    gt = lambda a, b: op('operator_gt', OPERAND1=a, OPERAND2=b)
    mul = lambda a, b: op('operator_multiply', NUM1=a, NUM2=b)

    if mode == 14:
        valid = op('operator_and',
                   OPERAND1=op('operator_and',
                               OPERAND1=eq(length('BS weight scales', 'l_scales'), 16),
                               OPERAND2=eq(length('BS packed bytes', 'l_packed'), 512)),
                   OPERAND2=op('operator_and',
                               OPERAND1=eq(length('BS signs', 'l_signs'), 2048),
                               OPERAND2=op('operator_or', OPERAND1=eq(var('BS codec'), 0),
                                           OPERAND2=op('operator_or', OPERAND1=eq(var('BS codec'), 1),
                                                       OPERAND2=eq(var('BS codec'), 2)))))
        work = dequantize_row(h) + [
            conditional(eq(var('BS dot valid'), 1),
                        transform.hadamard(h) + transform.apply_signs(h),
                        [('data_deletealloflist', {}, {'LIST': ['BS output', 'l_output']}, {}),
                         addlist('nonfinite')]),
        ]
        return [conditional(valid, work, [addlist('invalid dimensions')])]

    if mode == 15:
        if not with_benchmark:
            return [addlist('loader unavailable')]
        codec = ('bonsailoader_codec', {}, {}, False)
        status = ('bonsailoader_status', {'ID': 'emb'}, {}, False)
        copied = ('bonsailoader_copied', {'ID': 'emb'}, {}, False)
        s_status = ('bonsailoader_status', {'ID': 'emb_signs'}, {}, False)
        s_copied = ('bonsailoader_copied', {'ID': 'emb_signs'}, {}, False)
        tid = var('BS token id')
        valid = op('operator_and',
                   OPERAND1=op('operator_and',
                               OPERAND1=gt(tid, -1),
                               OPERAND2=op('operator_and',
                                           OPERAND1=op('operator_lt', OPERAND1=tid, OPERAND2=248320),
                                           OPERAND2=eq(tid, floor(tid)))),
                   OPERAND2=op('operator_or', OPERAND1=eq(codec, 'raw'),
                               OPERAND2=op('operator_or', OPERAND1=eq(codec, 'bslut4'),
                                           OPERAND2=eq(codec, 'bslut5'))))
        set_codec = conditional(eq(codec, 'raw'), [setvar('BS codec', 0)],
                                [conditional(eq(codec, 'bslut4'), [setvar('BS codec', 1)], [setvar('BS codec', 2)])])
        load_row = [
            ('bonsailoader_loadGroups',
             {'ID': 'emb', 'TENSOR': 'token_embd.weight', 'START': mul(tid, 16), 'COUNT': 16,
              'Q': 'l_packed', 'SCALE': 'l_scales'}, {}, {}),
            ('bonsailoader_loadDense',
             {'ID': 'emb_signs', 'TENSOR': 'prism.hadamard.signs.2048', 'START': 0, 'COUNT': 2048,
              'DEST': 'l_signs'}, {}, {}),
            conditional(op('operator_and',
                           OPERAND1=op('operator_and', OPERAND1=eq(status, 'ok'), OPERAND2=eq(copied, 16)),
                           OPERAND2=op('operator_and', OPERAND1=eq(s_status, 'ok'), OPERAND2=eq(s_copied, 2048))),
                        [set_codec] + dequantize_row(h) + [
                            conditional(eq(var('BS dot valid'), 1),
                                        transform.hadamard(h) + transform.apply_signs(h),
                                        [('data_deletealloflist', {}, {'LIST': ['BS output', 'l_output']}, {}),
                                         addlist('nonfinite')]),
                        ],
                        [addlist('error')]),
        ]
        return [conditional(valid, load_row, [addlist('invalid dimensions')])]

    # mode 16: final output_norm (RMS norm + elementwise F32 weight multiply)
    valid = op('operator_and',
               OPERAND1=gt(length('BS input', 'l_input'), 0),
               OPERAND2=eq(length('BS input', 'l_input'), length('BS norm weights', 'l_norm_weights')))
    if with_benchmark:
        n_status = ('bonsailoader_status', {'ID': 'norm_w'}, {}, False)
        n_copied = ('bonsailoader_copied', {'ID': 'norm_w'}, {}, False)
        ensure_weights = [conditional(eq(length('BS norm weights', 'l_norm_weights'), 0), [
            ('bonsailoader_loadDense',
             {'ID': 'norm_w', 'TENSOR': 'output_norm.weight', 'START': 0,
              'COUNT': length('BS input', 'l_input'), 'DEST': 'l_norm_weights'}, {}, {}),
            conditional(op('operator_and', OPERAND1=eq(n_status, 'ok'),
                           OPERAND2=eq(n_copied, length('BS input', 'l_input'))),
                        [], [addlist('error')]),
        ])]
        return ensure_weights + [conditional(eq(length('BS output', 'l_output'), 0),
                                             [conditional(valid, output_norm(h), [addlist('invalid dimensions')])])]
    return [conditional(valid, output_norm(h), [addlist('invalid dimensions')])]
