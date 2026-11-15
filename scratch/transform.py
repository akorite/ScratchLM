INV_SQRT_512_F32 = 0.04419417306780815


def replace_item(index, value, list_name='BS output', list_id='l_output'):
    return ('data_replaceitemoflist', {'INDEX': index, 'ITEM': value}, {'LIST': [list_name, list_id]}, {})


def hadamard(h, list_name='BS output', list_id='l_output', inline=False):
    if not inline and list_name == 'BS output' and list_id == 'l_output' and h.get('call_hadamard'):
        return [h['call_hadamard']()]
    var, op, item, length, setvar, repeat, change = (
        h[name] for name in ('var', 'op', 'item', 'length', 'setvar', 'repeat', 'change'))
    add = lambda a, b: op('operator_add', NUM1=a, NUM2=b)
    sub = lambda a, b: op('operator_subtract', NUM1=a, NUM2=b)
    mul = lambda a, b: op('operator_multiply', NUM1=a, NUM2=b)
    div = lambda a, b: op('operator_divide', NUM1=a, NUM2=b)
    at = lambda idx: item(list_name, list_id, idx)
    block = var('BS hadamard block')
    step = var('BS hadamard step')
    outer = var('BS hadamard outer')
    pair = var('BS hadamard pair')
    u = var('BS hadamard u')
    v = var('BS hadamard v')
    idx = var('BS index')
    butterfly = [
        setvar('BS hadamard u', at(pair)),
        setvar('BS hadamard v', at(add(pair, step))),
        replace_item(pair, add(u, v), list_name, list_id),
        replace_item(add(pair, step), sub(u, v), list_name, list_id),
        change('BS hadamard pair', 1),
    ]
    stage = [
        setvar('BS hadamard pair', block),
        repeat(outer, [repeat(step, butterfly), change('BS hadamard pair', step)]),
        setvar('BS hadamard step', mul(step, 2)),
        setvar('BS hadamard outer', div(outer, 2)),
    ]
    scale = [
        setvar('BS f32 input', mul(at(idx), INV_SQRT_512_F32)),
        h['call_f32'](),
        replace_item(idx, var('BS f32 output'), list_name, list_id),
        change('BS index', 1),
    ]
    per_block = [
        setvar('BS hadamard step', 1),
        setvar('BS hadamard outer', 256),
        repeat(9, stage),
        setvar('BS index', block),
        repeat(512, scale),
        change('BS hadamard block', 512),
    ]
    return [
        setvar('BS hadamard block', 1),
        repeat(div(length(list_name, list_id), 512), per_block),
    ]


def apply_signs(h, list_name='BS output', list_id='l_output', signs_name='BS signs', signs_id='l_signs'):
    var, op, item, length, setvar, repeat, change = (
        h[name] for name in ('var', 'op', 'item', 'length', 'setvar', 'repeat', 'change'))
    mul = lambda a, b: op('operator_multiply', NUM1=a, NUM2=b)
    idx = var('BS index')
    return [
        setvar('BS index', 1),
        repeat(length(list_name, list_id), [
            replace_item(idx, mul(item(list_name, list_id, idx), item(signs_name, signs_id, idx)),
                         list_name, list_id),
            change('BS index', 1),
        ]),
    ]


def make(mode, h, signs_name='BS signs', signs_id='l_signs'):
    if mode not in (12, 13):
        return None
    var, op, item, length, setvar, addlist, conditional, repeat, change = (
        h[name] for name in ('var', 'op', 'item', 'length', 'setvar',
                             'addlist', 'conditional', 'repeat', 'change'))
    eq = lambda a, b: op('operator_equals', OPERAND1=a, OPERAND2=b)
    gt = lambda a, b: op('operator_gt', OPERAND1=a, OPERAND2=b)
    mod = lambda a, b: op('operator_mod', NUM1=a, NUM2=b)
    mul = lambda a, b: op('operator_multiply', NUM1=a, NUM2=b)
    in_len = length('BS input', 'l_input')
    sign_len = length(signs_name, signs_id)
    valid = op('operator_and',
               OPERAND1=op('operator_and', OPERAND1=gt(in_len, 0), OPERAND2=eq(mod(in_len, 512), 0)),
               OPERAND2=op('operator_or', OPERAND1=eq(sign_len, 0), OPERAND2=eq(sign_len, in_len)))
    idx = var('BS index')
    plain_copy = [
        setvar('BS index', 1),
        repeat(in_len, [addlist(item('BS input', 'l_input', idx)), change('BS index', 1)]),
    ]
    if mode == 12:
        copy = conditional(gt(sign_len, 0), [
            setvar('BS index', 1),
            repeat(in_len, [
                addlist(mul(item('BS input', 'l_input', idx), item(signs_name, signs_id, idx))),
                change('BS index', 1),
            ]),
        ], plain_copy)
        work = [copy] + hadamard(h)
    else:
        work = plain_copy + hadamard(h) + [conditional(gt(sign_len, 0),
                                                       apply_signs(h, signs_name=signs_name, signs_id=signs_id))]
    return [conditional(valid, work, [addlist('invalid dimensions')])]
