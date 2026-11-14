"""Scratch-block primitives for the locked decoder layers."""

from scratch.transform import replace_item


def rope(h):
    var, op, floor, item, length, setvar, addlist, conditional, repeat, change = (
        h[name] for name in ('var', 'op', 'floor', 'item', 'length', 'setvar',
                             'addlist', 'conditional', 'repeat', 'change'))
    add = lambda a, b: op('operator_add', NUM1=a, NUM2=b)
    sub = lambda a, b: op('operator_subtract', NUM1=a, NUM2=b)
    mul = lambda a, b: op('operator_multiply', NUM1=a, NUM2=b)
    div = lambda a, b: op('operator_divide', NUM1=a, NUM2=b)
    eq = lambda a, b: op('operator_equals', OPERAND1=a, OPERAND2=b)
    gt = lambda a, b: op('operator_gt', OPERAND1=a, OPERAND2=b)
    math = lambda name, value: ('operator_mathop', {'NUM': value}, {'OPERATOR': [name, None]}, False)
    at = lambda index: item('BS input', 'l_input', index)
    size = sub(length('BS input', 'l_input'), 1)
    pos, head, pair, x, y, theta, cosine, sine = map(var, (
        'BS rope position', 'BS rope head', 'BS rope pair', 'BS rope x', 'BS rope y',
        'BS rope theta', 'BS rope cosine', 'BS rope sine'))
    offset = add(mul(head, 256), pair)
    valid = op('operator_and', OPERAND1=op('operator_and', OPERAND1=gt(size, 0),
               OPERAND2=eq(op('operator_mod', NUM1=size, NUM2=256), 0)),
               OPERAND2=op('operator_and', OPERAND1=gt(pos, -1), OPERAND2=eq(pos, floor(pos))))
    rotate = [
        setvar('BS rope x', at(add(offset, 2))), setvar('BS rope y', at(add(offset, 34))),
        setvar('BS f32 input', math('cos', mul(theta, 57.29577951308232))), h['call_f32'](),
        setvar('BS rope cosine', var('BS f32 output')),
        setvar('BS f32 input', math('sin', mul(theta, 57.29577951308232))), h['call_f32'](),
        setvar('BS rope sine', var('BS f32 output')),
        setvar('BS f32 input', sub(mul(x, cosine), mul(y, sine))), h['call_f32'](),
        replace_item(add(offset, 1), var('BS f32 output')),
        setvar('BS f32 input', add(mul(x, sine), mul(y, cosine))), h['call_f32'](),
        replace_item(add(offset, 33), var('BS f32 output')),
        setvar('BS f32 input', mul(theta, 0.6042963862419128)), h['call_f32'](),
        setvar('BS rope theta', var('BS f32 output')), change('BS rope pair', 1),
    ]
    per_head = [repeat(64, [addlist(0)]), setvar('BS rope pair', 0), setvar('BS rope theta', pos), repeat(32, rotate),
                setvar('BS rope pair', 64), repeat(192, [addlist(at(add(offset, 2))), change('BS rope pair', 1)]),
                change('BS rope head', 1)]
    return [setvar('BS rope position', at(1)), conditional(valid,
            [setvar('BS rope head', 0), repeat(div(size, 256), per_head)], [addlist('invalid dimensions')])]


def causal_softmax(h):
    var, op, floor, item, length, setvar, addlist, conditional, repeat, change = (
        h[name] for name in ('var', 'op', 'floor', 'item', 'length', 'setvar',
                             'addlist', 'conditional', 'repeat', 'change'))
    add = lambda a, b: op('operator_add', NUM1=a, NUM2=b)
    sub = lambda a, b: op('operator_subtract', NUM1=a, NUM2=b)
    div = lambda a, b: op('operator_divide', NUM1=a, NUM2=b)
    eq = lambda a, b: op('operator_equals', OPERAND1=a, OPERAND2=b)
    gt = lambda a, b: op('operator_gt', OPERAND1=a, OPERAND2=b)
    at = lambda index: item('BS input', 'l_input', index)
    count, index, maximum, total = map(var, ('BS subtotal', 'BS index', 'BS maximum', 'BS total'))
    value = at(add(index, 1))
    exp = ('operator_mathop', {'NUM': sub(value, maximum)}, {'OPERATOR': ['e ^', None]}, False)
    valid = op('operator_and', OPERAND1=op('operator_and', OPERAND1=gt(count, 0),
               OPERAND2=eq(count, floor(count))),
               OPERAND2=op('operator_lt', OPERAND1=count, OPERAND2=length('BS input', 'l_input')))
    scan = [conditional(gt(value, maximum), [setvar('BS maximum', value)]), change('BS index', 1)]
    total_loop = [setvar('BS total', add(total, exp)), change('BS index', 1)]
    emit = [setvar('BS f32 input', div(exp, total)), h['call_f32'](),
            addlist(var('BS f32 output')), change('BS index', 1)]
    return [setvar('BS subtotal', at(1)), conditional(valid, [
        setvar('BS maximum', at(2)), setvar('BS index', 1), repeat(count, scan),
        setvar('BS total', 0), setvar('BS index', 1), repeat(count, total_loop),
        setvar('BS index', 1), repeat(count, emit),
    ], [addlist('invalid dimensions')])]


def combine(h, gated):
    var, op, floor, item, length, setvar, addlist, conditional, repeat, change = (
        h[name] for name in ('var', 'op', 'floor', 'item', 'length', 'setvar',
                             'addlist', 'conditional', 'repeat', 'change'))
    add = lambda a, b: op('operator_add', NUM1=a, NUM2=b)
    sub = lambda a, b: op('operator_subtract', NUM1=a, NUM2=b)
    mul = lambda a, b: op('operator_multiply', NUM1=a, NUM2=b)
    div = lambda a, b: op('operator_divide', NUM1=a, NUM2=b)
    eq = lambda a, b: op('operator_equals', OPERAND1=a, OPERAND2=b)
    gt = lambda a, b: op('operator_gt', OPERAND1=a, OPERAND2=b)
    at = lambda index: item('BS input', 'l_input', index)
    count, index, x = map(var, ('BS subtotal', 'BS index', 'BS x'))
    valid = op('operator_and', OPERAND1=op('operator_and', OPERAND1=gt(count, 0),
               OPERAND2=eq(count, floor(count))),
               OPERAND2=eq(length('BS input', 'l_input'), add(1, mul(2, count))))
    rhs = at(add(add(index, count), 1))
    if gated:
        exp_positive = ('operator_mathop', {'NUM': sub(0, x)}, {'OPERATOR': ['e ^', None]}, False)
        exp_negative = ('operator_mathop', {'NUM': x}, {'OPERATOR': ['e ^', None]}, False)
        value = [conditional(gt(x, 0),
                 [setvar('BS f32 input', mul(div(x, add(1, exp_positive)), rhs))],
                 [setvar('BS f32 input', mul(mul(x, div(exp_negative, add(1, exp_negative))), rhs))])]
    else:
        value = [setvar('BS f32 input', add(x, rhs))]
    body = [setvar('BS x', at(add(index, 1)))] + value + [
        h['call_f32'](), addlist(var('BS f32 output')), change('BS index', 1)]
    return [setvar('BS subtotal', at(1)), conditional(valid,
            [setvar('BS index', 1), repeat(count, body)], [addlist('invalid dimensions')])]


def head_norm(h, gated):
    var, op, floor, item, length, setvar, addlist, conditional, repeat, change = (
        h[name] for name in ('var', 'op', 'floor', 'item', 'length', 'setvar',
                             'addlist', 'conditional', 'repeat', 'change'))
    add = lambda a, b: op('operator_add', NUM1=a, NUM2=b)
    sub = lambda a, b: op('operator_subtract', NUM1=a, NUM2=b)
    mul = lambda a, b: op('operator_multiply', NUM1=a, NUM2=b)
    div = lambda a, b: op('operator_divide', NUM1=a, NUM2=b)
    eq = lambda a, b: op('operator_equals', OPERAND1=a, OPERAND2=b)
    gt = lambda a, b: op('operator_gt', OPERAND1=a, OPERAND2=b)
    math = lambda name, value: ('operator_mathop', {'NUM': value}, {'OPERATOR': [name, None]}, False)
    at = lambda index: item('BS input', 'l_input', index)
    width, heads, head, index, total, divisor, gate = map(var, (
        'BS rope width', 'BS rope heads', 'BS rope head', 'BS rope pair', 'BS total', 'BS divisor', 'BS rope y'))
    count = mul(width, heads)
    offset = add(mul(head, width), index)
    value = at(add(offset, 3))
    weight = at(add(add(mul(count, 2 if gated else 1), index), 3))
    valid_int = lambda x: op('operator_and', OPERAND1=gt(x, 0), OPERAND2=eq(x, floor(x)))
    valid = op('operator_and', OPERAND1=op('operator_and', OPERAND1=valid_int(width), OPERAND2=valid_int(heads)),
               OPERAND2=eq(length('BS input', 'l_input'), add(add(2, mul(count, 2 if gated else 1)), width)))
    scan = [setvar('BS total', add(total, mul(value, value))), change('BS rope pair', 1)]
    emit = [setvar('BS f32 input', div(value, divisor)), h['call_f32'](),
            setvar('BS f32 input', mul(var('BS f32 output'), weight)), h['call_f32']()]
    if gated:
        exp_pos, exp_neg = math('e ^', sub(0, gate)), math('e ^', gate)
        emit += [setvar('BS rope y', at(add(add(offset, count), 3))),
                 conditional(gt(gate, 0),
                     [setvar('BS f32 input', mul(var('BS f32 output'), div(gate, add(1, exp_pos))))],
                     [setvar('BS f32 input', mul(var('BS f32 output'), mul(gate, div(exp_neg, add(1, exp_neg)))))]),
                 h['call_f32']()]
    emit += [addlist(var('BS f32 output')), change('BS rope pair', 1)]
    per_head = [setvar('BS total', 0), setvar('BS rope pair', 0), repeat(width, scan),
                setvar('BS divisor', math('sqrt', add(div(total, width), 9.999999974752427e-7))),
                setvar('BS rope pair', 0), repeat(width, emit), change('BS rope head', 1)]
    return [setvar('BS rope width', at(1)), setvar('BS rope heads', at(2)),
            conditional(valid, [setvar('BS rope head', 0), repeat(heads, per_head)], [addlist('invalid dimensions')])]
