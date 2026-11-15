"""Visible Scratch numeric blocks for modes 4–8.

Mode 4 takes [N, left..., right...] and returns one dot product; malformed
shapes return ['invalid dimensions']. Modes 5–8 take a vector and return a
vector: RMS normalization (the model's F32 epsilon), L2 normalization, SiLU, and
softmax respectively. Empty vectors return [], and a zero L2 vector stays
zero. Inputs must be finite numbers; arithmetic uses Scratch's binary64
semantics, including overflow in the dot product.
"""


def make(mode, helpers):
    if mode not in (4, 5, 6, 7, 8, 10):
        return None
    var, op, floor, item, length, setvar, addlist, conditional, repeat, change = (
        helpers[name] for name in ('var', 'op', 'floor', 'item', 'length', 'setvar',
                                   'addlist', 'conditional', 'repeat', 'change'))
    index, x, maximum, total, divisor, count = map(var, (
        'BS index', 'BS x', 'BS maximum', 'BS total', 'BS divisor', 'BS subtotal'))
    size = length('BS input', 'l_input')
    at = lambda i: item('BS input', 'l_input', i)
    math = lambda name, value: ('operator_mathop', {'NUM': value}, {'OPERATOR': [name, None]}, False)
    add = lambda a, b: op('operator_add', NUM1=a, NUM2=b)
    sub = lambda a, b: op('operator_subtract', NUM1=a, NUM2=b)
    mul = lambda a, b: op('operator_multiply', NUM1=a, NUM2=b)
    div = lambda a, b: op('operator_divide', NUM1=a, NUM2=b)
    eq = lambda a, b: op('operator_equals', OPERAND1=a, OPERAND2=b)
    gt = lambda a, b: op('operator_gt', OPERAND1=a, OPERAND2=b)
    both = lambda a, b: op('operator_and', OPERAND1=a, OPERAND2=b)

    if mode == 10:
        degrees = mul(at(index), 57.29577951308232)
        return [setvar('BS index', 1), repeat(size, [addlist(math('sin', degrees)), addlist(math('cos', degrees)),
                                                       change('BS index', 1)])]

    if mode == 4:
        valid = both(gt(size, 0), both(gt(count, -1), both(eq(count, floor(count)),
                     eq(size, add(1, mul(2, count))))))
        body = [setvar('BS total', add(total, mul(at(add(index, 1)), at(add(add(index, count), 1))))),
                change('BS index', 1)]
        return [setvar('BS subtotal', at(1)), conditional(valid,
                [setvar('BS index', 1), setvar('BS total', 0), repeat(count, body), addlist(total)],
                [addlist('invalid dimensions')])]

    if mode == 7:
        exp_x = math('e ^', x)
        sigmoid = conditional(gt(x, 0),
            [addlist(div(x, add(1, math('e ^', sub(0, x)))))],
            [addlist(mul(x, div(exp_x, add(1, exp_x))))])
        return [setvar('BS index', 1), repeat(size, [setvar('BS x', at(index)), sigmoid,
                                                       change('BS index', 1)])]

    if mode == 8:
        scan = [setvar('BS x', at(index)), conditional(gt(x, maximum), [setvar('BS maximum', x)]),
                change('BS index', 1)]
        exp_shifted = math('e ^', sub(at(index), maximum))
        accumulate = [setvar('BS total', add(total, exp_shifted)), change('BS index', 1)]
        emit = [addlist(div(exp_shifted, total)), change('BS index', 1)]
        return [conditional(gt(size, 0), [setvar('BS maximum', at(1)), setvar('BS index', 1),
                repeat(size, scan), setvar('BS total', 0), setvar('BS index', 1),
                repeat(size, accumulate), setvar('BS index', 1), repeat(size, emit)])]

    # Scaling before squaring avoids overflow and underflow for finite inputs.
    scan = [setvar('BS x', math('abs', at(index))),
            conditional(gt(x, maximum), [setvar('BS maximum', x)]), change('BS index', 1)]
    ratio = div(at(index), maximum)
    accumulate = [setvar('BS total', add(total, mul(ratio, ratio))), change('BS index', 1)]
    if mode == 5:
        initial = 0.0009999999987376214
        squared = add(div(total, size), mul(div(initial, maximum), div(initial, maximum)))
    else:
        initial = 9.999999974752427e-7
        squared = total
    normalized = [setvar('BS total', 0), setvar('BS index', 1), repeat(size, accumulate),
                  setvar('BS divisor', math('sqrt', squared))]
    if mode == 6:
        normalized += [conditional(gt(div(initial, maximum), divisor),
                                   [setvar('BS divisor', div(initial, maximum))])]
    normalized += [setvar('BS index', 1),
                   repeat(size, [addlist(div(div(at(index), maximum), divisor)), change('BS index', 1)])]
    return [setvar('BS maximum', initial), setvar('BS index', 1), repeat(size, scan),
            conditional(gt(maximum, 0), normalized, [repeat(size, [addlist(0)])])]
