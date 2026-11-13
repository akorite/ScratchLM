"""Visible Scratch text primitives using 256 hidden costume-lookup sprites.

Modes 20 and 21 perform exact UTF-16 equality and UTF-8 conversion, replacing
unpaired surrogates with U+FFFD. Each sprite searches 256 code units; costume
lookup and broadcasts are Scratch blocks, not extension code.
"""


def costume_data(asset, high):
    units = [{'name': 'u' + chr(unit), 'dataFormat': 'svg', 'assetId': asset,
              'md5ext': asset + '.svg', 'rotationCenterX': 0, 'rotationCenterY': 0}
             for unit in range(high * 256, (high + 1) * 256)]
    return [{'name': 'sentinel', 'dataFormat': 'svg', 'assetId': asset,
             'md5ext': asset + '.svg', 'rotationCenterX': 0, 'rotationCenterY': 0}] + units


def make(mode, helpers):
    if mode not in (20, 21):
        return None
    var, op, floor, item, setvar, addlist, conditional, repeat, change = (
        helpers[name] for name in ('var', 'op', 'floor', 'item', 'setvar',
                                   'addlist', 'conditional', 'repeat', 'change'))
    index, unit, next_unit, code, equal = map(var, (
        'BS text index', 'BS text unit', 'BS text next', 'BS text code', 'BS text equal'))
    at = lambda n: item('BS input', 'l_input', n)
    add = lambda a, b: op('operator_add', NUM1=a, NUM2=b)
    sub = lambda a, b: op('operator_subtract', NUM1=a, NUM2=b)
    mul = lambda a, b: op('operator_multiply', NUM1=a, NUM2=b)
    div = lambda a, b: op('operator_divide', NUM1=a, NUM2=b)
    mod = lambda a, b: op('operator_mod', NUM1=a, NUM2=b)
    eq = lambda a, b: op('operator_equals', OPERAND1=a, OPERAND2=b)
    lt = lambda a, b: op('operator_lt', OPERAND1=a, OPERAND2=b)
    gt = lambda a, b: op('operator_gt', OPERAND1=a, OPERAND2=b)
    both = lambda a, b: op('operator_and', OPERAND1=a, OPERAND2=b)
    textlen = lambda value: op('operator_length', STRING=value)
    def read_unit(text, position, destination):
        letter = op('operator_letter_of', LETTER=position, STRING=text)
        return [setvar('BS text needle', op('operator_join', STRING1='u', STRING2=letter)),
                setvar('BS text result', -1),
                ('event_broadcastandwait', {'BROADCAST_INPUT': 'BS lookup'}, {}, {}),
                setvar(destination, var('BS text result'))]
    if mode == 20:
        scan = (read_unit(at(1), index, 'BS text unit') +
                read_unit(at(2), index, 'BS text next') +
                [conditional(lt(unit, next_unit), [setvar('BS text equal', 0)]),
                 conditional(gt(unit, next_unit), [setvar('BS text equal', 0)]),
                 change('BS text index', 1)])
        return [setvar('BS text equal', 0), conditional(eq(textlen(at(1)), textlen(at(2))),
                [setvar('BS text equal', 1), setvar('BS text index', 1),
                 repeat(textlen(at(1)), scan)]), addlist(equal)]

    continuation = lambda value: add(128, mod(value, 64))
    byte2 = [addlist(add(192, floor(div(code, 64)))), addlist(continuation(code))]
    byte3 = [addlist(add(224, floor(div(code, 4096)))),
             addlist(continuation(floor(div(code, 64)))), addlist(continuation(code))]
    byte4 = [addlist(add(240, floor(div(code, 262144)))),
             addlist(continuation(floor(div(code, 4096)))),
             addlist(continuation(floor(div(code, 64)))), addlist(continuation(code))]
    emit = conditional(lt(code, 128), [addlist(code)], [
        conditional(lt(code, 2048), byte2, [conditional(lt(code, 65536), byte3, byte4)])])
    is_high = both(gt(unit, 55295), lt(unit, 56320))
    is_low = both(gt(unit, 56319), lt(unit, 57344))
    next_is_low = both(gt(next_unit, 56319), lt(next_unit, 57344))
    pair = [setvar('BS text code', add(65536, add(mul(sub(unit, 55296), 1024),
                                                     sub(next_unit, 56320)))),
            change('BS text index', 1)]
    high = [setvar('BS text code', 65533), conditional(lt(index, textlen(at(1))),
            read_unit(at(1), add(index, 1), 'BS text next') + [conditional(next_is_low, pair)])]
    body = (read_unit(at(1), index, 'BS text unit') +
            [setvar('BS text code', unit), conditional(is_high, high,
             [conditional(is_low, [setvar('BS text code', 65533)])]), emit,
             change('BS text index', 1)])
    return [setvar('BS text index', 1),
            repeat(textlen(at(1)), [conditional(lt(index, add(textlen(at(1)), 1)), body)])]
