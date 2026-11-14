"""Visible Scratch LUT4 diagnostic kernel for one resident PQ2 row."""


def script(h, head=False):
    var, op, floor, item, length, setvar, addlist, conditional, repeat, change = (
        h[name] for name in ('var', 'op', 'floor', 'item', 'length', 'setvar',
                             'addlist', 'conditional', 'repeat', 'change'))
    add = lambda a, b: op('operator_add', NUM1=a, NUM2=b)
    sub = lambda a, b: op('operator_subtract', NUM1=a, NUM2=b)
    mul = lambda a, b: op('operator_multiply', NUM1=a, NUM2=b)
    div = lambda a, b: op('operator_divide', NUM1=a, NUM2=b)
    mod = lambda a, b: op('operator_mod', NUM1=a, NUM2=b)
    eq = lambda a, b: op('operator_equals', OPERAND1=a, OPERAND2=b)
    v = lambda name: var('BS ' + name)
    q8 = lambda index: item('BS q8', 'l_q8', index)
    logical_group0 = sub(v('dot group'), 1)
    group0 = add(logical_group0, mul(v('tile row'), v('groups per row'))) if head else logical_group0
    q8_base = mul(floor(div(logical_group0, 2)), 257)
    q8_idx = lambda a: add(add(mul(floor(div(a, 256)), 257), 2), mod(a, 256))
    init_entry = [setvar('BS lut sum', sub(v('lut sum'),
                      q8(q8_idx(add(v('lut a0'), v('lut digit')))))),
                  change('BS lut digit', 1)]
    dp = [conditional(eq(v('lut pattern'), v('lut next')),
                      [setvar('BS lut pow', v('lut next')),
                       setvar('BS lut next', mul(v('lut pow'), v('lut base'))),
                       change('BS lut level', 1)]),
          addlist(add(item('BS lut', 'l_lut',
                         add(mul(v('lut chunk'), v('lut domain')),
                             add(mod(v('lut pattern'), v('lut pow')), 1))),
                     mul(floor(div(v('lut pattern'), v('lut pow'))),
                         q8(q8_idx(add(v('lut a0'), v('lut level')))))),
                  'BS lut', 'l_lut'),
          change('BS lut pattern', 1)]
    setup = [setvar('BS lut a0', sub(mul(v('lut chunk'), v('lut stride')),
                                    mul(floor(div(v('lut chunk'), v('lut groups'))),
                                        sub(mul(v('lut groups'), v('lut stride')), 128)))),
             setvar('BS lut sum', 0), setvar('BS lut digit', 0),
             repeat(v('lut stride'), init_entry),
             addlist(v('lut sum'), 'BS lut', 'l_lut'),
             setvar('BS lut pattern', 1), setvar('BS lut pow', 1),
             setvar('BS lut next', v('lut base')), setvar('BS lut level', 0),
             repeat(sub(v('lut domain'), 1), dp),
             change('BS lut chunk', 1)]
    inner = [setvar('BS subtotal', add(v('subtotal'),
                    item('BS lut', 'l_lut',
                         add(v('lut cur'), add(item('BS packed bytes', 'l_packed', v('byte cur')), 1))))),
             change('BS byte cur', 1), change('BS lut cur', v('lut domain'))]
    word, exponent, mantissa, sign = (v('word'), v('exponent'), v('mantissa'), v('sign'))
    normal = mul(sign, mul(add(1024, mantissa), item('BS powers', 'l_powers', add(exponent, 1))))
    subnormal = mul(sign, mul(mantissa, 2**-24))
    group = [setvar('BS word', item('BS weight scales', 'l_scales', add(group0, 1))),
             setvar('BS exponent', mod(floor(div(word, 1024)), 32)),
             setvar('BS mantissa', mod(word, 1024)),
             conditional(op('operator_gt', OPERAND1=word, OPERAND2=32767),
                         [setvar('BS sign', -1)], [setvar('BS sign', 1)]),
             conditional(eq(exponent, 31), [setvar('BS dot valid', 0)], [
                 conditional(eq(exponent, 0), [setvar('BS weight scale', subnormal)], [setvar('BS weight scale', normal)]),
                 setvar('BS subtotal', 0),
                 setvar('BS byte cur', add(mul(group0, 32), 1)),
                setvar('BS lut cur', mul(logical_group0, mul(v('lut groups'), v('lut domain')))),
                 repeat(v('lut groups'), inner),
                 setvar('BS f32 input', mul(v('weight scale'), q8(add(q8_base, 1)))), h['call_f32'](),
                 setvar('BS f32 input', mul(v('f32 output'), v('subtotal'))), h['call_f32'](),
                 setvar('BS f32 input', add(v('total'), v('f32 output'))), h['call_f32'](),
                 setvar('BS total', v('f32 output'))]), change('BS dot group', 1)]
    if head:
        codec = ('bonsailoader_codec', {}, {}, False)
        valid = op('operator_and',
                   OPERAND1=op('operator_and',
                               OPERAND1=op('operator_gt', OPERAND1=length('BS input', 'l_input'), OPERAND2=0),
                               OPERAND2=eq(mod(length('BS input', 'l_input'), 256), 0)),
                   OPERAND2=op('operator_or', OPERAND1=eq(codec, 'raw'),
                               OPERAND2=op('operator_or', OPERAND1=eq(codec, 'bslut4'),
                                           OPERAND2=eq(codec, 'bslut5'))))
        valid_shape = op('operator_and', OPERAND1=valid,
                       OPERAND2=op('operator_and', OPERAND1=op('operator_gt', OPERAND1=v('tile size'), OPERAND2=0),
                       OPERAND2=op('operator_and',
                           OPERAND1=eq(v('tile size'), floor(v('tile size'))),
                           OPERAND2=op('operator_and',
                               OPERAND1=op('operator_lt', OPERAND1=mul(v('tile size'), div(length('BS input', 'l_input'), 128)), OPERAND2=123362),
                               OPERAND2=op('operator_and',
                                   OPERAND1=op('operator_gt', OPERAND1=v('head rows'), OPERAND2=-1),
                                   OPERAND2=eq(v('head rows'), floor(v('head rows'))))))))
    else:
        valid = op('operator_and',
                   OPERAND1=eq(length('BS input', 'l_input'), mul(length('BS weight scales', 'l_scales'), 128)),
                   OPERAND2=op('operator_and',
                               OPERAND1=eq(length('BS packed bytes', 'l_packed'), mul(length('BS weight scales', 'l_scales'), 32)),
                               OPERAND2=eq(mod(length('BS weight scales', 'l_scales'), 2), 0)))
    timer = ('sensing_timer', {}, {}, False)
    work = [setvar('BS benchmark start', timer),
            ('data_deletealloflist', {}, {'LIST': ['BS q8', 'l_q8']}, {}),
            ('data_deletealloflist', {}, {'LIST': ['BS lut', 'l_lut']}, {})]
    work += h['q8_script']('BS q8', 'l_q8')
    if head:
        work += [conditional(eq(codec, 'raw'), [setvar('BS codec', 0)],
                 [conditional(eq(codec, 'bslut4'), [setvar('BS codec', 1)], [setvar('BS codec', 2)])])]
        valid = valid_shape
    work += [setvar('BS benchmark quant', sub(timer, v('benchmark start'))),
             setvar('BS benchmark start', timer),
             conditional(eq(v('codec'), 0),
                         [setvar('BS lut base', 4), setvar('BS lut domain', 256),
                          setvar('BS lut stride', 4), setvar('BS lut groups', 32)],
                 [conditional(eq(v('codec'), 1),
                         [setvar('BS lut base', 3), setvar('BS lut domain', 81),
                          setvar('BS lut stride', 4), setvar('BS lut groups', 32)],
                         [setvar('BS lut base', 3), setvar('BS lut domain', 243),
                          setvar('BS lut stride', 5), setvar('BS lut groups', 26)])]),
             setvar('BS lut chunk', 0),
             repeat(div(mul(length('BS input', 'l_input'), v('lut groups')), 128), setup),
             setvar('BS benchmark setup', sub(timer, v('benchmark start')))]
    if head:
        status = ('bonsailoader_status', {'ID': 'head'}, {}, False)
        copied = ('bonsailoader_copied', {'ID': 'head'}, {}, False)
        row = [setvar('BS dot group', 1), setvar('BS dot valid', 1), setvar('BS total', 0),
               repeat(v('groups per row'), group),
               conditional(eq(v('dot valid'), 1), [addlist(v('total'))], [setvar('BS head valid', 0)]),
               change('BS tile row', 1)]
        successful = [setvar('BS tile row', 0), setvar('BS benchmark start', timer),
                      repeat(v('tile count'), row),
                      setvar('BS benchmark compute', add(v('benchmark compute'), sub(timer, v('benchmark start')))),
                      change('BS scanned', v('tile count'))]
        transfer = [('bonsailoader_loadGroups',
                     {'ID': 'head', 'TENSOR': v('head tensor'), 'START': mul(v('scanned'), v('groups per row')),
                      'COUNT': mul(v('tile count'), v('groups per row')), 'Q': 'l_packed', 'SCALE': 'l_scales'}, {}, {}),
                    setvar('BS benchmark transfer', add(v('benchmark transfer'), sub(timer, v('benchmark start')))),
                    conditional(op('operator_and', OPERAND1=eq(status, 'ok'),
                                   OPERAND2=eq(copied, mul(v('tile count'), v('groups per row')))),
                                successful, [setvar('BS head valid', 0)])]
        tile = [conditional(eq(v('cancel'), 0), [
                    setvar('BS tile count', sub(v('head rows'), v('scanned'))),
                    conditional(op('operator_gt', OPERAND1=v('tile count'), OPERAND2=v('tile size')),
                                [setvar('BS tile count', v('tile size'))]),
                    setvar('BS benchmark start', timer), *transfer,
                    ('control_wait', {'DURATION': 0}, {}, {})], [setvar('BS head valid', 0)])]
        work += [setvar('BS groups per row', div(length('BS input', 'l_input'), 128)),
                 setvar('BS scanned', 0), setvar('BS head valid', 1), setvar('BS benchmark transfer', 0),
                 setvar('BS benchmark compute', 0),
                 repeat(floor(div(sub(add(v('head rows'), v('tile size')), 1), v('tile size'))),
                        [conditional(eq(v('head valid'), 1), tile)]),
                 conditional(eq(v('head valid'), 0), [addlist('error')])]
    else:
        work += [setvar('BS benchmark start', timer),
                 repeat(v('iterations'), [setvar('BS dot group', 1), setvar('BS dot valid', 1), setvar('BS total', 0),
                                           repeat(length('BS weight scales', 'l_scales'), group)]),
                 setvar('BS benchmark compute', sub(timer, v('benchmark start'))),
                 conditional(eq(v('dot valid'), 1), [addlist(v('total'))], [addlist('nonfinite')])]
    return [conditional(valid, work, [addlist('invalid dimensions')])]
