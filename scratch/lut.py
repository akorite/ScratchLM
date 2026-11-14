"""Visible Scratch LUT4 diagnostic kernel for one resident PQ2 row."""

from scratch import embedding


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
    base = sub(v('dot group'), 1)
    if head:
        base = add(base, mul(v('tile row'), v('groups per row')))
    group = group_body(h, base, sub(v('dot group'), 1)) + [change('BS dot group', 1)]
    if head:
        codec = ('bonsailoader_codec', {}, {}, False)
        valid = op('operator_and',
                   OPERAND1=op('operator_and',
                               OPERAND1=op('operator_gt', OPERAND1=length('BS input', 'l_input'), OPERAND2=0),
                               OPERAND2=eq(mod(length('BS input', 'l_input'), 256), 0)),
                  OPERAND2=embedding.valid_codec_name(op, codec))
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
        work += [embedding.set_codec_from_loader(h, codec)]
        valid = valid_shape
    work += [setvar('BS benchmark quant', sub(timer, v('benchmark start'))),
             setvar('BS benchmark start', timer),
             *codec_tables(h),
             setvar('BS lut chunk', 0),
             *build_lut(h),
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
                    *embedding.settle(h, status),
                    setvar('BS benchmark transfer', add(v('benchmark transfer'), sub(timer, v('benchmark start')))),
                    conditional(op('operator_and', OPERAND1=eq(status, 'ok'),
                                   OPERAND2=eq(copied, mul(v('tile count'), v('groups per row')))),
                                successful, [setvar('BS head valid', 0)])]
        tile = [conditional(eq(v('cancel'), 0), [
                    setvar('BS tile count', sub(v('head rows'), v('scanned'))),
                    conditional(op('operator_gt', OPERAND1=v('tile count'), OPERAND2=v('tile size')),
                                [setvar('BS tile count', v('tile size'))]),
                    setvar('BS benchmark start', timer), *transfer,
                    ('control_wait', {'DURATION': 0}, {}, {})],
                    [setvar('BS head valid', 0), setvar('BS cancel', 0)])]
        # token_embd is 2048 wide, so every row is 16 PQ2 groups regardless of the input list.
        work += [setvar('BS groups per row', 16),
                 setvar('BS scanned', 0), setvar('BS head valid', 1), setvar('BS benchmark transfer', 0),
                 setvar('BS benchmark compute', 0),
                 repeat(floor(div(sub(add(v('head rows'), v('tile size')), 1), v('tile size'))),
                        [conditional(eq(v('head valid'), 1), tile)]),
                 conditional(eq(v('cancel'), 1), [setvar('BS head valid', 0), setvar('BS cancel', 0)]),
                 conditional(eq(v('head valid'), 0), [addlist('error')])]
    else:
        work += [setvar('BS benchmark start', timer),
                 repeat(v('iterations'), [setvar('BS dot group', 1), setvar('BS dot valid', 1), setvar('BS total', 0),
                                           repeat(length('BS weight scales', 'l_scales'), group)]),
                 setvar('BS benchmark compute', sub(timer, v('benchmark start'))),
                 conditional(eq(v('dot valid'), 1), [addlist(v('total'))], [addlist('nonfinite')])]
    result = [conditional(valid, work, ([setvar('BS head valid', 0)] if head else []) + [addlist('invalid dimensions')])]
    # A post-completion flag is indistinguishable from a new pre-tile request.
    return result + ([setvar('BS cancel', 0)] if head else [])


def greedy_select(h, list_name='BS output', list_id='l_output'):
    var, op, item, length, setvar, conditional, repeat, change = (
        h[name] for name in ('var', 'op', 'item', 'length', 'setvar', 'conditional', 'repeat', 'change'))
    gt = lambda a, b: op('operator_gt', OPERAND1=a, OPERAND2=b)
    sub = lambda a, b: op('operator_subtract', NUM1=a, NUM2=b)
    add = lambda a, b: op('operator_add', NUM1=a, NUM2=b)
    eq = lambda a, b: op('operator_equals', OPERAND1=a, OPERAND2=b)
    tokens, logits = ('BS top tokens', 'l_top_tokens'), ('BS top logits', 'l_top_logits')
    n = length(*logits)
    value, index, pos, scan = (var('BS x'), var('BS index'), var('BS top position'), var('BS top scan'))
    insert = lambda target, value: ('data_insertatlist', {'INDEX': pos, 'ITEM': value},
                                    {'LIST': list(target)}, {})
    delete = lambda target, index: ('data_deleteoflist', {'INDEX': index}, {'LIST': list(target)}, {})
    clear = lambda target: ('data_deletealloflist', {}, {'LIST': list(target)}, {})
    # Ascending token scan and strict comparisons keep lower IDs first on ties.
    step = [
        setvar('BS x', item(list_name, list_id, index)),
        conditional(op('operator_or', OPERAND1=op('operator_lt', OPERAND1=n, OPERAND2=8),
                       OPERAND2=gt(value, item(*logits, n))), [
            setvar('BS top position', add(n, 1)),
            setvar('BS top scan', 1),
            repeat(n, [
                conditional(op('operator_and', OPERAND1=eq(pos, add(n, 1)),
                               OPERAND2=gt(value, item(*logits, scan))),
                            [setvar('BS top position', scan)]),
                change('BS top scan', 1),
            ]),
            insert(tokens, sub(index, 1)), insert(logits, value),
            conditional(gt(n, 8), [delete(tokens, 9), delete(logits, 9)]),
        ]),
        change('BS index', 1),
    ]
    return [
        clear(tokens), clear(logits),
        setvar('BS index', 1),
        repeat(length(list_name, list_id), step),
        conditional(gt(n, 0), [
            setvar('BS greedy token', item(*tokens, 1)),
            setvar('BS greedy logit', item(*logits, 1)),
            setvar('BS second logit', -1e38),
            conditional(gt(n, 1), [setvar('BS second logit', item(*logits, 2))]),
        ]),
    ]


def pipeline(h, call_head=None):
    from scratch import embedding, transform
    var, op, item, length, setvar, addlist, conditional, repeat, change = (
        h[name] for name in ('var', 'op', 'item', 'length', 'setvar',
                             'addlist', 'conditional', 'repeat', 'change'))
    eq = lambda a, b: op('operator_equals', OPERAND1=a, OPERAND2=b)
    in_len = length('BS input', 'l_input')
    load_norm = embedding.load_dense(
        h, 'norm_w', 'output_norm.weight', 'BS norm weights', 'l_norm_weights', in_len, [setvar('BS head valid', 0)])
    load_signs = embedding.load_dense(
        h, 'head_signs', 'prism.hadamard.signs.2048', 'BS signs', 'l_signs', in_len, [setvar('BS head valid', 0)])
    idx = var('BS index')
    copy_to_input = [
        setvar('BS index', 1),
        repeat(in_len, [
            transform.replace_item(idx, item('BS output', 'l_output', idx), 'BS input', 'l_input'),
            change('BS index', 1),
        ]),
        ('data_deletealloflist', {}, {'LIST': ['BS output', 'l_output']}, {}),
    ]
    run_pipeline = (
        embedding.output_norm(h)
        + transform.apply_signs(h)
        + transform.hadamard(h)
        + copy_to_input
        + ([call_head()] if call_head else script(h, head=True))
        + [conditional(eq(var('BS head valid'), 1), greedy_select(h))]
    )
    valid_in = op('operator_and',
                  OPERAND1=op('operator_gt', OPERAND1=in_len, OPERAND2=0),
                  OPERAND2=eq(op('operator_mod', NUM1=in_len, NUM2=512), 0))
    return [
        ('data_deletealloflist', {}, {'LIST': ['BS top tokens', 'l_top_tokens']}, {}),
        ('data_deletealloflist', {}, {'LIST': ['BS top logits', 'l_top_logits']}, {}),
        setvar('BS head valid', 1),
        conditional(valid_in,
                    load_norm + load_signs + [
                        conditional(eq(var('BS head valid'), 1), run_pipeline, [addlist('error')]),
                    ],
                    [setvar('BS head valid', 0), addlist('invalid dimensions')]),
        # As with the head, post-completion flags belong to the next run.
        setvar('BS cancel', 0),
    ]


def codec_tables(h):
    """Set the LUT base, domain, stride and groups from the selected package codec."""
    var, op, setvar, conditional = (h[name] for name in ('var', 'op', 'setvar', 'conditional'))
    eq = lambda a, b: op('operator_equals', OPERAND1=a, OPERAND2=b)
    v = lambda name: var('BS ' + name)
    return [conditional(eq(v('codec'), 0),
                        [setvar('BS lut base', 4), setvar('BS lut domain', 256),
                         setvar('BS lut stride', 4), setvar('BS lut groups', 32)],
                [conditional(eq(v('codec'), 1),
                        [setvar('BS lut base', 3), setvar('BS lut domain', 81),
                         setvar('BS lut stride', 4), setvar('BS lut groups', 32)],
                        [setvar('BS lut base', 3), setvar('BS lut domain', 243),
                         setvar('BS lut stride', 5), setvar('BS lut groups', 26)])])]


def build_lut(h):
    """Build the block LUT for the whole quantized activation in `BS input`."""
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
    q8_idx = lambda a: add(add(mul(floor(div(a, 256)), 257), 2), mod(a, 256))
    init_entry = [setvar('BS lut sum', sub(v('lut sum'), q8(q8_idx(add(v('lut a0'), v('lut digit')))))),
                  change('BS lut digit', 1)]
    dp = [conditional(eq(v('lut pattern'), v('lut next')),
                      [setvar('BS lut pow', v('lut next')),
                       setvar('BS lut next', mul(v('lut pow'), v('lut base'))),
                       change('BS lut level', 1)]),
          addlist(add(item('BS lut', 'l_lut', add(mul(v('lut chunk'), v('lut domain')),
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
    return [setvar('BS lut chunk', 0),
            repeat(div(mul(length('BS input', 'l_input'), v('lut groups')), 128), setup)]


def group_body(h, base, logical=None):
    """Accumulate one 128-weight PQ2 group into BS total.

    `base` indexes this tile's packed bytes and scales; `logical` indexes the
    activation's LUT and Q8_K data, which repeat for every row of the tile.
    """
    if logical is None:
        logical = base
    from scratch import embedding
    var, op, floor, item, setvar, addlist, repeat, change = (
        h[name] for name in ('var', 'op', 'floor', 'item', 'setvar', 'addlist', 'repeat', 'change'))
    add = lambda a, b: op('operator_add', NUM1=a, NUM2=b)
    mul = lambda a, b: op('operator_multiply', NUM1=a, NUM2=b)
    div = lambda a, b: op('operator_divide', NUM1=a, NUM2=b)
    v = lambda name: var('BS ' + name)
    q8 = lambda index: item('BS q8', 'l_q8', index)
    q8_base = mul(floor(div(logical, 2)), 257)
    inner = [setvar('BS subtotal', add(v('subtotal'),
                    item('BS lut', 'l_lut',
                         add(v('lut cur'), add(item('BS packed bytes', 'l_packed', v('byte cur')), 1))))),
             change('BS byte cur', 1), change('BS lut cur', v('lut domain'))]
    return embedding.decode_fp16_scale(h, item('BS weight scales', 'l_scales', add(base, 1)), on_valid=[
        setvar('BS subtotal', 0),
        setvar('BS byte cur', add(mul(base, 32), 1)),
        setvar('BS lut cur', mul(logical, mul(v('lut groups'), v('lut domain')))),
        repeat(v('lut groups'), inner),
        setvar('BS f32 input', mul(v('weight scale'), q8(add(q8_base, 1)))), h['call_f32'](),
        setvar('BS f32 input', mul(v('f32 output'), v('subtotal'))), h['call_f32'](),
        setvar('BS f32 input', add(v('total'), v('f32 output'))), h['call_f32'](),
        setvar('BS total', v('f32 output')),
    ])
