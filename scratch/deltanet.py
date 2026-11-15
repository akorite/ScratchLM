"""Scratch blocks for one Gated DeltaNet decoder layer.

Stage 1 covers the input projections and the short convolution; later stages add
the delta rule, the gated norm and the output projection. All arithmetic stays
in Scratch; weight tiles come from the selected package through the loader.
"""

from scratch import attention, embedding, transform

weight = attention.weight

CONV_WIDTH = 6144
TAPS = 4


def conv(h):
    """Depthwise kernel-4 convolution over the circular buffer, then SiLU."""
    var, op, item, setvar, addlist, conditional, repeat, change = (
        h[name] for name in ('var', 'op', 'item', 'setvar', 'addlist', 'conditional', 'repeat', 'change'))
    add = lambda a, b: op('operator_add', NUM1=a, NUM2=b)
    sub = lambda a, b: op('operator_subtract', NUM1=a, NUM2=b)
    mul = lambda a, b: op('operator_multiply', NUM1=a, NUM2=b)
    div = lambda a, b: op('operator_divide', NUM1=a, NUM2=b)
    eq = lambda a, b: op('operator_equals', OPERAND1=a, OPERAND2=b)
    math = lambda name, value: ('operator_mathop', {'NUM': value}, {'OPERATOR': [name, None]}, False)
    v = lambda name: var('BS ' + name)
    channel, tap, total = v('dn channel'), v('dn tap'), v('dn sum')
    # The dense conv1d tensor is channel-major: tap t of channel c lives at c*4+t.
    weight = lambda t: item('BS conv w', 'l_conv_w', add(mul(channel, TAPS), add(t, 1)))
    # Taps 0..2 are history for this layer; tap 3 is this token's fused QKV projection.
    conv_base = add(mul(v('conv base'), 1), mul(tap, CONV_WIDTH))
    sample = [conditional(eq(tap, TAPS - 1),
                          [addlist(item('BS qkv', 'l_qkv', add(channel, 1)), 'BS taps', 'l_taps')],
                          [addlist(item('BS conv state', 'l_conv_state',
                                        add(conv_base, add(channel, 1))), 'BS taps', 'l_taps')]),
              change('BS dn tap', 1)]
    accumulate = [h['setvar']('BS f32 input', add(total, mul(item('BS taps', 'l_taps', add(tap, 1)),
                                                              weight(tap)))), h['call_f32'](),
                  setvar('BS dn sum', var('BS f32 output')), change('BS dn tap', 1)]
    body = [('data_deletealloflist', {}, {'LIST': ['BS taps', 'l_taps']}, {}),
            setvar('BS dn tap', 0), repeat(TAPS, sample),
            setvar('BS dn sum', 0), setvar('BS dn tap', 0), repeat(TAPS, accumulate),
            addlist(total, 'BS conv raw', 'l_conv_raw'),
            h['setvar']('BS f32 input', div(total, add(1, math('e ^', mul(-1, total))))), h['call_f32'](),
            addlist(var('BS f32 output'), 'BS conv silu', 'l_conv_silu'),
            change('BS dn channel', 1)]
    return [('data_deletealloflist', {}, {'LIST': ['BS conv raw', 'l_conv_raw']}, {}),
            ('data_deletealloflist', {}, {'LIST': ['BS conv silu', 'l_conv_silu']}, {}),
            setvar('BS dn channel', 0), repeat(CONV_WIDTH, body)]


def shift_state(h):
    """Roll the circular buffer: the new taps are old taps 1..2 plus this token."""
    var, op, item, setvar, addlist, conditional, repeat, change = (
        h[name] for name in ('var', 'op', 'item', 'setvar', 'addlist', 'conditional', 'repeat', 'change'))
    add = lambda a, b: op('operator_add', NUM1=a, NUM2=b)
    mul = lambda a, b: op('operator_multiply', NUM1=a, NUM2=b)
    eq = lambda a, b: op('operator_equals', OPERAND1=a, OPERAND2=b)
    v = lambda name: var('BS ' + name)
    base = mul(v('conv base'), 1)
    # Next state stays tap-major, matching the buffer layout: tap t of channel c at t*6144+c.
    per_channel = [conditional(eq(v('dn tap'), TAPS - 2),
                               [addlist(item('BS qkv', 'l_qkv', add(v('dn channel'), 1)),
                                        'BS next state', 'l_next_state')],
                               [addlist(item('BS conv state', 'l_conv_state',
                                             add(add(base, mul(add(v('dn tap'), 1), CONV_WIDTH)),
                                                 add(v('dn channel'), 1))),
                                        'BS next state', 'l_next_state')]),
                   change('BS dn channel', 1)]
    build = [setvar('BS dn tap', 0),
             repeat(TAPS - 1, [setvar('BS dn channel', 0), repeat(CONV_WIDTH, per_channel),
                               change('BS dn tap', 1)])]
    write = [transform.replace_item(add(add(base, v('dn channel')), 1),
                                    item('BS next state', 'l_next_state', add(v('dn channel'), 1)),
                                    'BS conv state', 'l_conv_state'),
             change('BS dn channel', 1)]
    return [('data_deletealloflist', {}, {'LIST': ['BS next state', 'l_next_state']}, {}),
            *build,
            *attention.ensure(h, 'BS conv state', 'l_conv_state',
                              add(base, (TAPS - 1) * CONV_WIDTH)),
            setvar('BS dn channel', 0), repeat((TAPS - 1) * CONV_WIDTH, write)]


def dense_rows(h, weight_name, weight_id, rows, dest, dest_id):
    """Dot `BS input` with each dense F32 row of a weight list (no rotation)."""
    var, op, item, length, setvar, addlist, repeat, change = (
        h[name] for name in ('var', 'op', 'item', 'length', 'setvar', 'addlist', 'repeat', 'change'))
    add = lambda a, b: op('operator_add', NUM1=a, NUM2=b)
    sub = lambda a, b: op('operator_subtract', NUM1=a, NUM2=b)
    mul = lambda a, b: op('operator_multiply', NUM1=a, NUM2=b)
    v = lambda name: var('BS ' + name)
    width = length('BS input', 'l_input')
    product = [h['setvar']('BS f32 input', add(v('dn sum'), mul(
                   item('BS input', 'l_input', add(v('dn index'), 1)),
                   item(weight_name, weight_id, add(mul(v('dn row'), width), add(v('dn index'), 1)))))), h['call_f32'](),
               setvar('BS dn sum', var('BS f32 output')), change('BS dn index', 1)]
    row = [setvar('BS dn sum', 0), setvar('BS dn index', 0), repeat(width, product),
           addlist(v('dn sum'), dest, dest_id), change('BS dn row', 1)]
    return [('data_deletealloflist', {}, {'LIST': [dest, dest_id]}, {}),
            setvar('BS dn row', 0), repeat(rows, row)]


def split(h):
    """Split the convolved QKV vector into q, k and v halves."""
    var, item, setvar, addlist, repeat, change, op = (
        h[name] for name in ('var', 'item', 'setvar', 'addlist', 'repeat', 'change', 'op'))
    add = lambda a, b: op('operator_add', NUM1=a, NUM2=b)
    v = lambda name: var('BS ' + name)
    move = lambda source, source_id, offset, dest, dest_id: [
        ('data_deletealloflist', {}, {'LIST': [dest, dest_id]}, {}),
        setvar('BS dn channel', 0),
        repeat(2048, [addlist(item(source, source_id, add(v('dn channel'), add(offset, 1))),
                              dest, dest_id), change('BS dn channel', 1)])]
    # q and k enter the recurrence L2-normalized; v is the raw convolution output.
    return (move('BS qk l2', 'l_qk_l2', 0, 'BS q', 'l_q')
            + move('BS qk l2', 'l_qk_l2', 2048, 'BS k', 'l_k')
            + move('BS conv silu', 'l_conv_silu', 4096, 'BS v', 'l_v'))


def l2(h, source, source_id, dest, dest_id, heads=16, width=128, eps='9.999999974752427e-7'):
    """Per-head L2 normalization: y = x / max(||x||, eps)."""
    var, op, item, setvar, addlist, repeat, change = (
        h[name] for name in ('var', 'op', 'item', 'setvar', 'addlist', 'repeat', 'change'))
    add = lambda a, b: op('operator_add', NUM1=a, NUM2=b)
    mul = lambda a, b: op('operator_multiply', NUM1=a, NUM2=b)
    div = lambda a, b: op('operator_divide', NUM1=a, NUM2=b)
    gt = lambda a, b: op('operator_gt', OPERAND1=a, OPERAND2=b)
    math = lambda name, value: ('operator_mathop', {'NUM': value}, {'OPERATOR': [name, None]}, False)
    v = lambda name: var('BS ' + name)
    at = lambda position: add(mul(v('dn head'), width), add(v('dn index'), position))
    square = [setvar('BS dn sum', add(v('dn sum'), mul(item(source, source_id, at(1)),
                                                      item(source, source_id, at(1))))),
              change('BS dn index', 1)]
    emit = [h['setvar']('BS f32 input', mul(item(source, source_id, at(1)), v('dn scale'))), h['call_f32'](),
            addlist(var('BS f32 output'), dest, dest_id), change('BS dn index', 1)]
    head = [setvar('BS dn sum', 0), setvar('BS dn index', 0), repeat(width, square),
            setvar('BS dn scale', 1), setvar('BS dn scale', math('sqrt', v('dn sum'))),
            h['setvar']('BS f32 input', div(1, v('dn scale'))), h['call_f32'](),
            setvar('BS dn scale', var('BS f32 output')),
            setvar('BS dn index', 0), repeat(width, emit), change('BS dn head', 1)]
    return [('data_deletealloflist', {}, {'LIST': [dest, dest_id]}, {}),
            setvar('BS dn head', 0), repeat(heads, head)]


def delta(h):
    """The Gated DeltaNet recurrence: decay, delta rule update and readout per head.

    Every inner loop walks its slice with one incrementing cursor instead of
    recomputing `base + head*16384 + row*128 + col + 1` per element, which was
    four extra blocks per state read.
    """
    var, op, floor, item, length, setvar, addlist, conditional, repeat, change = (
        h[name] for name in ('var', 'op', 'floor', 'item', 'length', 'setvar',
                             'addlist', 'conditional', 'repeat', 'change'))
    add = lambda a, b: op('operator_add', NUM1=a, NUM2=b)
    sub = lambda a, b: op('operator_subtract', NUM1=a, NUM2=b)
    mul = lambda a, b: op('operator_multiply', NUM1=a, NUM2=b)
    div = lambda a, b: op('operator_divide', NUM1=a, NUM2=b)
    eq = lambda a, b: op('operator_equals', OPERAND1=a, OPERAND2=b)
    gt = lambda a, b: op('operator_gt', OPERAND1=a, OPERAND2=b)
    v = lambda name: var('BS ' + name)
    head, row, col = v('dn head'), v('dn j'), v('dn i')
    cursor, vector = v('dn cursor'), v('dn vec')
    beta = lambda hd: item('BS beta sigmoid', 'l_beta_sigmoid', add(hd, 1))
    state = lambda: item('BS state', 'l_state', cursor)
    read = lambda name, list_id, at: item(name, list_id, add(vector, at))
    decay = item('BS decay', 'l_decay', add(head, 1))
    # one row of the state is contiguous: base + head*16384 + row*128 + 1
    row_cursor = lambda: [setvar('BS dn cursor',
                                 add(add(add(mul(v('state base'), 1), mul(head, 16384)), mul(row, 128)), 1))]
    vector_cursor = lambda: [setvar('BS dn vec', add(mul(head, 128), 1))]
    f32 = h['call_f32']
    # decay: S[j][:] *= exp(g)
    scale = [*row_cursor(), setvar('BS dn i', 0),
             repeat(128, [h['setvar']('BS f32 input', mul(state(), decay)), f32(),
                          ('data_replaceitemoflist', {'INDEX': cursor, 'ITEM': var('BS f32 output')},
                           {'LIST': ['BS state', 'l_state']}, {}),
                          change('BS dn cursor', 1)])]
    # accumulator loops keep the native reduction order: exact f32 sum, one rounding
    dot = lambda name, list_id: [setvar('BS dn sum', add(v('dn sum'), mul(state(), read(name, list_id, col)))),
                                 change('BS dn cursor', 1), change('BS dn i', 1)]
    round_sum = [h['setvar']('BS f32 input', v('dn sum')), f32(),
                 setvar('BS dn sum', var('BS f32 output'))]
    update = [*vector_cursor(),
              setvar('BS dn sum', 0), setvar('BS dn i', 0), *row_cursor(),
              repeat(128, dot('BS k', 'l_k')), *round_sum,
              h['setvar']('BS f32 input', mul(sub(read('BS v', 'l_v', row), v('dn sum')), beta(head))), f32(),
              setvar('BS dn delta', var('BS f32 output')),
              *row_cursor(), setvar('BS dn i', 0),
              repeat(128, [h['setvar']('BS f32 input', add(state(), mul(v('dn delta'), read('BS k', 'l_k', col)))),
                           f32(),
                           ('data_replaceitemoflist', {'INDEX': cursor, 'ITEM': var('BS f32 output')},
                            {'LIST': ['BS state', 'l_state']}, {}),
                           change('BS dn cursor', 1), change('BS dn i', 1)]),
              *row_cursor(), setvar('BS dn sum', 0), setvar('BS dn i', 0),
              repeat(128, dot('BS q', 'l_q')), *round_sum,
              h['setvar']('BS f32 input', mul(v('dn sum'), 0.08838834764831845)), f32(),
              addlist(var('BS f32 output'), 'BS gdn out', 'l_gdn_out'),
              change('BS dn j', 1)]
    scale_all = [setvar('BS dn j', 0), repeat(128, [*scale, change('BS dn j', 1)])]
    per_head = [*scale_all, setvar('BS dn j', 0), repeat(128, update), change('BS dn head', 1)]
    return [('data_deletealloflist', {}, {'LIST': ['BS gdn out', 'l_gdn_out']}, {}),
            *attention.ensure(h, 'BS state', 'l_state', add(mul(v('state base'), 1), 262144)),
            setvar('BS dn head', 0), repeat(16, per_head)]


def gates(h):
    """beta = sigmoid(beta_raw); decay = exp(ssm_a * softplus(alpha_raw + dt_bias))."""
    var, op, item, setvar, addlist, conditional, repeat, change = (
        h[name] for name in ('var', 'op', 'item', 'setvar', 'addlist', 'conditional', 'repeat', 'change'))
    add = lambda a, b: op('operator_add', NUM1=a, NUM2=b)
    sub = lambda a, b: op('operator_subtract', NUM1=a, NUM2=b)
    mul = lambda a, b: op('operator_multiply', NUM1=a, NUM2=b)
    div = lambda a, b: op('operator_divide', NUM1=a, NUM2=b)
    gt = lambda a, b: op('operator_gt', OPERAND1=a, OPERAND2=b)
    math = lambda name, value: ('operator_mathop', {'NUM': value}, {'OPERATOR': [name, None]}, False)
    v = lambda name: var('BS ' + name)
    idx = v('dn head')
    beta_raw = item('BS beta', 'l_beta', add(idx, 1))
    sigmoid = [h['setvar']('BS f32 input', div(1, add(1, math('e ^', mul(-1, beta_raw))))), h['call_f32'](),
               addlist(var('BS f32 output'), 'BS beta sigmoid', 'l_beta_sigmoid')]
    x = add(item('BS alpha', 'l_alpha', add(idx, 1)), item('BS dt', 'l_dt', add(idx, 1)))
    # softplus(x) = x for x > 20, else ln(1 + exp(x)); the decay is exp(ssm_a * softplus).
    softplus = [conditional(gt(x, 20), [h['setvar']('BS dn sp', x)],
                            [h['setvar']('BS f32 input', math('e ^', x)), h['call_f32'](),
                             h['setvar']('BS f32 input', math('ln', add(1, var('BS f32 output')))), h['call_f32'](),
                             h['setvar']('BS dn sp', var('BS f32 output'))]),
                h['setvar']('BS f32 input', mul(item('BS ssm a', 'l_ssm_a', add(idx, 1)), v('dn sp'))),
                h['call_f32'](),
                h['setvar']('BS f32 input', math('e ^', var('BS f32 output'))), h['call_f32'](),
                addlist(var('BS f32 output'), 'BS decay', 'l_decay')]
    return [('data_deletealloflist', {}, {'LIST': ['BS beta sigmoid', 'l_beta_sigmoid']}, {}),
            ('data_deletealloflist', {}, {'LIST': ['BS decay', 'l_decay']}, {}),
            setvar('BS dn head', 0), repeat(16, sigmoid + softplus + [change('BS dn head', 1)])]


def gated_norm(h):
    """Per-head RMS norm with the ssm_norm weight, multiplied by silu(z)."""
    var, op, item, setvar, addlist, repeat, change = (
        h[name] for name in ('var', 'op', 'item', 'setvar', 'addlist', 'repeat', 'change'))
    add = lambda a, b: op('operator_add', NUM1=a, NUM2=b)
    sub = lambda a, b: op('operator_subtract', NUM1=a, NUM2=b)
    mul = lambda a, b: op('operator_multiply', NUM1=a, NUM2=b)
    div = lambda a, b: op('operator_divide', NUM1=a, NUM2=b)
    math = lambda name, value: ('operator_mathop', {'NUM': value}, {'OPERATOR': [name, None]}, False)
    v = lambda name: var('BS ' + name)
    head, i = v('dn head'), v('dn i')
    out = lambda d: item('BS gdn out', 'l_gdn_out', add(add(mul(head, 128), d), 1))
    z = lambda d: item('BS z', 'l_z', add(add(mul(head, 128), d), 1))
    square = [setvar('BS dn sum', add(v('dn sum'), mul(out(i), out(i)))), change('BS dn i', 1)]
    emit = [h['setvar']('BS f32 input', mul(mul(out(i), v('dn scale')),
                                             item('BS ssm norm', 'l_ssm_norm', add(i, 1)))), h['call_f32'](),
            h['setvar']('BS f32 input', mul(var('BS f32 output'),
                                            div(z(i), add(1, math('e ^', mul(-1, z(i))))))), h['call_f32'](),
            addlist(var('BS f32 output'), 'BS final out', 'l_final_out'), change('BS dn i', 1)]
    per_head = [setvar('BS dn sum', 0), setvar('BS dn i', 0), repeat(128, square),
                h['setvar']('BS f32 input', div(v('dn sum'), 128)), h['call_f32'](),
                h['setvar']('BS f32 input', add(var('BS f32 output'), 9.999999974752427e-7)), h['call_f32'](),
                h['setvar']('BS f32 input', math('sqrt', var('BS f32 output'))), h['call_f32'](),
                h['setvar']('BS f32 input', div(1, var('BS f32 output'))), h['call_f32'](),
                setvar('BS dn scale', var('BS f32 output')),
                setvar('BS dn i', 0), repeat(128, emit), change('BS dn head', 1)]
    return [('data_deletealloflist', {}, {'LIST': ['BS final out', 'l_final_out']}, {}),
            setvar('BS dn head', 0), repeat(16, per_head)]


def _unused_weight(h, suffix):
    """'blk.<layer><suffix>' as a Scratch join expression."""
    op = h['op']
    join = lambda a, b: op('operator_join', STRING1=a, STRING2=b)
    return join('blk.', join(h['var']('BS layer'), suffix))


def stage(h):
    """One Gated DeltaNet layer for the token in `BS input`."""
    var, op, item, setvar, addlist, length, conditional, repeat, change = (
        h[name] for name in ('var', 'op', 'item', 'setvar', 'addlist', 'length', 'conditional',
                             'repeat', 'change'))
    gt = lambda a, b: op('operator_gt', OPERAND1=a, OPERAND2=b)
    add = lambda a, b: op('operator_add', NUM1=a, NUM2=b)
    eq = lambda a, b: op('operator_equals', OPERAND1=a, OPERAND2=b)
    codec = ('bonsailoader_codec', {}, {}, False)
    body = [
        embedding.set_codec_from_loader(h, codec),
        *embedding.load_dense(h, 'dn-norm', weight(h, '.attn_norm.weight'), 'BS norm w', 'l_norm_w', 2048,
                              [setvar('BS proj valid', 0)]),
        *embedding.load_dense(h, 'dn-conv', weight(h, '.ssm_conv1d.weight'), 'BS conv w', 'l_conv_w',
                              TAPS * CONV_WIDTH, [setvar('BS proj valid', 0)]),
        *embedding.load_dense(h, 'dn-signs', 'prism.hadamard.signs.2048', 'BS signs', 'l_signs', 2048,
                              [setvar('BS proj valid', 0)]),
        *attention.weighted_norm(h, 'BS norm w', 'l_norm_w', 'BS normed', 'l_normed'),
        *attention.copy(h, 'BS input', 'l_input', 'BS layer input', 'l_layer_input'),
        *attention.copy(h, 'BS normed', 'l_normed', 'BS input', 'l_input'),
        *attention.rotate_in(h, 2048),
        *attention.matvec(h, weight(h, '.attn_qkv.weight'), CONV_WIDTH, 'BS qkv', 'l_qkv', request='dn-qkv', proc='BS mv dn-qkv'),
        *conv(h),
        *shift_state(h),
        # z is its own folded projection; beta and alpha are small dense rows.
        *attention.copy(h, 'BS normed', 'l_normed', 'BS input', 'l_input'),
        *attention.rotate_in(h, 2048),
        *attention.matvec(h, weight(h, '.attn_gate.weight'), 2048, 'BS z', 'l_z', request='dn-z', proc='BS mv dn-z'),
        *embedding.load_dense(h, 'dn-beta', weight(h, '.ssm_beta.weight'), 'BS beta w', 'l_beta_w',
                              16 * 2048, [setvar('BS proj valid', 0)]),
        *attention.copy(h, 'BS normed', 'l_normed', 'BS input', 'l_input'),
        *dense_rows(h, 'BS beta w', 'l_beta_w', 16, 'BS beta', 'l_beta'),
        *embedding.load_dense(h, 'dn-alpha', weight(h, '.ssm_alpha.weight'), 'BS alpha w', 'l_alpha_w',
                              16 * 2048, [setvar('BS proj valid', 0)]),
        *attention.copy(h, 'BS normed', 'l_normed', 'BS input', 'l_input'),
        *dense_rows(h, 'BS alpha w', 'l_alpha_w', 16, 'BS alpha', 'l_alpha'),
        # q and k are adjacent head groups, normalized together as 32 heads of 128.
        ('data_deletealloflist', {}, {'LIST': ['BS qk', 'l_qk']}, {}),
        setvar('BS dn channel', 0),
        repeat(4096, [addlist(item('BS conv silu', 'l_conv_silu', add(var('BS dn channel'), 1)), 'BS qk', 'l_qk'),
                      change('BS dn channel', 1)]),
        *l2(h, 'BS qk', 'l_qk', 'BS qk l2', 'l_qk_l2', heads=32),
        *split(h),
        *embedding.load_dense(h, 'dn-a', weight(h, '.ssm_a'), 'BS ssm a', 'l_ssm_a', 16,
                              [setvar('BS proj valid', 0)]),
        *embedding.load_dense(h, 'dn-dt', weight(h, '.ssm_dt.bias'), 'BS dt', 'l_dt', 16,
                              [setvar('BS proj valid', 0)]),
        *gates(h),
        *delta(h),
        *embedding.load_dense(h, 'dn-normw', weight(h, '.ssm_norm.weight'), 'BS ssm norm', 'l_ssm_norm', 128,
                              [setvar('BS proj valid', 0)]),
        *embedding.load_dense(h, 'dn-postw', weight(h, '.post_attention_norm.weight'), 'BS post w', 'l_post_w', 2048,
                              [setvar('BS proj valid', 0)]),
        *embedding.load_dense(h, 'dn-signsw', 'prism.hadamard.signs.6144', 'BS signs wide', 'l_signs_wide', 6144,
                              [setvar('BS proj valid', 0)]),
        *gated_norm(h),
        *attention.copy(h, 'BS final out', 'l_final_out', 'BS input', 'l_input'),
        *attention.rotate_in(h, 2048),
        *attention.matvec(h, weight(h, '.ssm_out.weight'), 2048, 'BS attn out', 'l_attn_out', request='dn-out', proc='BS mv dn-out'),
        *h['combine'](h, gated=False, rhs=('BS attn out', 'l_attn_out'), input_name='BS layer input',
                      input_id='l_layer_input', output=('BS residual', 'l_residual')),
        *attention.copy(h, 'BS residual', 'l_residual', 'BS input', 'l_input'),
        *attention.weighted_norm(h, 'BS post w', 'l_post_w', 'BS normed2', 'l_normed2'),
        *attention.copy(h, 'BS normed2', 'l_normed2', 'BS input', 'l_input'),
        *attention.rotate_in(h, 2048),
        *attention.matvec(h, weight(h, '.ffn_gate.weight'), 6144, 'BS ffn gate', 'l_ffn_gate', request='dn-fg', proc='BS mv dn-g'),
        *attention.copy(h, 'BS normed2', 'l_normed2', 'BS input', 'l_input'),
        *attention.rotate_in(h, 2048),
        *attention.matvec(h, weight(h, '.ffn_up.weight'), 6144, 'BS ffn up', 'l_ffn_up', request='dn-fu', proc='BS mv dn-u'),
        *h['combine'](h, gated=True, rhs=('BS ffn up', 'l_ffn_up'), input_name='BS ffn gate',
                      input_id='l_ffn_gate', output=('BS ffn', 'l_ffn')),
        *attention.copy(h, 'BS ffn', 'l_ffn', 'BS input', 'l_input'),
        *attention.rotate_in(h, 6144),
        *attention.matvec(h, weight(h, '.ffn_down.weight'), 2048, 'BS ffn out', 'l_ffn_out', request='dn-fd', proc='BS mv dn-d'),
        *h['combine'](h, gated=False, rhs=('BS ffn out', 'l_ffn_out'), input_name='BS residual',
                      input_id='l_residual', output=('BS dn layer out', 'l_dn_layer_out')),
        ('data_deletealloflist', {}, {'LIST': ['BS output', 'l_output']}, {}),
        *attention.copy(h, 'BS dn layer out', 'l_dn_layer_out', 'BS output', 'l_output'),
    ]
    valid = op('operator_and', OPERAND1=eq(length('BS input', 'l_input'), 2048),
                OPERAND2=eq(h['var']('BS step valid'), 1))
    return [conditional(valid, body, [conditional(eq(h['var']('BS step valid'), 1),
                                                          [h['addlist']('invalid dimensions')])])]
