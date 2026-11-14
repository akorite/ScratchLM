"""Scratch blocks for one Gated DeltaNet decoder layer.

Stage 1 covers the input projections and the short convolution; later stages add
the delta rule, the gated norm and the output projection. All arithmetic stays
in Scratch; weight tiles come from the selected package through the loader.
"""

from scratch import attention, embedding

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
    # Taps 0..2 are history; tap 3 is this token's fused QKV projection.
    sample = [conditional(eq(tap, TAPS - 1),
                          [addlist(item('BS qkv', 'l_qkv', add(channel, 1)), 'BS taps', 'l_taps')],
                          [addlist(item('BS conv state', 'l_conv_state',
                                        add(mul(tap, CONV_WIDTH), add(channel, 1))), 'BS taps', 'l_taps')]),
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
    """Roll the circular buffer: the new history is old taps 1..2 plus this token."""
    var, op, item, setvar, addlist, repeat, change = (
        h[name] for name in ('var', 'op', 'item', 'setvar', 'addlist', 'repeat', 'change'))
    add = lambda a, b: op('operator_add', NUM1=a, NUM2=b)
    mul = lambda a, b: op('operator_multiply', NUM1=a, NUM2=b)
    v = lambda name: var('BS ' + name)
    roll = [setvar('BS dn tap', 0),
            repeat(TAPS - 2, [addlist(item('BS conv state', 'l_conv_state',
                                           add(mul(add(v('dn tap'), 1), CONV_WIDTH), add(v('dn channel'), 1))),
                                    'BS next state', 'l_next_state'),
                              change('BS dn tap', 1)]),
            addlist(item('BS qkv', 'l_qkv', add(v('dn channel'), 1)), 'BS next state', 'l_next_state'),
            change('BS dn channel', 1)]
    return [('data_deletealloflist', {}, {'LIST': ['BS next state', 'l_next_state']}, {}),
            setvar('BS dn channel', 0), repeat(CONV_WIDTH, roll),
            *attention.copy(h, 'BS next state', 'l_next_state', 'BS conv state', 'l_conv_state')]


def stage(h):
    """Projections and convolution for one token, writing `BS conv silu`."""
    var, op, setvar, length, conditional = (h[name] for name in ('var', 'op', 'setvar', 'length', 'conditional'))
    eq = lambda a, b: op('operator_equals', OPERAND1=a, OPERAND2=b)
    codec = ('bonsailoader_codec', {}, {}, False)
    body = [
        embedding.set_codec_from_loader(h, codec),
        *embedding.load_dense(h, 'dn-norm', 'blk.0.attn_norm.weight', 'BS norm w', 'l_norm_w', 2048,
                              [setvar('BS proj valid', 0)]),
        *embedding.load_dense(h, 'dn-conv', 'blk.0.ssm_conv1d.weight', 'BS conv w', 'l_conv_w',
                              TAPS * CONV_WIDTH, [setvar('BS proj valid', 0)]),
        *embedding.load_dense(h, 'dn-signs', 'prism.hadamard.signs.2048', 'BS signs', 'l_signs', 2048,
                              [setvar('BS proj valid', 0)]),
        *attention.weighted_norm(h, 'BS norm w', 'l_norm_w', 'BS normed', 'l_normed'),
        *attention.copy(h, 'BS input', 'l_input', 'BS layer input', 'l_layer_input'),
        *attention.copy(h, 'BS normed', 'l_normed', 'BS input', 'l_input'),
        *attention.rotate_in(h, 2048),
        *attention.matvec(h, 'blk.0.attn_qkv.weight', CONV_WIDTH, 'BS qkv', 'l_qkv', request='dn-qkv'),
        *conv(h),
        *shift_state(h),
    ]
    valid = eq(length('BS input', 'l_input'), 2048)
    return [conditional(valid, body, [h['addlist']('invalid dimensions')])]
