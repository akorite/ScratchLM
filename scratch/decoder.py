"""Scratch blocks for the 24-layer decoder."""

from scratch import attention

ATTENTION_LAYERS = (3, 7, 11, 15, 19, 23)
STATE_SIZE = 262144          # one DeltaNet layer: 16 heads x 128 x 128
CONV_STATE = 3 * 6144        # one DeltaNet layer: three convolution taps
KV_LAYER = 128 * 512         # one attention layer: 128 tokens x 2 heads x 256


def stack(h):
    """Run the locked layer schedule over the token vector in `BS input`."""
    var, op, setvar, change = (h[n] for n in ('var', 'op', 'setvar', 'change'))
    add = lambda a, b: op('operator_add', NUM1=a, NUM2=b)
    mul = lambda a, b: op('operator_multiply', NUM1=a, NUM2=b)
    v = lambda name: var('BS ' + name)
    run = []
    for layer in range(24):
        run += [setvar('BS layer', layer), setvar('BS state base', 0), setvar('BS kv base', 0),
                setvar('BS conv base', 0)]
        if layer in ATTENTION_LAYERS:
            run += [setvar('BS kv base', mul(v('attn slot'), KV_LAYER)),
                    h['call_attention'](),
                    setvar('BS attn slot', add(v('attn slot'), 1))]
        else:
            run += [setvar('BS state base', mul(v('dn slot'), STATE_SIZE)),
                    setvar('BS conv base', mul(v('dn slot'), CONV_STATE)),
                    h['call_deltanet'](),
                    setvar('BS dn slot', add(v('dn slot'), 1))]
        if layer < 23:
            run += [*attention.copy(h, 'BS output', 'l_output', 'BS input', 'l_input')]
    return [setvar('BS dn slot', 0), setvar('BS attn slot', 0), *run]
