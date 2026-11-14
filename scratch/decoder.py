"""Scratch blocks for the 24-layer decoder."""

from scratch import attention, embedding, lut, transform

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
    # Readout: final output_norm, the full head scan over the vocabulary, then greedy selection.
    # Start from the token embedding when a token id is supplied, else the vector in `BS input`.
    codec = ('bonsailoader_codec', {}, {}, False)
    # One script run is one decode step: the prompt list supplies the token for the first
    # runs, later runs feed the generated token back; all state lives in the lists.
    step, prompt_len = h['var']('BS step'), h['length']('BS prompt tokens', 'l_prompt')
    choose = [h['conditional'](h['op']('operator_lt', OPERAND1=step, OPERAND2=prompt_len),
                [setvar('BS token id', h['item']('BS prompt tokens', 'l_prompt',
                                                add(step, 1))), setvar('BS decode by id', 1)],
                [h['conditional'](h['op']('operator_gt', OPERAND1=step, OPERAND2=0),
                                  [setvar('BS token id', h['var']('BS greedy token'))], []),
                 h['conditional'](h['op']('operator_gt', OPERAND1=h['var']('BS token id'), OPERAND2=-1),
                                  [setvar('BS decode by id', 1)], [setvar('BS decode by id', 0)])])]
    entry = [setvar('BS layer position', step), *choose,
             embedding.set_codec_from_loader(h, codec),
             *embedding.load_dense(h, 'decode-signs', 'prism.hadamard.signs.2048', 'BS signs', 'l_signs',
                                   2048, [h['addlist']('load failed')])]
    entry += [h['conditional'](h['op']('operator_equals', OPERAND1=h['var']('BS decode by id'), OPERAND2=1),
                               [h['call_embed'](),
                                *attention.copy(h, 'BS output', 'l_output', 'BS input', 'l_input')], [])]
    readout = [
        *attention.copy(h, 'BS output', 'l_output', 'BS input', 'l_input'),
        *embedding.load_dense(h, 'outnorm', 'output_norm.weight', 'BS norm weights', 'l_norm_weights',
                              2048, [h['addlist']('load failed')]),
        ('data_deletealloflist', {}, {'LIST': ['BS output', 'l_output']}, {}),
        *embedding.output_norm(h),
        # token_embd is stored inverse-rotated: signs first, then the Hadamard transform,
        # matching the primitive pipeline's order.
        *transform.apply_signs(h),
        *transform.hadamard(h),
        *attention.copy(h, 'BS output', 'l_output', 'BS input', 'l_input'),
        ('data_deletealloflist', {}, {'LIST': ['BS output', 'l_output']}, {}),
        h['call_head'](),
        *lut.greedy_select(h),
        h['addlist'](h['var']('BS greedy token')),
    ]
    return [setvar('BS dn slot', 0), setvar('BS attn slot', 0), *entry, *run, *readout,
            h['change']('BS step', 1)]
