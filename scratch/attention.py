"""Scratch blocks for one full-attention decoder layer.

The layer reads its 2048-wide input from `BS input`, the zero-based position
from `BS layer position`, and writes `BS output`. Weight tiles come from the
selected package through the loader; all activation math stays in Scratch.
Order mirrors the native graph: attn_norm, Q with gate, per-head q/k norm,
RoPE, causal softmax over the KV cache, gated output projection, residual,
post norm, SwiGLU MLP, residual.
"""

from scratch import embedding, lut, numeric, transform

Q_NORM = 'blk.3.attn_q_norm.weight'
K_NORM = 'blk.3.attn_k_norm.weight'
TILE = 64


def rotate_in(h, sign_width):
    """Forward normalized Hadamard rotation with the model's explicit signs."""
    name, list_id = ('BS signs', 'l_signs') if sign_width == 2048 else ('BS signs wide', 'l_signs_wide')
    return [('data_deletealloflist', {}, {'LIST': ['BS output', 'l_output']}, {}),
            *transform.make(12, h, signs_name=name, signs_id=list_id),
            *copy(h, 'BS output', 'l_output', 'BS input', 'l_input')]


def rope_list(h, source, source_id, dest, dest_id, position):
    """Rotate a 2048/512-wide list with the position prefixed, matching mode 30's layout."""
    var, item, length, setvar, addlist, conditional, repeat, change = (
        h[name] for name in ('var', 'item', 'length', 'setvar', 'addlist', 'conditional',
                             'repeat', 'change'))
    idx = var('BS index')
    prefill = [('data_deletealloflist', {}, {'LIST': ['BS rope in', 'l_rope_in']}, {}),
               addlist(position, 'BS rope in', 'l_rope_in'),
               setvar('BS index', 1),
               repeat(length(source, source_id), [addlist(item(source, source_id, idx), 'BS rope in', 'l_rope_in'),
                                                  change('BS index', 1)]),
               ('data_deletealloflist', {}, {'LIST': ['BS rope out', 'l_rope_out']}, {}),
               *h['rope'](h, input_name='BS rope in', input_id='l_rope_in',
                          output_name='BS rope out', output_id='l_rope_out'),
               ('data_deletealloflist', {}, {'LIST': [dest, dest_id]}, {}),
               setvar('BS index', 1),
               repeat(length('BS rope out', 'l_rope_out'), [addlist(item('BS rope out', 'l_rope_out', idx),
                                                                     dest, dest_id),
                                                            change('BS index', 1)])]
    return prefill


def matvec(h, tensor, rows, dest_name, dest_id, request='layer'):
    """Project `BS input` through a PQ2 tensor, one tile of rows at a time."""
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
    status = ('bonsailoader_status', {'ID': request}, {}, False)
    copied = ('bonsailoader_copied', {'ID': request}, {}, False)
    groups = div(length('BS input', 'l_input'), 128)
    tile_rows = v('proj tile')
    tile_groups = mul(tile_rows, groups)
    # The loader replaces l_packed/l_scales with the current tile, so row indices are tile-relative.
    row_base = mul(v('tile row'), groups)
    row = [setvar('BS total', 0), setvar('BS dot group', 0),
           repeat(groups, lut.group_body(h, add(row_base, v('dot group')), v('dot group'))
                  + [change('BS dot group', 1)]),
           addlist(v('total'), dest_name, dest_id), change('BS tile row', 1)]
    chunk = [setvar('BS proj tile', sub(rows, v('proj scanned'))),
             conditional(gt(tile_rows, TILE), [setvar('BS proj tile', TILE)]),
             ('bonsailoader_loadGroups',
              {'ID': request, 'TENSOR': tensor, 'START': mul(v('proj scanned'), groups),
               'COUNT': tile_groups, 'Q': 'l_packed', 'SCALE': 'l_scales'}, {}, {}),
             ('control_wait', {'DURATION': 0}, {}, {}),
             # A refill may need several frames; wait until the request settles.
             ('control_repeat_until',
              {'CONDITION': op('operator_not', OPERAND=op('operator_equals', OPERAND1=status, OPERAND2='pending'))},
              {}, {'SUBSTACK': [('control_wait', {'DURATION': 0}, {}, {})]}),
             conditional(op('operator_and', OPERAND1=eq(status, 'ok'),
                            OPERAND2=eq(copied, tile_groups)),
                         [setvar('BS tile row', 0), repeat(tile_rows, row),
                          change('BS proj scanned', tile_rows)],
                         [setvar('BS proj valid', 0)])]
    return [('data_deletealloflist', {}, {'LIST': [dest_name, dest_id]}, {}),
            setvar('BS proj scanned', 0), setvar('BS proj valid', 1),
            conditional(eq(op('operator_mod', NUM1=length('BS input', 'l_input'), NUM2=256), 0),
                        [('data_deletealloflist', {}, {'LIST': ['BS q8', 'l_q8']}, {}),
                         ('data_deletealloflist', {}, {'LIST': ['BS lut', 'l_lut']}, {}),
                         *h['q8_script']('BS q8', 'l_q8'),
                         *lut.codec_tables(h), *lut.build_lut(h),
                         repeat(floor(div(sub(add(rows, TILE), 1), TILE)),
                                [conditional(eq(v('proj valid'), 1), chunk)])],
                        [setvar('BS proj valid', 0)])]


def copy(h, source, source_id, dest, dest_id):
    """Replace the destination list with a copy of the source list."""
    var, item, length, setvar, addlist, repeat, change = (
        h[name] for name in ('var', 'item', 'length', 'setvar', 'addlist', 'repeat', 'change'))
    idx = var('BS index')
    return [('data_deletealloflist', {}, {'LIST': [dest, dest_id]}, {}),
            setvar('BS index', 1),
            repeat(length(source, source_id), [addlist(item(source, source_id, idx), dest, dest_id),
                                               change('BS index', 1)])]


def append(h, source, source_id, dest, dest_id):
    """Append the source list to the destination list."""
    var, item, length, setvar, addlist, repeat, change = (
        h[name] for name in ('var', 'item', 'length', 'setvar', 'addlist', 'repeat', 'change'))
    idx = var('BS index')
    return [setvar('BS index', 1),
            repeat(length(source, source_id), [addlist(item(source, source_id, idx), dest, dest_id),
                                               change('BS index', 1)])]


def weighted_norm(h, weight_name, weight_id, dest_name, dest_id):
    """RMS-norm `BS input` into `dest`, then multiply by a dense weight list."""
    var, item, length, setvar, addlist, repeat, change = (
        h[name] for name in ('var', 'item', 'length', 'setvar', 'addlist', 'repeat', 'change'))
    idx = var('BS index')
    scale = [h['setvar']('BS f32 input', h['op']('operator_multiply',
                         NUM1=item('BS output', 'l_output', idx), NUM2=item(weight_name, weight_id, idx))),
             h['call_f32'](),
             addlist(var('BS f32 output'), dest_name, dest_id),
             change('BS index', 1)]
    return ([('data_deletealloflist', {}, {'LIST': ['BS output', 'l_output']}, {})] + numeric.make(5, h) +
            [('data_deletealloflist', {}, {'LIST': [dest_name, dest_id]}, {}),
             setvar('BS index', 1), repeat(length('BS output', 'l_output'), scale)])


def head_norm(h, source, source_id, dest, dest_id, weight_name, weight_id, width, heads, gate=None):
    """Weighted per-head RMS norm, optionally multiplied by a SiLU gate list."""
    var, op, item, length, setvar, addlist, conditional, repeat, change = (
        h[name] for name in ('var', 'op', 'item', 'length', 'setvar', 'addlist', 'conditional',
                             'repeat', 'change'))
    add = lambda a, b: op('operator_add', NUM1=a, NUM2=b)
    sub = lambda a, b: op('operator_subtract', NUM1=a, NUM2=b)
    mul = lambda a, b: op('operator_multiply', NUM1=a, NUM2=b)
    div = lambda a, b: op('operator_divide', NUM1=a, NUM2=b)
    gt = lambda a, b: op('operator_gt', OPERAND1=a, OPERAND2=b)
    math = lambda name, value: ('operator_mathop', {'NUM': value}, {'OPERATOR': [name, None]}, False)
    v = lambda name: var('BS ' + name)
    index, total = v('layer index'), v('layer sum')
    at = lambda position: add(add(mul(v('layer head'), width), index), position)
    value = item(source, source_id, at(1))
    scale = [setvar('BS layer sum', add(total, mul(value, value))), change('BS layer index', 1)]
    emit = [h['setvar']('BS f32 input', div(value, v('layer divisor'))), h['call_f32'](),
            h['setvar']('BS f32 input', mul(var('BS f32 output'),
                                            item(weight_name, weight_id, add(index, 1)))), h['call_f32']()]
    if gate:
        gated = item(gate[0], gate[1], at(1))
        emit += [conditional(gt(gated, 0),
                             [h['setvar']('BS f32 input', mul(var('BS f32 output'),
                                              mul(gated, div(1, add(1, math('e ^', mul(-1, gated)))))))],
                             [h['setvar']('BS f32 input', mul(var('BS f32 output'),
                                              mul(gated, div(math('e ^', gated), add(1, math('e ^', gated))))))]),
                 h['call_f32']()]
    emit += [addlist(var('BS f32 output'), dest, dest_id), change('BS layer index', 1)]
    return [('data_deletealloflist', {}, {'LIST': [dest, dest_id]}, {}),
            setvar('BS layer head', 0),
            repeat(heads, [setvar('BS layer index', 0), setvar('BS layer sum', 0),
                           repeat(width, scale),
                           setvar('BS layer divisor', math('sqrt', add(div(total, width),
                                                                       9.999999974752427e-7))),
                           setvar('BS layer index', 0), repeat(width, emit),
                           change('BS layer head', 1)])]


def gate_mul(h, vec, vec_id, gate, gate_id, dest, dest_id):
    """Multiply a vector by sigmoid(gate) elementwise (full-attention output gate)."""
    var, op, item, length, setvar, addlist, conditional, repeat, change = (
        h[name] for name in ('var', 'op', 'item', 'length', 'setvar', 'addlist', 'conditional',
                             'repeat', 'change'))
    add = lambda a, b: op('operator_add', NUM1=a, NUM2=b)
    sub = lambda a, b: op('operator_subtract', NUM1=a, NUM2=b)
    mul = lambda a, b: op('operator_multiply', NUM1=a, NUM2=b)
    div = lambda a, b: op('operator_divide', NUM1=a, NUM2=b)
    gt = lambda a, b: op('operator_gt', OPERAND1=a, OPERAND2=b)
    math = lambda name, value: ('operator_mathop', {'NUM': value}, {'OPERATOR': [name, None]}, False)
    idx = var('BS index')
    value = item(vec, vec_id, idx)
    g = item(gate, gate_id, idx)
    body = [conditional(gt(g, 0),
                        [h['setvar']('BS f32 input', div(value, add(1, math('e ^', mul(-1, g)))))],
                        [h['setvar']('BS f32 input', mul(value, div(math('e ^', g), add(1, math('e ^', g))))) ]),
            h['call_f32'](), addlist(var('BS f32 output'), dest, dest_id), change('BS index', 1)]
    return [setvar('BS index', 1), repeat(length(vec, vec_id), body)]


def attention(h):
    """One query head over the cached keys and values."""
    var, op, item, setvar, addlist, conditional, repeat, change = (
        h[name] for name in ('var', 'op', 'item', 'setvar', 'addlist', 'conditional', 'repeat', 'change'))
    add = lambda a, b: op('operator_add', NUM1=a, NUM2=b)
    sub = lambda a, b: op('operator_subtract', NUM1=a, NUM2=b)
    mul = lambda a, b: op('operator_multiply', NUM1=a, NUM2=b)
    div = lambda a, b: op('operator_divide', NUM1=a, NUM2=b)
    gt = lambda a, b: op('operator_gt', OPERAND1=a, OPERAND2=b)
    math = lambda name, value: ('operator_mathop', {'NUM': value}, {'OPERATOR': [name, None]}, False)
    v = lambda name: var('BS ' + name)
    head, token, index, total, maximum, weight = map(v, (
        'layer head', 'layer token', 'layer index', 'layer sum', 'layer maximum', 'layer weight'))
    query = lambda j: item('BS q', 'l_q', add(mul(head, 256), add(j, 1)))
    kv_head = h['floor'](div(head, 4))
    kv = lambda cache: lambda j: item(cache[0], cache[1], add(add(mul(token, 512),
                                                              mul(kv_head, 256)), add(j, 1)))
    score = lambda t: item('BS scores', 'l_scores', add(t, 1))
    scores = [setvar('BS layer sum', 0), setvar('BS layer index', 0),
              repeat(256, [h['setvar']('BS f32 input', add(total, h['op'](
                  'operator_multiply', NUM1=query(index), NUM2=kv(('BS k cache', 'l_k_cache'))(index)))),
                  h['call_f32'](), setvar('BS layer sum', var('BS f32 output')),
                  change('BS layer index', 1)]),
              h['setvar']('BS f32 input', mul(total, 0.0625)), h['call_f32'](),
              addlist(var('BS f32 output'), 'BS scores', 'l_scores'),
              change('BS layer token', 1)]
    scan = [conditional(gt(score(token), maximum), [setvar('BS maximum', score(token))]),
            change('BS layer token', 1)]
    exponent = math('e ^', sub(score(token), maximum))
    normalize = [setvar('BS layer sum', add(total, exponent)), change('BS layer token', 1)]
    emit = [addlist(div(math('e ^', sub(score(token), maximum)), total), 'BS weights', 'l_weights'),
            change('BS layer token', 1)]
    weighted = [setvar('BS layer weight', item('BS weights', 'l_weights', add(token, 1))),
                setvar('BS layer index', 0),
                repeat(256, [h['setvar']('BS f32 input', add(
                    item('BS attn', 'l_attn', add(mul(head, 256), add(index, 1))),
                    mul(weight, kv(('BS v cache', 'l_v_cache'))(index)))),
                    h['call_f32'](),
                    transform.replace_item(add(mul(head, 256), add(index, 1)), var('BS f32 output'),
                                           'BS attn', 'l_attn'),
                    change('BS layer index', 1)]),
                change('BS layer token', 1)]
    tokens = add(v('layer position'), 1)
    return [
        ('data_deletealloflist', {}, {'LIST': ['BS scores', 'l_scores']}, {}),
        ('data_deletealloflist', {}, {'LIST': ['BS weights', 'l_weights']}, {}),
        setvar('BS layer token', 0), repeat(tokens, scores),
        setvar('BS maximum', score(0)), setvar('BS layer token', 0), repeat(tokens, scan),
        setvar('BS layer sum', 0), setvar('BS layer token', 0), repeat(tokens, normalize),
        setvar('BS layer token', 0), repeat(tokens, emit),
        setvar('BS layer token', 0), repeat(tokens, weighted),
        change('BS layer head', 1),
    ]


def layer(h):
    var, op, floor, item, length, setvar, addlist, conditional, repeat, change = (
        h[name] for name in ('var', 'op', 'floor', 'item', 'length', 'setvar', 'addlist', 'conditional',
                             'repeat', 'change'))
    add = lambda a, b: op('operator_add', NUM1=a, NUM2=b)
    mul = lambda a, b: op('operator_multiply', NUM1=a, NUM2=b)
    eq = lambda a, b: op('operator_equals', OPERAND1=a, OPERAND2=b)
    lt = lambda a, b: op('operator_lt', OPERAND1=a, OPERAND2=b)
    v = lambda name: var('BS ' + name)
    codec = ('bonsailoader_codec', {}, {}, False)
    split = [setvar('BS layer head', 0),
             repeat(8, [setvar('BS index', 0),
                        repeat(256, [addlist(item('BS q full', 'l_q_full',
                                                  add(add(mul(v('layer head'), 512), v('index')), 1)),
                                             'BS q', 'l_q'),
                                     addlist(item('BS q full', 'l_q_full',
                                                  add(add(add(mul(v('layer head'), 512), v('index')), 257), 0)),
                                             'BS gate', 'l_gate'),
                                     change('BS index', 1)]),
                        change('BS layer head', 1)])]
    valid = op('operator_and', OPERAND1=eq(length('BS input', 'l_input'), 2048),
               OPERAND2 = op('operator_and', OPERAND1=eq(v('layer position'), floor(v('layer position'))),
                             OPERAND2=op('operator_and', OPERAND1=op('operator_gt', OPERAND1=v('layer position'),
                                                                    OPERAND2=-1),
                                         OPERAND2=lt(v('layer position'), 128))))
    body = [
        embedding.set_codec_from_loader(h, codec),
        *embedding.load_dense(h, 'qnorm', Q_NORM, 'BS q norm', 'l_q_norm', 256, [setvar('BS proj valid', 0)]),
        *embedding.load_dense(h, 'knorm', K_NORM, 'BS k norm', 'l_k_norm', 256, [setvar('BS proj valid', 0)]),
        *embedding.load_dense(h, 'normw', 'blk.3.attn_norm.weight', 'BS norm w', 'l_norm_w', 2048,
                              [setvar('BS proj valid', 0)]),
        *embedding.load_dense(h, 'postw', 'blk.3.post_attention_norm.weight', 'BS post w', 'l_post_w', 2048,
                              [setvar('BS proj valid', 0)]),
        *embedding.load_dense(h, 'signs', 'prism.hadamard.signs.2048', 'BS signs', 'l_signs', 2048,
                              [setvar('BS proj valid', 0)]),
        *embedding.load_dense(h, 'signsw', 'prism.hadamard.signs.6144', 'BS signs wide', 'l_signs_wide', 6144,
                              [setvar('BS proj valid', 0)]),
        *weighted_norm(h, 'BS norm w', 'l_norm_w', 'BS normed', 'l_normed'),
        *copy(h, 'BS input', 'l_input', 'BS layer input', 'l_layer_input'),
        *copy(h, 'BS normed', 'l_normed', 'BS input', 'l_input'),
        *rotate_in(h, 2048),
        *matvec(h, 'blk.3.attn_q.weight', 4096, 'BS q full', 'l_q_full', request='layer-q'),
        conditional(eq(v('proj valid'), 1), split),
        *head_norm(h, 'BS q', 'l_q', 'BS qn', 'l_qn', 'BS q norm', 'l_q_norm', 256, 8),
        *rope_list(h, 'BS qn', 'l_qn', 'BS q', 'l_q', v('layer position')),
        *copy(h, 'BS normed', 'l_normed', 'BS input', 'l_input'),
        *rotate_in(h, 2048),
        *matvec(h, 'blk.3.attn_k.weight', 512, 'BS k', 'l_k', request='layer-k'),
        *head_norm(h, 'BS k', 'l_k', 'BS kn', 'l_kn', 'BS k norm', 'l_k_norm', 256, 2),
        *rope_list(h, 'BS kn', 'l_kn', 'BS k', 'l_k', v('layer position')),
        *copy(h, 'BS normed', 'l_normed', 'BS input', 'l_input'),
        *rotate_in(h, 2048),
        *matvec(h, 'blk.3.attn_v.weight', 512, 'BS v', 'l_v', request='layer-v'),
        *append(h, 'BS k', 'l_k', 'BS k cache', 'l_k_cache'),
        *append(h, 'BS v', 'l_v', 'BS v cache', 'l_v_cache'),
        ('data_deletealloflist', {}, {'LIST': ['BS attn', 'l_attn']}, {}),
        setvar('BS index', 0),
        repeat(2048, [addlist(0, 'BS attn', 'l_attn'), change('BS index', 1)]),
        setvar('BS layer head', 0), repeat(8, attention(h)),
        *gate_mul(h, 'BS attn', 'l_attn', 'BS gate', 'l_gate', 'BS attn gated', 'l_attn_gated'),
        *copy(h, 'BS attn gated', 'l_attn_gated', 'BS input', 'l_input'),
        *rotate_in(h, 2048),
        *matvec(h, 'blk.3.attn_output.weight', 2048, 'BS attn out', 'l_attn_out', request='layer-o'),
        *h['combine'](h, gated=False, rhs=('BS attn out', 'l_attn_out'), input_name='BS layer input',
                      input_id='l_layer_input', output=('BS residual', 'l_residual')),
        *copy(h, 'BS residual', 'l_residual', 'BS input', 'l_input'),
        *weighted_norm(h, 'BS post w', 'l_post_w', 'BS normed', 'l_normed'),
        *copy(h, 'BS normed', 'l_normed', 'BS input', 'l_input'),
        *rotate_in(h, 2048),
        *matvec(h, 'blk.3.ffn_gate.weight', 6144, 'BS ffn gate', 'l_ffn_gate', request='layer-g'),
        *copy(h, 'BS normed', 'l_normed', 'BS input', 'l_input'),
        *rotate_in(h, 2048),
        *matvec(h, 'blk.3.ffn_up.weight', 6144, 'BS ffn up', 'l_ffn_up', request='layer-u'),
        *h['combine'](h, gated=True, rhs=('BS ffn up', 'l_ffn_up'), input_name='BS ffn gate',
                      input_id='l_ffn_gate', output=('BS ffn', 'l_ffn')),
        *copy(h, 'BS ffn', 'l_ffn', 'BS input', 'l_input'),
        *rotate_in(h, 6144),
        *matvec(h, 'blk.3.ffn_down.weight', 2048, 'BS ffn out', 'l_ffn_out', request='layer-d'),
        *h['combine'](h, gated=False, rhs=('BS ffn out', 'l_ffn_out'), input_name='BS residual',
                      input_id='l_residual', output=('BS layer out', 'l_layer_out')),
        ('data_deletealloflist', {}, {'LIST': ['BS output', 'l_output']}, {}),
        *copy(h, 'BS layer out', 'l_layer_out', 'BS output', 'l_output'),
    ]
    return [conditional(valid, body, [addlist('invalid dimensions')])]
