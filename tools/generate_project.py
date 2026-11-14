#!/usr/bin/env python3
"""Generate a reproducible Scratch project containing visible numeric blocks."""
import hashlib
import json
from pathlib import Path
import struct
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scratch import lut, numeric, text


class Blocks:
    def __init__(self):
        self.blocks = {}
        self.counter = 0

    def add(self, opcode, parent=None, inputs=None, fields=None, top=False):
        self.counter += 1
        name = f'b{self.counter:04d}'
        self.blocks[name] = {'opcode': opcode, 'next': None, 'parent': parent,
                             'inputs': inputs or {}, 'fields': fields or {},
                             'shadow': False, 'topLevel': top}
        if top:
            self.blocks[name].update(x=40, y=40)
        return name

    def expression(self, value, parent):
        if isinstance(value, tuple):
            if value[0] == 'var':
                return [3, [12, value[1], value[2]], [4, '0']]
            opcode, inputs, fields, boolean = value
            block = self.add(opcode, parent, fields=fields)
            self.blocks[block]['inputs'] = {name: self.expression(child, block) for name, child in inputs.items()}
            return [2, block] if boolean else [3, block, [4, '0']]
        if isinstance(value, str):
            return [1, [10, value]]
        return [1, [4, str(value)]]

    def stack(self, instructions, parent):
        first = previous = None
        for opcode, inputs, fields, branches in instructions:
            block = self.add(opcode, parent if previous is None else previous, fields=fields)
            if opcode == 'procedures_call':
                self.blocks[block]['mutation'] = {'tagName': 'mutation', 'children': [],
                                                  'proccode': 'BS f32', 'argumentids': '[]', 'warp': 'true'}
            if previous:
                self.blocks[previous]['next'] = block
            if first is None:
                first = block
            self.blocks[block]['inputs'] = {name: self.expression(value, block) for name, value in inputs.items()}
            for name, body in branches.items():
                child = self.stack(body, block)
                self.blocks[block]['inputs'][name] = [2, child] if child else [1, None]
            previous = block
        return first


def var(name):
    return ('var', name, 'v_' + name.replace('BS ', '').replace(' ', '_'))


def op(opcode, **inputs):
    return (opcode, inputs, {}, opcode in ('operator_equals', 'operator_lt', 'operator_gt', 'operator_and', 'operator_or'))


def floor(value):
    return ('operator_mathop', {'NUM': value}, {'OPERATOR': ['floor', None]}, False)


def item(list_name, list_id, index):
    return ('data_itemoflist', {'INDEX': index}, {'LIST': [list_name, list_id]}, False)


def length(list_name, list_id):
    return ('data_lengthoflist', {}, {'LIST': [list_name, list_id]}, False)


def setvar(name, value):
    return ('data_setvariableto', {'VALUE': value}, {'VARIABLE': [name, var(name)[2]]}, {})


def addlist(value, name='BS output', list_id='l_output'):
    return ('data_addtolist', {'ITEM': value}, {'LIST': [name, list_id]}, {})


def conditional(condition, yes, no=None):
    if no is None:
        return ('control_if', {'CONDITION': condition}, {}, {'SUBSTACK': yes})
    return ('control_if_else', {'CONDITION': condition}, {}, {'SUBSTACK': yes, 'SUBSTACK2': no})


def repeat(count, body):
    return ('control_repeat', {'TIMES': count}, {}, {'SUBSTACK': body})


def change(name, amount):
    return ('data_changevariableby', {'VALUE': amount}, {'VARIABLE': [name, var(name)[2]]}, {})


def call_f32():
    return ('procedures_call', {}, {}, {})


def f32_procedure(graph):
    definition = graph.add('procedures_definition', top=True)
    graph.blocks[definition]['x'] = 1200
    prototype = graph.add('procedures_prototype', definition)
    graph.blocks[prototype]['shadow'] = True
    graph.blocks[prototype]['mutation'] = {'tagName': 'mutation', 'children': [], 'proccode': 'BS f32',
                                           'argumentids': '[]', 'argumentnames': '[]',
                                           'argumentdefaults': '[]', 'warp': 'true'}
    graph.blocks[definition]['inputs']['custom_block'] = [1, prototype]
    x, absolute, exponent, power, unit, number, fraction, sign = map(var, (
        'BS f32 input', 'BS f32 abs', 'BS f32 exponent', 'BS f32 power',
        'BS f32 unit', 'BS f32 integer', 'BS f32 fraction', 'BS f32 sign'))
    log_abs = ('operator_mathop', {'NUM': absolute}, {'OPERATOR': ['ln', None]}, False)
    power_at = lambda index: item('BS f32 powers', 'l_f32_powers', index)
    rounded = op('operator_or',
                 OPERAND1=op('operator_gt', OPERAND1=fraction, OPERAND2=0.5),
                 OPERAND2=op('operator_and',
                             OPERAND1=op('operator_equals', OPERAND1=fraction, OPERAND2=0.5),
                             OPERAND2=op('operator_equals', OPERAND1=op('operator_mod', NUM1=number, NUM2=2), OPERAND2=1)))
    body = [
        setvar('BS f32 output', 0),
        conditional(op('operator_gt', OPERAND1=('operator_mathop', {'NUM': x}, {'OPERATOR': ['abs', None]}, False), OPERAND2=0), [
            setvar('BS f32 abs', ('operator_mathop', {'NUM': x}, {'OPERATOR': ['abs', None]}, False)),
            conditional(op('operator_lt', OPERAND1=x, OPERAND2=0), [setvar('BS f32 sign', -1)], [setvar('BS f32 sign', 1)]),
            setvar('BS f32 exponent', floor(op('operator_divide', NUM1=log_abs, NUM2=0.6931471805599453))),
            conditional(op('operator_lt', OPERAND1=exponent, OPERAND2=-149), [setvar('BS f32 exponent', -149)]),
            conditional(op('operator_gt', OPERAND1=exponent, OPERAND2=127), [setvar('BS f32 exponent', 127)]),
            setvar('BS f32 power', power_at(op('operator_add', NUM1=exponent, NUM2=150))),
            conditional(op('operator_and', OPERAND1=op('operator_gt', OPERAND1=exponent, OPERAND2=-149),
                           OPERAND2=op('operator_lt', OPERAND1=absolute, OPERAND2=power)), [change('BS f32 exponent', -1)]),
            conditional(op('operator_or',
                           OPERAND1=op('operator_gt', OPERAND1=absolute, OPERAND2=op('operator_multiply', NUM1=2, NUM2=power)),
                           OPERAND2=op('operator_equals', OPERAND1=absolute, OPERAND2=op('operator_multiply', NUM1=2, NUM2=power))),
                        [change('BS f32 exponent', 1)]),
            conditional(op('operator_lt', OPERAND1=exponent, OPERAND2=-126),
                        [setvar('BS f32 unit', 2**-149)],
                        [setvar('BS f32 unit', power_at(op('operator_add', NUM1=exponent, NUM2=127)))]),
            setvar('BS f32 integer', floor(op('operator_divide', NUM1=absolute, NUM2=unit))),
            setvar('BS f32 fraction', op('operator_subtract', NUM1=op('operator_divide', NUM1=absolute, NUM2=unit), NUM2=number)),
            conditional(rounded, [change('BS f32 integer', 1)]),
            setvar('BS f32 output', op('operator_multiply', NUM1=sign, NUM2=op('operator_multiply', NUM1=number, NUM2=unit))),
        ]),
    ]
    graph.blocks[definition]['next'] = graph.stack(body, definition)


def q8_script(output_name='BS output', output_id='l_output'):
    emit = lambda value: addlist(value, output_name, output_id)
    index, start, maximum, absmax, x, scale, rounded = map(var, (
        'BS index', 'BS group start', 'BS maximum', 'BS absmax', 'BS x', 'BS iscale', 'BS rounded'))
    absolute = ('operator_mathop', {'NUM': x}, {'OPERATOR': ['abs', None]}, False)
    result = var('BS f32 output')
    fraction = var('BS fraction')
    scan = [setvar('BS x', item('BS input', 'l_input', index)),
            conditional(op('operator_gt', OPERAND1=absolute, OPERAND2=absmax),
                        [setvar('BS absmax', absolute), setvar('BS maximum', x)]),
            change('BS index', 1)]
    round_up = op('operator_or', OPERAND1=op('operator_gt', OPERAND1=fraction, OPERAND2=0.5),
                  OPERAND2=op('operator_and', OPERAND1=op('operator_equals', OPERAND1=fraction, OPERAND2=0.5),
                              OPERAND2=op('operator_equals', OPERAND1=op('operator_mod', NUM1=rounded, NUM2=2), OPERAND2=1)))
    quantize = [
        setvar('BS x', item('BS input', 'l_input', index)),
        setvar('BS f32 input', op('operator_multiply', NUM1=scale, NUM2=x)), call_f32(),
        setvar('BS rounded', floor(result)),
        setvar('BS fraction', op('operator_subtract', NUM1=result, NUM2=rounded)),
        conditional(round_up, [change('BS rounded', 1)]),
        conditional(op('operator_gt', OPERAND1=rounded, OPERAND2=127), [setvar('BS rounded', 127)]),
        emit(rounded), change('BS index', 1),
    ]
    nonzero = [
        setvar('BS f32 input', op('operator_divide', NUM1=-127, NUM2=maximum)), call_f32(),
        setvar('BS iscale', result),
        setvar('BS f32 input', op('operator_divide', NUM1=1, NUM2=scale)), call_f32(), emit(result),
        setvar('BS index', start), repeat(256, quantize),
    ]
    group = [
        setvar('BS maximum', 0), setvar('BS absmax', 0), setvar('BS index', start), repeat(256, scan),
        conditional(op('operator_equals', OPERAND1=absmax, OPERAND2=0),
                    [emit(0), repeat(256, [emit(0)])], nonzero),
        change('BS group start', 256),
    ]
    return [setvar('BS group start', 1),
            conditional(op('operator_equals', OPERAND1=op('operator_mod', NUM1=length('BS input', 'l_input'), NUM2=256), OPERAND2=0),
                        [repeat(op('operator_divide', NUM1=length('BS input', 'l_input'), NUM2=256), group)],
                        [emit('invalid group length')])]


def scalar_dot_script():
    group, position, word, exponent, mantissa, sign, weight_scale, subtotal, total, divisor, base = map(var, (
        'BS dot group', 'BS dot position', 'BS word', 'BS exponent', 'BS mantissa', 'BS sign',
        'BS weight scale', 'BS subtotal', 'BS total', 'BS divisor', 'BS base'))
    g0 = op('operator_subtract', NUM1=group, NUM2=1)
    packed_index = op('operator_add', NUM1=op('operator_add', NUM1=op('operator_multiply', NUM1=g0, NUM2=32),
                                             NUM2=floor(op('operator_divide', NUM1=position, NUM2=4))), NUM2=1)
    q8_base = op('operator_multiply', NUM1=floor(op('operator_divide', NUM1=g0, NUM2=2)), NUM2=257)
    q8_index = op('operator_add', NUM1=op('operator_add', NUM1=q8_base, NUM2=2),
                  NUM2=op('operator_add', NUM1=op('operator_multiply', NUM1=op('operator_mod', NUM1=g0, NUM2=2), NUM2=128),
                          NUM2=position))
    power_index = op('operator_add', NUM1=op('operator_mod', NUM1=position, NUM2=4), NUM2=1)
    digit = op('operator_mod', NUM1=floor(op('operator_divide', NUM1=item('BS packed bytes', 'l_packed', packed_index),
                                            NUM2=divisor)), NUM2=base)
    product = op('operator_multiply', NUM1=op('operator_subtract', NUM1=digit, NUM2=1),
                 NUM2=item('BS q8', 'l_q8', q8_index))
    inner = [
        conditional(op('operator_equals', OPERAND1=var('BS codec'), OPERAND2=0),
                    [setvar('BS divisor', item('BS raw powers', 'l_raw_powers', power_index)), setvar('BS base', 4)],
                    [setvar('BS divisor', item('BS trit powers', 'l_trit_powers', power_index)), setvar('BS base', 3)]),
        setvar('BS subtotal', op('operator_add', NUM1=subtotal, NUM2=product)),
        change('BS dot position', 1),
    ]
    normal_scale = op('operator_multiply', NUM1=sign, NUM2=op('operator_multiply',
                      NUM1=op('operator_add', NUM1=1024, NUM2=mantissa),
                      NUM2=item('BS powers', 'l_powers', op('operator_add', NUM1=exponent, NUM2=1))))
    subnormal_scale = op('operator_multiply', NUM1=sign, NUM2=op('operator_multiply', NUM1=mantissa, NUM2=2**-24))
    group_body = [
        setvar('BS word', item('BS weight scales', 'l_scales', group)),
        setvar('BS exponent', op('operator_mod', NUM1=floor(op('operator_divide', NUM1=word, NUM2=1024)), NUM2=32)),
        setvar('BS mantissa', op('operator_mod', NUM1=word, NUM2=1024)),
        conditional(op('operator_gt', OPERAND1=word, OPERAND2=32767), [setvar('BS sign', -1)], [setvar('BS sign', 1)]),
        conditional(op('operator_equals', OPERAND1=exponent, OPERAND2=31), [setvar('BS dot valid', 0)], [
            conditional(op('operator_equals', OPERAND1=exponent, OPERAND2=0),
                        [setvar('BS weight scale', subnormal_scale)], [setvar('BS weight scale', normal_scale)]),
            setvar('BS subtotal', 0), setvar('BS dot position', 0), repeat(128, inner),
            setvar('BS f32 input', op('operator_multiply', NUM1=weight_scale,
                   NUM2=item('BS q8', 'l_q8', op('operator_add', NUM1=q8_base, NUM2=1)))), call_f32(),
            setvar('BS f32 input', op('operator_multiply', NUM1=var('BS f32 output'), NUM2=subtotal)), call_f32(),
            setvar('BS f32 input', op('operator_add', NUM1=total, NUM2=var('BS f32 output'))), call_f32(),
            setvar('BS total', var('BS f32 output')),
        ]),
        change('BS dot group', 1),
    ]
    valid_shape = op('operator_and',
        OPERAND1=op('operator_equals', OPERAND1=length('BS input', 'l_input'),
                    OPERAND2=op('operator_multiply', NUM1=length('BS weight scales', 'l_scales'), NUM2=128)),
        OPERAND2=op('operator_and',
            OPERAND1=op('operator_equals', OPERAND1=length('BS packed bytes', 'l_packed'),
                        OPERAND2=op('operator_multiply', NUM1=length('BS weight scales', 'l_scales'), NUM2=32)),
            OPERAND2=op('operator_equals', OPERAND1=op('operator_mod', NUM1=length('BS weight scales', 'l_scales'), NUM2=2), OPERAND2=0)))
    timer = ('sensing_timer', {}, {}, False)
    work = [setvar('BS benchmark start', timer), ('data_deletealloflist', {}, {'LIST': ['BS q8', 'l_q8']}, {})]
    work += q8_script('BS q8', 'l_q8')
    work += [setvar('BS benchmark quant', op('operator_subtract', NUM1=timer, NUM2=var('BS benchmark start'))),
             setvar('BS benchmark setup', 0), setvar('BS benchmark start', timer),
             repeat(var('BS iterations'), [setvar('BS dot group', 1), setvar('BS dot valid', 1), setvar('BS total', 0),
                                           repeat(length('BS weight scales', 'l_scales'), group_body)]),
             setvar('BS benchmark compute', op('operator_subtract', NUM1=timer, NUM2=var('BS benchmark start'))),
             conditional(op('operator_equals', OPERAND1=var('BS dot valid'), OPERAND2=1),
                         [addlist(total)], [addlist('nonfinite')])]
    return [conditional(valid_shape, work, [addlist('invalid dimensions')])]


def project(with_text=False, with_benchmark=False):
    graph = Blocks()
    hat = graph.add('event_whenflagclicked', top=True)
    w, e, m, sign, idx = map(var, ('BS word', 'BS exponent', 'BS mantissa', 'BS sign', 'BS index'))
    scaled = op('operator_multiply', NUM1=sign, NUM2=op('operator_multiply', NUM1=op('operator_add', NUM1=1024, NUM2=m), NUM2=item('BS powers', 'l_powers', op('operator_add', NUM1=e, NUM2=1))))
    subnormal = op('operator_multiply', NUM1=sign, NUM2=op('operator_multiply', NUM1=m, NUM2=2**-24))
    loop = [
        setvar('BS word', item('BS input', 'l_input', idx)),
        setvar('BS exponent', op('operator_mod', NUM1=floor(op('operator_divide', NUM1=w, NUM2=1024)), NUM2=32)),
        setvar('BS mantissa', op('operator_mod', NUM1=w, NUM2=1024)),
        ('control_if_else', {'CONDITION': op('operator_gt', OPERAND1=w, OPERAND2=32767)}, {}, {'SUBSTACK': [setvar('BS sign', -1)], 'SUBSTACK2': [setvar('BS sign', 1)]}),
        ('control_if_else', {'CONDITION': op('operator_equals', OPERAND1=e, OPERAND2=31)}, {}, {
            'SUBSTACK': [addlist('nonfinite')],
            'SUBSTACK2': [('control_if_else', {'CONDITION': op('operator_equals', OPERAND1=e, OPERAND2=0)}, {}, {'SUBSTACK': [addlist(subnormal)], 'SUBSTACK2': [addlist(scaled)]})]}),
        ('data_changevariableby', {'VALUE': 1}, {'VARIABLE': ['BS index', 'v_index']}, {}),
    ]
    fallback = [repeat(length('BS input', 'l_input'), loop)]
    if with_benchmark:
        fallback = [conditional(op('operator_equals', OPERAND1=var('BS mode'), OPERAND2=11),
                                lut.script(globals(), head=True), fallback)]
    if with_text:
        for mode in (21, 20):
            fallback = [conditional(op('operator_equals', OPERAND1=var('BS mode'), OPERAND2=mode),
                                    text.make(mode, globals()), fallback)]
    fallback = [conditional(op('operator_equals', OPERAND1=var('BS mode'), OPERAND2=9),
                            lut.script(globals()), fallback)]
    for mode in reversed((4, 5, 6, 7, 8, 10)):
        fallback = [conditional(op('operator_equals', OPERAND1=var('BS mode'), OPERAND2=mode),
                                numeric.make(mode, globals()), fallback)]
    script = [
        ('data_deletealloflist', {}, {'LIST': ['BS output', 'l_output']}, {}),
        setvar('BS done', 0), setvar('BS index', 1),
        conditional(op('operator_equals', OPERAND1=var('BS mode'), OPERAND2=1), q8_script(), [
            conditional(op('operator_equals', OPERAND1=var('BS mode'), OPERAND2=2), scalar_dot_script(), [
                conditional(op('operator_equals', OPERAND1=var('BS mode'), OPERAND2=3), [
                    repeat(length('BS input', 'l_input'), [
                        setvar('BS f32 input', item('BS input', 'l_input', idx)), call_f32(),
                        addlist(var('BS f32 output')), change('BS index', 1),
                    ]),
                ], fallback),
            ]),
        ]),
        setvar('BS done', 1),
    ]
    graph.blocks[hat]['next'] = graph.stack(script, hat)
    f32_procedure(graph)
    svg = (b'<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"></svg>' if with_text else
           b'<svg xmlns="http://www.w3.org/2000/svg" width="480" height="360"></svg>')
    asset = hashlib.md5(svg).hexdigest()
    names = ('BS done', 'BS mode', 'BS word', 'BS exponent', 'BS mantissa', 'BS sign', 'BS index',
             'BS group start', 'BS maximum', 'BS absmax', 'BS x', 'BS iscale', 'BS rounded', 'BS fraction',
             'BS f32 input', 'BS f32 output', 'BS f32 abs', 'BS f32 exponent', 'BS f32 power',
             'BS f32 unit', 'BS f32 integer', 'BS f32 fraction', 'BS f32 sign',
             'BS codec', 'BS dot group', 'BS dot position', 'BS weight scale', 'BS subtotal',
             'BS total', 'BS divisor', 'BS base', 'BS dot valid', 'BS lut base', 'BS lut domain',
             'BS lut chunk', 'BS lut pattern', 'BS lut sum', 'BS lut power', 'BS lut digit',
             'BS iterations', 'BS benchmark start', 'BS benchmark quant', 'BS benchmark setup', 'BS benchmark compute')
    if with_benchmark:
        names += ('BS head rows', 'BS head tensor', 'BS tile size', 'BS tile count', 'BS tile row',
                  'BS scanned', 'BS head valid', 'BS cancel', 'BS benchmark transfer', 'BS groups per row')
    if with_text:
        names += ('BS text index', 'BS text unit', 'BS text next', 'BS text code', 'BS text equal',
                  'BS text needle', 'BS text result')
    variables = {var(name)[2]: [name, 0] for name in names}
    variables[var('BS iterations')[2]][1] = 1
    if with_benchmark:
        variables[var('BS head rows')[2]][1] = 248320
        variables[var('BS head tensor')[2]][1] = 'token_embd.weight'
        variables[var('BS tile size')[2]][1] = 1920
        variables[var('BS mode')[2]][1] = 11
    stage = {'isStage': True, 'name': 'Stage', 'variables': variables,
             'lists': {'l_input': ['BS input', [0, 1, 15360, 31744]], 'l_output': ['BS output', []],
                       'l_powers': ['BS powers', [2**(i - 25) for i in range(31)]],
                       'l_f32_powers': ['BS f32 powers', [2**i for i in range(-149, 128)]],
                       'l_packed': ['BS packed bytes', []], 'l_scales': ['BS weight scales', []],
                       'l_q8': ['BS q8', []], 'l_raw_powers': ['BS raw powers', [1, 4, 16, 64]],
                       'l_trit_powers': ['BS trit powers', [1, 3, 9, 27]], 'l_lut': ['BS lut', []]},
             'broadcasts': {}, 'blocks': graph.blocks, 'comments': {}, 'currentCostume': 0,
             'costumes': [{'name': 'backdrop', 'dataFormat': 'svg', 'assetId': asset,
                           'md5ext': asset + '.svg', 'rotationCenterX': 240, 'rotationCenterY': 180}],
             'sounds': [], 'volume': 100, 'layerOrder': 0, 'tempo': 60,
             'videoTransparency': 50, 'videoState': 'off', 'textToSpeechLanguage': None}
    if with_benchmark:
        fixture = Path(__file__).resolve().parents[1] / 'fixtures/native/layer3-q-input.f32'
        stage['lists']['l_input'][1] = list(struct.unpack('<2048f', fixture.read_bytes()))
    targets = [stage]
    if with_text:
        stage['broadcasts'] = {'bs-lookup': 'BS lookup'}
        costume_number = ('looks_costumenumbername', {}, {'NUMBER_NAME': ['number', None]}, False)
        for high in range(256):
            lookup = Blocks()
            event = lookup.add('event_whenbroadcastreceived', fields={'BROADCAST_OPTION': ['BS lookup', 'bs-lookup']}, top=True)
            body = [('looks_switchcostumeto', {'COSTUME': 1}, {}, {}),
                    ('looks_switchcostumeto', {'COSTUME': var('BS text needle')}, {}, {}),
                    conditional(op('operator_gt', OPERAND1=costume_number, OPERAND2=1),
                                [setvar('BS text result', op('operator_add', NUM1=high * 256,
                                                             NUM2=op('operator_subtract', NUM1=costume_number, NUM2=2)))])]
            lookup.blocks[event]['next'] = lookup.stack(body, event)
            targets.append({'isStage': False, 'name': f'BS units {high:03d}', 'variables': {}, 'lists': {},
                            'broadcasts': {}, 'blocks': lookup.blocks, 'comments': {}, 'currentCostume': 0,
                            'costumes': text.costume_data(asset, high), 'sounds': [], 'volume': 100,
                            'layerOrder': high + 1, 'visible': False, 'x': 0, 'y': 0, 'size': 100,
                            'direction': 90, 'draggable': False, 'rotationStyle': 'all around'})
    content = {'targets': targets, 'monitors': [], 'extensions': ['bonsailoader'] if with_benchmark else [],
               'meta': {'semver': '3.0.0', 'vm': '0.2.0', 'agent': 'BonsaiScratch'}}
    return json.dumps(content, ensure_ascii=True, separators=(',', ':')).encode('utf-8'), asset + '.svg', svg


def main():
    if len(sys.argv) not in (2, 3) or (len(sys.argv) == 3 and sys.argv[2] not in ('--text', '--benchmark')):
        raise SystemExit('usage: generate_project.py OUTPUT.sb3 [--text|--benchmark]')
    data, name, svg = project(with_text='--text' in sys.argv, with_benchmark='--benchmark' in sys.argv)
    with zipfile.ZipFile(Path(sys.argv[1]), 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for filename, contents in (('project.json', data), (name, svg)):
            info = zipfile.ZipInfo(filename, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, contents)


if __name__ == '__main__':
    main()
