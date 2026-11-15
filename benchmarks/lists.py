#!/usr/bin/env python3
"""Measure compiled Scratch list/variable operation cost, one phase per project.

TurboWarp's `sensing_timer` reads `runtime.currentMSecs`, which only advances
between VM steps, so a warp loop cannot time itself. Each phase therefore runs
in its own project and is timed externally at two iteration counts; the slope
is the per-operation cost. That slope includes the compiled loop overhead.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import tempfile
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import generate_project as g  # noqa: E402

SVG = b'<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"></svg>'
ASSET = hashlib.md5(SVG).hexdigest()
COSTUMES = [{'name': 'backdrop', 'dataFormat': 'svg', 'assetId': ASSET,
             'md5ext': ASSET + '.svg', 'rotationCenterX': 0, 'rotationCenterY': 0}]


def replace_item(index, value, name, list_id):
    return ('data_replaceitemoflist', {'INDEX': index, 'ITEM': value}, {'LIST': [name, list_id]}, {})


g.replace_item = replace_item


def phase_body(phase, iterations):
    x, idx, a = g.var('BS x'), g.var('BS idx'), ('BS a', 'l_a')
    if phase == 'calc':
        return [g.repeat(iterations, [g.change('BS x', 1)])]
    if phase == 'add':
        return [('data_deletealloflist', {}, {'LIST': list(a)}, {}),
                g.repeat(iterations, [g.addlist(x, *a)])]
    if phase == 'replace':
        return [('data_deletealloflist', {}, {'LIST': list(a)}, {}),
                g.addlist(0, *a),
                g.repeat(iterations, [replace_item(1, x, *a)])]
    if phase == 'add_wide':
        # the kernel's real destinations are 2,048 or 6,144 items, not one
        return [('data_deletealloflist', {}, {'LIST': list(a)}, {}),
                g.repeat(iterations, [g.addlist(x, *a)])]
    if phase == 'replace_wide':
        return [('data_deletealloflist', {}, {'LIST': list(a)}, {}),
                g.repeat(2048, [g.addlist(0, *a)]),
                g.repeat(iterations, [replace_item(g.op('operator_add', NUM1=g.op(
                    'operator_mod', NUM1=idx, NUM2=2048), NUM2=1), x, *a), g.change('BS idx', 1)])]
    if phase == 'item':
        # 303,264 is the largest bslut5 LUT (6144-wide activation), so the read
        # touches the same list size the kernel uses; the index varies every read.
        return [('data_deletealloflist', {}, {'LIST': list(a)}, {}),
                g.repeat(303264, [g.addlist(x, *a)]),
                g.repeat(iterations, [g.setvar('BS x', g.item(*a, g.op(
                    'operator_add', NUM1=g.op('operator_mod', NUM1=idx, NUM2=303264), NUM2=1))),
                    g.change('BS idx', 1)])]
    if phase == 'f32':
        return [g.setvar('BS f32 input', 1.0000001), g.repeat(iterations, [g.call_f32()])]
    raise ValueError(phase)


def build(phase, iterations):
    graph = g.Blocks()
    hat = graph.add('event_whenflagclicked', top=True)
    graph.blocks[hat]['next'] = graph.stack(
        [g.setvar('BS done', 0), g.call_proc('bench'), g.setvar('BS done', 1)], hat)
    g.add_procedure(graph, 'bench', phase_body(phase, iterations), warp='true')
    if phase == 'f32':
        g.f32_procedure(graph)
    names = ('BS done', 'BS mode', 'BS x', 'BS idx', 'BS f32 input', 'BS f32 output',
             'BS f32 abs', 'BS f32 exponent', 'BS f32 power', 'BS f32 unit', 'BS f32 integer',
             'BS f32 fraction', 'BS f32 sign')
    variables = {'v_' + name.replace(' ', '_'): [name, 0] for name in names}
    lists = {'l_a': ['BS a', []], 'l_input': ['BS input', []], 'l_output': ['BS output', []]}
    if phase == 'f32':
        lists['l_f32_powers'] = ['BS f32 powers', [2 ** i for i in range(-149, 128)]]
    stage = {'isStage': True, 'name': 'Stage', 'variables': variables, 'lists': lists,
             'broadcasts': {}, 'blocks': graph.blocks, 'comments': {}, 'currentCostume': 0,
             'costumes': COSTUMES, 'sounds': [], 'volume': 100, 'layerOrder': 0, 'tempo': 60,
             'videoTransparency': 50, 'videoState': 'off', 'textToSpeechLanguage': None}
    return json.dumps({'targets': [stage], 'monitors': [], 'extensions': [],
                       'meta': {'semver': '3.0.0', 'vm': '0.2.0', 'agent': 'BonsaiScratch'}},
                      ensure_ascii=True, separators=(',', ':')).encode('utf-8')


def run(project, env):
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.perf_counter()
    subprocess.run(['node', str(ROOT / 'tests/scratch/run-vm.js'), str(project), '[]', '0'],
                   env=env, capture_output=True, text=True, timeout=900, check=True)
    wall = time.perf_counter() - start
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    return wall, (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--iterations', type=int, default=20000000)
    parser.add_argument('--baseline', type=int, default=1000)
    parser.add_argument('--repetitions', type=int, default=2)
    args = parser.parse_args()
    env = {key: value for key, value in os.environ.items() if not key.startswith('BONSAI_')}
    env.update(BONSAI_TURBOWARP_VM=os.environ['BONSAI_TURBOWARP_VM'],
               BONSAI_FRAMERATE='250', BONSAI_TIMEOUT_MS='900000')
    results = []
    with tempfile.TemporaryDirectory() as temp:
        for phase in ('calc', 'add', 'replace', 'item', 'add_wide', 'replace_wide', 'f32'):
            projects = {}
            for count in (args.baseline, args.iterations):
                data = build(phase, count)
                project = Path(temp) / f'{phase}-{count}.sb3'
                projects[count] = project
                with zipfile.ZipFile(project, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
                    archive.writestr('project.json', data)
                    archive.writestr(ASSET + '.svg', SVG)
            runs = {args.baseline: [], args.iterations: []}
            for _ in range(args.repetitions):
                for count in (args.baseline, args.iterations):
                    runs[count].append(run(projects[count], env))
            base_cpu = statistics.median(cpu for _, cpu in runs[args.baseline])
            full_cpu = statistics.median(cpu for _, cpu in runs[args.iterations])
            per_op = (full_cpu - base_cpu) / (args.iterations - args.baseline)
            results.append({'phase': phase, 'iterations': args.iterations, 'baseline': args.baseline,
                            'baseline_cpu_seconds': [cpu for _, cpu in runs[args.baseline]],
                            'cpu_seconds': [cpu for _, cpu in runs[args.iterations]],
                            'baseline_wall_seconds': [wall for wall, _ in runs[args.baseline]],
                            'wall_seconds': [wall for wall, _ in runs[args.iterations]],
                            'ns_per_op': 1e9 * per_op})
            print(phase, round(1e9 * per_op, 1), 'ns/op')
    payload = {'schema': 'bonsai-list-ops/v1', 'vm': os.environ['BONSAI_TURBOWARP_VM'],
               'framerate': 250, 'note': ('One warp loop per project, timed externally at two iteration '
                                          'counts; the slope includes compiled loop overhead. '
                                          'sensing_timer cannot time warp loops (frame-granular clock).'),
               'results': results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + '\n')


if __name__ == '__main__':
    main()
