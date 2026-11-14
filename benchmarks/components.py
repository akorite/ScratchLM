#!/usr/bin/env python3
"""Component-level cost attribution with interleaved rounds.

Runs one DeltaNet layer (mode 41) and one head scan (mode 11) under variants
that stub a single stage, and interleaves the variants round by round so host
drift cancels. Wall and child CPU time are both recorded; the host has two
CPUs, so contention inflates both and only paired differences are trusted.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
import generate_project  # noqa: E402
from scratch import attention, deltanet, lut  # noqa: E402


def zero_fill(h, name, list_id, count, value=0):
    return [('data_deletealloflist', {}, {'LIST': [name, list_id]}, {}),
            h['repeat'](count, [h['addlist'](value, name, list_id)])]


ORIGINAL = {'lut': lut.build_lut, 'q8': generate_project.q8_script,
            'delta': deltanet.delta, 'f32': generate_project.f32_procedure}


def apply_stubs(stubs, tile):
    """Set the tile and replace only the requested stages; restore all others."""
    attention.TILE = tile
    lut.build_lut = (lambda h: []) if 'lut-build' in stubs else ORIGINAL['lut']
    generate_project.q8_script = (lambda *a, **k: []) if 'q8' in stubs else ORIGINAL['q8']
    deltanet.delta = ((lambda h: zero_fill(h, 'BS gdn out', 'l_gdn_out', 16 * 128))
                      if 'delta' in stubs else ORIGINAL['delta'])
    define = generate_project.add_procedure
    generate_project.f32_procedure = ((lambda graph: define(graph, 'BS f32', [], warp='true'))
                                      if 'f32' in stubs else ORIGINAL['f32'])


def project(tile, stubs, benchmark):
    apply_stubs(stubs, tile)
    data, name, svg = generate_project.project(with_benchmark=benchmark)
    if benchmark and 'warp-head' in stubs:
        doc = json.loads(data)
        for block in doc['targets'][0]['blocks'].values():
            mutation = block.get('mutation')
            if mutation and mutation.get('proccode') == 'BS head':
                mutation['warp'] = 'true'
        data = json.dumps(doc, ensure_ascii=True, separators=(',', ':')).encode('utf-8')
    return data, name, svg


def write(path, data, name, svg):
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for filename, contents in (('project.json', data), (name, svg)):
            info = zipfile.ZipInfo(filename, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, contents)


def run(cmd, env):
    with open(os.devnull, 'wb') as null:
        start = time.perf_counter()
        completed = subprocess.run(cmd, env=env, stdout=null, stderr=null, timeout=900)
    wall = time.perf_counter() - start
    if completed.returncode:
        raise RuntimeError(f'{cmd} failed: {completed.returncode}')
    return wall


def cpu_run(cmd, env):
    import resource
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    wall = run(cmd, env)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    return wall, (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--rounds', type=int, default=3)
    args = parser.parse_args()
    vm = os.environ['BONSAI_TURBOWARP_VM']
    package = os.environ['BONSAI_MODEL_PACKAGE']
    layers = [('tile64', 64, []), ('tile640', 640, []), ('tile640-no-lut-build', 640, ['lut-build']),
              ('tile640-no-q8', 640, ['q8']), ('tile640-no-delta', 640, ['delta']),
              ('tile640-no-f32', 640, ['f32'])]
    heads = [('head-base', []), ('head-warp', ['warp-head'])]
    cases = {}
    results = {}
    with tempfile.TemporaryDirectory() as temp:
        for label, tile, stubs in layers:
            data, name, svg = project(tile, stubs, False)
            path = Path(temp) / f'{label}.sb3'
            write(path, data, name, svg)
            cases[label] = (path, '41')
        for label, stubs in heads:
            data, name, svg = project(640, stubs, True)
            path = Path(temp) / f'{label}.sb3'
            write(path, data, name, svg)
            cases[label] = (path, 'head')
        base_env = {key: value for key, value in os.environ.items() if not key.startswith('BONSAI_')}
        base_env.update(BONSAI_TURBOWARP_VM=vm, BONSAI_MODEL_PACKAGE=package, BONSAI_FRAMERATE='250',
                        BONSAI_TIMEOUT_MS='900000', BONSAI_LAYER='0', BONSAI_LAYER_POSITION='0',
                        BONSAI_ACTIVATION_F32='fixtures/native/layer3-q-input.f32')
        orders = [list(cases)] + [list(cases)[::-1]] * (args.rounds - 1)
        for index, order in enumerate(orders):
            for label in order:
                path, kind = cases[label]
                if kind == '41':
                    cmd = ['node', str(ROOT / 'tests/scratch/run-vm.js'), str(path), '[]', '41']
                else:
                    cmd = ['node', str(ROOT / 'tests/scratch/run-vm.js'), str(path), '--head', '248320',
                           'token_embd.weight', '640']
                wall, cpu = cpu_run(cmd, base_env)
                results.setdefault(label, []).append({'round': index, 'wall_seconds': wall, 'cpu_seconds': cpu})
                print(label, round(wall, 2), round(cpu, 2))
    summary = {label: {'median_wall_seconds': statistics.median(r['wall_seconds'] for r in runs),
                       'median_cpu_seconds': statistics.median(r['cpu_seconds'] for r in runs)}
               for label, runs in results.items()}
    payload = {'schema': 'bonsai-components/v1', 'vm': Path(vm).name, 'rounds': args.rounds,
               'cpu_count': os.cpu_count(),
               'note': ('One DeltaNet layer (mode 41) or the full head scan (mode 11, 248,320 rows, tile 640); '
                        'variants differ only by a stubbed stage. Rounds alternate order; host has two CPUs.'),
               'summary': summary, 'results': results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
