#!/usr/bin/env python3
"""Time one mode-42 decode step end-to-end, optionally at a different matvec tile.

The project is generated in-process so `attention.TILE` can be varied without
editing scratch/. The VM's `sensing_timer` is frame-granular (runtime.currentMSecs
updates once per VM step), so per-call timers mis-attribute time when a warp
procedure runs several layers inside one step; this harness therefore reports
external wall time only.
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
import generate_project  # noqa: E402
from scratch import attention, deltanet, lut, transform  # noqa: E402


def zero_fill(h, name, list_id, count, value=0):
    return [('data_deletealloflist', {}, {'LIST': [name, list_id]}, {}),
            h['repeat'](count, [h['addlist'](value, name, list_id)])]


def apply_stubs(stubs):
    """Replace selected stages with shape-preserving no-ops for cost attribution."""
    if 'layers' in stubs:
        attention.layer = lambda h: []
        deltanet.stage = lambda h: []
    if 'head' in stubs:
        original = lut.script
        lut.script = lambda h, head=False: [] if head else original(h, head=head)
    if 'delta' in stubs:
        deltanet.delta = lambda h: zero_fill(h, 'BS gdn out', 'l_gdn_out', 16 * 128)
    if 'conv' in stubs:
        deltanet.conv = lambda h: (zero_fill(h, 'BS conv raw', 'l_conv_raw', 6144, 1)
                                   + zero_fill(h, 'BS conv silu', 'l_conv_silu', 6144, 1))
    if 'attention-core' in stubs:
        attention.attention = lambda h: []
    if 'lut-build' in stubs:
        lut.build_lut = lambda h: []
    if 'hadamard' in stubs:
        transform.hadamard = lambda *args, **kwargs: []


def project_bytes(tile, warp_head=False):
    attention.TILE = tile
    data, name, svg = generate_project.project()
    if warp_head:
        doc = json.loads(data)
        blocks = doc['targets'][0]['blocks']
        for block in blocks.values():
            mutation = block.get('mutation')
            if mutation and mutation.get('proccode') == 'BS head':
                mutation['warp'] = 'true'
        data = json.dumps(doc, ensure_ascii=True, separators=(',', ':')).encode('utf-8')
    return data, name, svg


def write_project(path, data, name, svg):
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for filename, contents in (('project.json', data), (name, svg)):
            info = zipfile.ZipInfo(filename, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, contents)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--repetitions', type=int, default=1)
    parser.add_argument('--tile', type=int, action='append', default=None)
    parser.add_argument('--framerate', type=int, default=250)
    parser.add_argument('--token-id', type=int, default=1237)
    parser.add_argument('--warp-head', action='store_true')
    parser.add_argument('--stub', default='', help='comma list: layers,head,delta,conv,attention-core,lut-build,hadamard')
    args = parser.parse_args()
    tiles = args.tile or [attention.TILE]
    vm = os.environ['BONSAI_TURBOWARP_VM']
    package = os.environ['BONSAI_MODEL_PACKAGE']
    stubs = [item for item in args.stub.split(',') if item]
    apply_stubs(stubs)
    projects, hashes = {}, {}
    with tempfile.TemporaryDirectory() as temp:
        for tile in tiles:
            data, name, svg = project_bytes(tile, args.warp_head)
            project = Path(temp) / f'decode-{tile}.sb3'
            write_project(project, data, name, svg)
            projects[tile] = project
            hashes[tile] = hashlib.sha256(data).hexdigest()
        runs = []
        for _ in range(args.repetitions):
          for tile in tiles:
            env = {key: value for key, value in os.environ.items() if not key.startswith('BONSAI_')}
            env.update(BONSAI_TURBOWARP_VM=vm, BONSAI_MODEL_PACKAGE=package,
                       BONSAI_TOKEN_ID=str(args.token_id), BONSAI_LAYER_POSITION='0',
                       BONSAI_TIMEOUT_MS='1800000', BONSAI_FRAMERATE=str(args.framerate),
                       BONSAI_DEBUG_VARS='BS greedy token,BS greedy logit,BS second logit')
            before = resource.getrusage(resource.RUSAGE_CHILDREN)
            start = time.perf_counter()
            completed = subprocess.run(
                ['node', str(ROOT / 'tests/scratch/run-vm.js'), str(projects[tile]), '[]', '42'],
                env=env, capture_output=True, text=True, timeout=1850, check=True)
            wall = time.perf_counter() - start
            after = resource.getrusage(resource.RUSAGE_CHILDREN)
            cpu = (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)
            values = json.loads(completed.stdout)
            runs.append({'wall_seconds': wall, 'cpu_seconds': cpu, 'tile': tile,
                         'framerate': args.framerate,
                         **{key: values.get(key) for key in
                            ('BS greedy token', 'BS greedy logit', 'BS second logit')}})
    summary = {tile: {'wall_seconds': [r['wall_seconds'] for r in runs if r['tile'] == tile],
                      'cpu_seconds': [r['cpu_seconds'] for r in runs if r['tile'] == tile],
                      'median_wall_seconds': statistics.median(
                          r['wall_seconds'] for r in runs if r['tile'] == tile),
                      'median_cpu_seconds': statistics.median(
                          r['cpu_seconds'] for r in runs if r['tile'] == tile)} for tile in tiles}
    payload = {'schema': 'bonsai-decode-step/v1', 'cpu': next(
        (line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
         if line.startswith('model name')), None),
        'turbowarp_vm': Path(vm).name, 'turbomode': True, 'tiles': tiles, 'stubs': stubs,
        'warp_head': args.warp_head,
        'framerate': args.framerate, 'token_id': args.token_id, 'project_sha256': hashes,
        'source_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted((ROOT / 'scratch').glob('*.py'))},
        'generator_sha256': hashlib.sha256((ROOT / 'tools/generate_project.py').read_bytes()).hexdigest(),
        'note': 'One mode-42 decode step (24 layers + output_norm + head scan + greedy), Node VM, no renderer.',
        'runs': runs, 'summary_by_tile': summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({tile: {key: round(value, 2) for key, value in entry.items() if 'median' in key}
                      for tile, entry in summary.items()}))


if __name__ == '__main__':
    main()
