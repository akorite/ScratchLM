#!/usr/bin/env python3
"""Measure a full compiled Scratch vocabulary-head scan on a locked input."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import struct
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]

def load_m4_native(fixture, manifest, path):
    if fixture['schema'] != 'bonsai-native-m4-head/v1' or manifest['schema'] != 'bonsai-native-reference/v1':
        raise ValueError('unexpected native manifest schema')
    for key in ('model', 'runtime', 'requested_context', 'vocabulary_size'):
        if fixture[key] != manifest[key]:
            raise ValueError(f'native {key} does not match M4 fixture')
    if (fixture['build']['binary']['sha256'] != manifest['build']['binary']['sha256']
            or manifest['input_tokens'] != fixture['input_tokens'][:1]
            or manifest['vocabulary_size'] != 248320):
        raise ValueError('native build, input tokens, or vocabulary does not match M4 fixture')
    captures = [c for c in manifest['captures'] if c['name'] == 'result_norm' and c['role'] == 'out']
    if len(captures) != 1 or captures[0]['payload']['sha256'] != fixture['post_norm']['sha256']:
        raise ValueError('native result_norm capture does not match M4 fixture')
    step = manifest['steps'][0]
    if step['index'] != 0 or step['token'] != fixture['input_tokens'][0]:
        raise ValueError('native logits step does not match M4 fixture')
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    info = step['logits']
    if len(data) != 248320 * 4 or info['bytes'] != len(data) or info['sha256'] != sha:
        raise ValueError('native full logits size or SHA-256 mismatch')
    values = struct.unpack('<248320f', data)
    if not all(math.isfinite(v) for v in values):
        raise ValueError('native full logits contain nonfinite values')
    order = sorted(range(len(values)), key=lambda i: (-values[i], i))[:len(fixture['top_k'])]
    if ([{'token': i, 'logit': values[i]} for i in order] != fixture['top_k']):
        raise ValueError('native full logits top-k does not match M4 fixture')
    return values, sha


def logit_metrics(logits, native):
    if len(logits) != len(native) or not logits or not all(
            isinstance(v, (int, float)) and math.isfinite(v) for v in (*logits, *native)):
        raise ValueError('incomplete or nonfinite head logits')
    errors = [value - reference for value, reference in zip(logits, native)]
    cosine = math.fsum(value * reference for value, reference in zip(logits, native))
    norm = math.sqrt(math.fsum(value * value for value in logits) *
                     math.fsum(value * value for value in native))
    return {'max_abs_error': max(abs(value) for value in errors),
            'max_relative_error': max(abs(value - reference) / max(abs(reference), 1e-12)
                                      for value, reference in zip(logits, native)),
            'rms_error': math.sqrt(math.fsum(value * value for value in errors) / len(errors)),
            'cosine': cosine / norm if norm else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path)
    parser.add_argument('--bslut4', type=Path)
    parser.add_argument('--bslut5', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--native-projection', type=Path)
    parser.add_argument('--m4-manifest', type=Path)
    parser.add_argument('--native-logits', type=Path, help='full logits from reference.py run')
    parser.add_argument('--native-manifest', type=Path, help='manifest from the same reference.py run')
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--tile-rows', type=int, default=640)
    args = parser.parse_args()
    if args.repetitions < 1 or not 1 <= args.tile_rows <= 7710:
        parser.error('repetitions must be positive and tile-rows in 1..7710')
    if args.m4_manifest:
        if not args.bslut5 and not args.bslut4 and not args.raw:
            parser.error('--m4-manifest requires at least one package flag')
        if not args.native_logits or not args.native_manifest:
            parser.error('--m4-manifest requires --native-logits and --native-manifest')
        m4_manifest = json.loads(args.m4_manifest.read_text())
        pre_norm_path = (args.m4_manifest.parent / m4_manifest['pre_norm']['file']).resolve()
        pre_norm = pre_norm_path.read_bytes()
        if (len(pre_norm) != m4_manifest['pre_norm']['bytes'] or
                hashlib.sha256(pre_norm).hexdigest() != m4_manifest['pre_norm']['sha256']):
            parser.error('M4 pre-norm activation size or SHA-256 mismatch')
        native, native_sha = load_m4_native(m4_manifest, json.loads(args.native_manifest.read_text()),
                                            args.native_logits)
    else:
        if args.native_logits or args.native_manifest:
            parser.error('--native-logits and --native-manifest require --m4-manifest')
        if not args.raw or not args.bslut4 or not args.native_projection:
            parser.error('--raw, --bslut4, and --native-projection are required without --m4-manifest')
        native_bytes = args.native_projection.read_bytes()
        if len(native_bytes) != 248320 * 4:
            parser.error('native projection must contain 248320 little-endian F32 values')
        native = struct.unpack('<248320f', native_bytes)
        native_sha = hashlib.sha256(native_bytes).hexdigest()
    cpu = platform.processor()
    cpuinfo = Path('/proc/cpuinfo')
    if cpuinfo.exists():
        cpu = next((line.split(':', 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                    if line.startswith('model name')), cpu)
    results = []
    checksum = None
    with tempfile.TemporaryDirectory() as temp:
        project = Path(temp) / 'head.sb3'
        subprocess.run([sys.executable, str(ROOT / 'tools/generate_project.py'), str(project), '--benchmark'], check=True)
        project_sha256 = hashlib.sha256(project.read_bytes()).hexdigest()
        cases = [(c, p) for c, p in (('raw', args.raw), ('bslut4', args.bslut4), ('bslut5', args.bslut5)) if p]
        for codec, package in cases:
            runs = []
            for _ in range(args.repetitions):
                env = dict(os.environ, BONSAI_MODEL_PACKAGE=str(package), BONSAI_BENCHMARK='1', BONSAI_TIMEOUT_MS='1800000')
                flag = '--m4' if args.m4_manifest else '--head'
                if args.m4_manifest:
                    env['BONSAI_ACTIVATION_F32'] = str(pre_norm_path)
                start = time.perf_counter()
                completed = subprocess.run(['node', str(ROOT / 'tests/scratch/run-vm.js'), str(project), flag,
                                            '248320', 'token_embd.weight', str(args.tile_rows)], env=env,
                                           capture_output=True, text=True, timeout=1850, check=True)
                wall = time.perf_counter() - start
                result = json.loads(completed.stdout)
                logits = result.pop('output')
                if len(logits) != 248320 or any(not isinstance(x, (float, int)) for x in logits):
                    raise RuntimeError('incomplete or invalid head output')
                digest = hashlib.sha256(struct.pack('<248320f', *logits)).hexdigest()
                if checksum is None:
                    checksum = digest
                if digest != checksum:
                    raise RuntimeError('head output differs across repeated runs or codecs')
                result.update(logit_metrics(logits, native))
                if args.m4_manifest:
                    tokens = result['top_k_tokens']
                    top_errs = [logits[entry['token']] - entry['logit'] for entry in m4_manifest['top_k']]
                    margin = m4_manifest['margin_top1_top2']
                    result.update({'top_k_match': tokens == [entry['token'] for entry in m4_manifest['top_k']],
                                   'greedy_match': result['greedy_token'] == m4_manifest['greedy_token'],
                                   'top_k_max_abs_error': max(abs(x) for x in top_errs),
                                   'margin': result['greedy_logit'] - result['second_logit'],
                                   'native_margin': margin,
                                   'native_margin_exceeds_twice_max_error': margin > 2 * result['max_abs_error']})
                result.update({'wall_seconds': wall, 'rows': len(logits), 'output_f32_sha256': digest})
                runs.append(result)
            times = [run['wall_seconds'] for run in runs]
            results.append({'codec': codec, 'median_wall_seconds': statistics.median(times),
                            'worst_wall_seconds': max(times), 'runs': runs})
    payload = {'schema': 'bonsai-full-head/v1', 'cpu': cpu, 'platform': platform.platform(),
               'turbowarp_vm': '0cb82afe62506f3d9f2ff849c95593f0473b5d2c',
               'compiler': True, 'turbo_mode': True, 'tensor': 'token_embd.weight',
               'input': str(args.m4_manifest) if args.m4_manifest else 'fixtures/native/layer3-q-input.f32',
               'rows': 248320, 'width': 2048,
               'benchmark_project_sha256': project_sha256, 'warp_timer': False,
               'framerate': int(os.environ.get('BONSAI_FRAMERATE', 250)),
               'node_version': subprocess.check_output(['node', '--version'], text=True).strip(),
               'native_projection_sha256': native_sha,
               'native_manifest_sha256': (hashlib.sha256(args.native_manifest.read_bytes()).hexdigest()
                                          if args.m4_manifest else None),
               'tile_rows': args.tile_rows, 'tile_group_bytes': args.tile_rows * 16 * 34,
               'note': ('End-to-end M4 head (output_norm -> signs+Hadamard -> Q8_K -> 248,320-row scan -> greedy).'
                        if args.m4_manifest else
                        'Head scan only; input is a captured transformed activation, not final normalized model output. Node VM, no browser renderer.'),
               'results': results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + '\n')


if __name__ == '__main__':
    main()
