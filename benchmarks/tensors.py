#!/usr/bin/env python3
"""Measure complete representative projections against pinned CPU captures."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', required=True, type=Path)
    parser.add_argument('--bslut4', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--repetitions', type=int, default=3)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error('repetitions must be positive')
    cases = [(2048, 4096, 'blk.3.attn_q.weight', 'layer3-q-input.f32', 'layer3-q-output.f32'),
             (6144, 2048, 'blk.0.ffn_down.weight', 'layer0-ffn-down-input.f32', 'layer0-ffn-down-output.f32')]
    results = []
    with tempfile.TemporaryDirectory() as temp:
        project = Path(temp) / 'tensors.sb3'
        subprocess.run([sys.executable, str(ROOT / 'tools/generate_project.py'), str(project), '--benchmark'], check=True)
        project_hash = hashlib.sha256(project.read_bytes()).hexdigest()
        for width, rows, tensor, activation, output in cases:
            native = struct.unpack(f'<{rows}f', (ROOT / 'fixtures/native' / output).read_bytes())
            digest = None
            for codec, package in [('raw', args.raw), ('bslut4', args.bslut4)]:
                runs = []
                for _ in range(args.repetitions):
                    env = dict(os.environ, BONSAI_MODEL_PACKAGE=str(package), BONSAI_BENCHMARK='1',
                               BONSAI_ACTIVATION_F32=str(ROOT / 'fixtures/native' / activation), BONSAI_TIMEOUT_MS='600000')
                    start = time.perf_counter()
                    completed = subprocess.run(['node', str(ROOT / 'tests/scratch/run-vm.js'), str(project), '--head',
                                                str(rows), tensor, '120'], env=env, capture_output=True, text=True,
                                               timeout=620, check=True)
                    wall = time.perf_counter() - start
                    run = json.loads(completed.stdout)
                    actual = run.pop('output')
                    if len(actual) != rows or any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in actual):
                        raise RuntimeError('invalid or incomplete projection')
                    checksum = hashlib.sha256(struct.pack(f'<{rows}f', *actual)).hexdigest()
                    if digest is None:
                        digest = checksum
                    if checksum != digest:
                        raise RuntimeError('projection changed across runs or codecs')
                    errors = [a - b for a, b in zip(actual, native)]
                    dot = math.fsum(a * b for a, b in zip(actual, native))
                    norm = math.sqrt(math.fsum(a*a for a in actual) * math.fsum(b*b for b in native))
                    run.update({'wall_seconds': wall, 'rows': rows, 'output_f32_sha256': checksum,
                                'max_abs_error': max(abs(x) for x in errors),
                                'max_relative_error': max(abs(a-b) / max(abs(b), 1e-12) for a, b in zip(actual, native)),
                                'rms_error': math.sqrt(math.fsum(x*x for x in errors) / rows),
                                'cosine': dot / norm if norm else None})
                    runs.append(run)
                times = [run['wall_seconds'] for run in runs]
                results.append({'width': width, 'tensor': tensor, 'codec': codec,
                                'median_wall_seconds': statistics.median(times), 'worst_wall_seconds': max(times), 'runs': runs})
        payload = {'schema': 'bonsai-full-tensors/v1', 'platform': platform.platform(),
                   'vm_revision': '0cb82afe62506f3d9f2ff849c95593f0473b5d2c', 'compiler': True,
                   'turbo_mode': True, 'warp_timer': False, 'tile_rows': 120, 'benchmark_project_sha256': project_hash,
                   'note': 'Captured native transformed input and CPU SIMD output; Node VM without renderer. Scratch tick times are coarse.',
                   'results': results}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + '\n')


if __name__ == '__main__':
    main()
