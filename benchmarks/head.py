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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', required=True, type=Path)
    parser.add_argument('--bslut4', required=True, type=Path)
    parser.add_argument('--bslut5', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--native-projection', required=True, type=Path)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--tile-rows', type=int, default=1280)
    args = parser.parse_args()
    if args.repetitions < 1 or not 1 <= args.tile_rows <= 7710:
        parser.error('repetitions must be positive and tile-rows in 1..7710')
    native_bytes = args.native_projection.read_bytes()
    if len(native_bytes) != 248320 * 4:
        parser.error('native projection must contain 248320 little-endian F32 values')
    native = struct.unpack('<248320f', native_bytes)
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
        cases = [('raw', args.raw), ('bslut4', args.bslut4)]
        if args.bslut5:
            cases.append(('bslut5', args.bslut5))
        for codec, package in cases:
            runs = []
            for _ in range(args.repetitions):
                env = dict(os.environ, BONSAI_MODEL_PACKAGE=str(package), BONSAI_BENCHMARK='1', BONSAI_TIMEOUT_MS='1800000')
                start = time.perf_counter()
                completed = subprocess.run(['node', str(ROOT / 'tests/scratch/run-vm.js'), str(project), '--head',
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
                errors = [value - reference for value, reference in zip(logits, native)]
                cosine = math.fsum(value * reference for value, reference in zip(logits, native))
                norm = math.sqrt(math.fsum(value * value for value in logits) *
                                 math.fsum(value * value for value in native))
                result.update({'max_abs_error': max(abs(value) for value in errors),
                               'max_relative_error': max(abs(value - reference) / max(abs(reference), 1e-12)
                                                         for value, reference in zip(logits, native)),
                               'rms_error': math.sqrt(math.fsum(value * value for value in errors) / len(errors)),
                               'cosine': cosine / norm if norm else None})
                result.update({'wall_seconds': wall, 'rows': len(logits), 'output_f32_sha256': digest})
                runs.append(result)
            times = [run['wall_seconds'] for run in runs]
            results.append({'codec': codec, 'median_wall_seconds': statistics.median(times),
                            'worst_wall_seconds': max(times), 'runs': runs})
    payload = {'schema': 'bonsai-full-head/v1', 'cpu': cpu, 'platform': platform.platform(),
               'turbowarp_vm': '0cb82afe62506f3d9f2ff849c95593f0473b5d2c',
               'compiler': True, 'turbo_mode': True, 'tensor': 'token_embd.weight',
               'input': 'fixtures/native/layer3-q-input.f32', 'rows': 248320, 'width': 2048,
              'benchmark_project_sha256': project_sha256, 'warp_timer': False,
              'framerate': int(os.environ.get('BONSAI_FRAMERATE', 250)),
              'node_version': subprocess.check_output(['node', '--version'], text=True).strip(),
               'native_projection_sha256': hashlib.sha256(native_bytes).hexdigest(),
               'tile_rows': args.tile_rows, 'tile_group_bytes': args.tile_rows * 16 * 34,
               'note': 'Head scan only; input is a captured transformed activation, not final normalized model output. Node VM, no browser renderer.',
               'results': results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + '\n')


if __name__ == '__main__':
    main()
