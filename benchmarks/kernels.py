#!/usr/bin/env python3
"""Measure compiled resident-row kernels; no full-model performance claim."""
import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests/scratch'))
from test_transport_integration import add_loader_block


def main():
    parser = argparse.ArgumentParser()
    cpu = platform.processor()
    cpuinfo = Path('/proc/cpuinfo')
    if cpuinfo.exists():
        cpu = next((line.split(':', 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                    if line.startswith('model name')), cpu)
    parser.add_argument('--raw', required=True, type=Path)
    parser.add_argument('--bslut4', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--repetitions', type=int, default=3)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error('repetitions must be positive')
    with tempfile.TemporaryDirectory() as temp:
        results = []
        shapes = [(2048, 'blk.3.attn_q.weight', 'layer3-q-input.f32'),
                  (6144, 'blk.0.ffn_down.weight', 'layer0-ffn-down-input.f32')]
        for width, tensor, fixture in shapes:
            project = Path(temp) / f'resident-{width}.sb3'
            subprocess.run([sys.executable, str(ROOT / 'tools/generate_project.py'), str(project)], check=True)
            add_loader_block(project, width // 128, 'l_scales', tensor=tensor)
            expected_output = None
            for label, package, mode in [('scalar-raw', args.raw, 2), ('raw-lut4', args.raw, 9),
                                         ('bs-lut4', args.bslut4, 9)]:
                for contributions in (1000000, 10000000):
                    iterations = (contributions + width - 1) // width
                    runs = []
                    for _ in range(args.repetitions):
                        env = dict(os.environ, BONSAI_MODEL_PACKAGE=str(package), BONSAI_BENCHMARK='1',
                                   BONSAI_ACTIVATION_F32=str(ROOT / 'fixtures/native' / fixture),
                                   BONSAI_TIMEOUT_MS='600000')
                        completed = subprocess.run(['node', str(ROOT / 'tests/scratch/run-vm.js'), str(project), '--dot', str(mode), str(iterations)],
                                                   env=env, capture_output=True, text=True, timeout=620, check=True)
                        runs.append(json.loads(completed.stdout))
                    if expected_output is None:
                        expected_output = runs[0]['output']
                    if any(run['output'] != expected_output for run in runs):
                        raise RuntimeError('kernel output differs across runs or codecs')
                    times = [run['compute_tick_seconds'] for run in runs]
                    results.append({'kernel': label, 'tensor': tensor, 'width': width, 'iterations': iterations,
                                    'contributions': iterations * width,
                                    'lookups': iterations * width // 4 if mode == 9 else None,
                                    'median_compute_tick_seconds': statistics.median(times),
                                    'worst_compute_tick_seconds': max(times), 'runs': runs})
        payload = {'schema': 'bonsai-resident-kernels/v1', 'cpu': cpu, 'platform': platform.platform(),
                   'turbowarp_vm': '0cb82afe62506f3d9f2ff849c95593f0473b5d2c', 'compiler': True,
                   'turbo_mode': True, 'warp_timer': False, 'row': 0,
                   'node_version': subprocess.check_output(['node', '--version'], text=True).strip(),
                   'note': 'Each resident real row is repeated; Scratch tick timer is coarse. Not full-token or browser performance.',
                   'results': results}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + '\n')


if __name__ == '__main__':
    main()
