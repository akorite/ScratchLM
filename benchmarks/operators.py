#!/usr/bin/env python3
"""Compare compiled Scratch primitives to native calibration and holdout vectors."""
import argparse
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    fixtures = ROOT / 'fixtures/native/operators'
    results = []
    with tempfile.TemporaryDirectory() as temp:
        project = Path(temp) / 'operators.sb3'
        subprocess.run([sys.executable, str(ROOT / 'tools/generate_project.py'), str(project)], check=True)
        for name, mode in [('rms', 5), ('l2', 6), ('silu', 7)]:
            for step in (0, 1):
                inp = fixtures / f'{name}-{step}-input.f32'
                data = (fixtures / f'{name}-{step}-output.f32').read_bytes()
                native = struct.unpack(f'<{len(data) // 4}f', data)
                env = dict(os.environ, BONSAI_ACTIVATION_F32=str(inp))
                completed = subprocess.run(['node', str(ROOT / 'tests/scratch/run-vm.js'), str(project), '[]', str(mode)],
                                           env=env, capture_output=True, text=True, timeout=120, check=True)
                actual = json.loads(completed.stdout)
                if len(actual) != len(native) or any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in actual):
                    raise RuntimeError(f'{name}: invalid operator output')
                errors = [a - b for a, b in zip(actual, native)]
                norm = math.sqrt(math.fsum(a*a for a in actual) * math.fsum(b*b for b in native))
                results.append({'operator': name, 'step': step, 'role': 'calibration' if step == 0 else 'stateful holdout',
                                'elements': len(native), 'max_abs_error': max(abs(x) for x in errors),
                                'max_relative_error': max(abs(a-b) / max(abs(b), 1e-12) for a, b in zip(actual, native)),
                                'rms_error': math.sqrt(math.fsum(x*x for x in errors) / len(errors)),
                                'cosine': math.fsum(a*b for a, b in zip(actual, native)) / norm if norm else None})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'schema': 'bonsai-operator-errors/v1',
        'vm_revision': '0cb82afe62506f3d9f2ff849c95593f0473b5d2c', 'compiler': True,
        'note': 'Native CPU F32 versus compiled Scratch binary64 intermediates. Diagnostic, not a complete tolerance gate.',
        'results': results}, indent=2) + '\n')


if __name__ == '__main__':
    main()
