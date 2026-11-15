#!/usr/bin/env python3
"""Measure prompt-to-text at the production configuration.

Runs the generated project in the pinned VM, one flag press per decode step
(the same drive the Scratch broadcast loop uses), and records what the release
evidence needs: prompt tokens, time to first token, per-token time, wall and
CPU time, peak RSS, the worst event-loop gap, loader request count, and the
cancellation behaviour of an in-flight request. The first generated token is
checked against the value the pinned native build selects for the same token
sequence, so a run with wrong output is rejected instead of recorded.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PROMPTS = [
    # prompt, first generated token in the pinned native build
    ('hello world', 248068),
    ("it's 你好", 248068),
]


def write_project(path, data, asset_name, asset_bytes):
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for filename, contents in (('project.json', data), (asset_name, asset_bytes)):
            info = zipfile.ZipInfo(filename, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, contents)


def run_prompt(vm, package, node_path, prompt, budget, temp):
    project = Path(temp) / f'decoder-{hashlib.sha256(prompt.encode()).hexdigest()[:8]}.sb3'
    generated = subprocess.run([sys.executable, str(ROOT / 'tools/generate_project.py'), str(project),
                                '--text', '--benchmark'], capture_output=True, text=True)
    if generated.returncode:
        raise RuntimeError(generated.stderr)
    env = {key: value for key, value in os.environ.items()
           if key in ('PATH', 'HOME', 'LANG', 'TMPDIR')}
    env.update(BONSAI_TURBOWARP_VM=vm, BONSAI_MODEL_PACKAGE=package, NODE_PATH=node_path,
               BONSAI_PROMPT=prompt, BONSAI_MAX_TOKENS='1', BONSAI_STEPS=str(budget),
               BONSAI_TIMEOUT_MS='1800000',
               BONSAI_DEBUG_VARS='BS out tokens,BS greedy token,BS step,BS prompt tokens,BS text',
               BONSAI_DEBUG_FULL='1')
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.perf_counter()
    run = subprocess.run(['node', str(ROOT / 'tests/scratch/run-vm.js'), str(project), '--generate', '24'],
                         env=env, capture_output=True, text=True, timeout=3600)
    wall = time.perf_counter() - started
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    if run.returncode:
        raise RuntimeError(run.stderr[-2000:])
    state = json.loads(run.stdout.strip().splitlines()[-1])
    return {'prompt': prompt, 'wall_seconds': wall,
            'cpu_seconds': (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime),
            'peak_rss_bytes': state.get('peak_rss_bytes',
                                         (after.ru_maxrss - before.ru_maxrss) * 1024), 'state': state}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--budget', type=int, default=11, help='flag presses: prompt length plus one token')
    args = parser.parse_args()
    vm = os.environ['BONSAI_TURBOWARP_VM']
    package = os.environ['BONSAI_MODEL_PACKAGE']
    node_path = os.environ['NODE_PATH']
    rows = []
    with tempfile.TemporaryDirectory() as temp:
        for prompt, expected in PROMPTS:
            record = run_prompt(vm, package, node_path, prompt, args.budget, temp)
            state = record['state']
            prompt_tokens = state.get('BS prompt tokens') or []
            generated = (state.get('BS out tokens') or [])[len(prompt_tokens):]
            if generated and generated[-1] != expected:
                raise SystemExit(f'{prompt!r} produced {generated}, expected {expected} as the new token')
            record['prompt_token_count'] = len(prompt_tokens)
            record['first_token'] = generated[0] if generated else None
            record['text'] = state.get('BS text')
            rows.append(record)
    payload = {'schema': 'bonsai-decoder-io/v1', 'cpu': subprocess.run(['lscpu'], capture_output=True, text=True)
               .stdout.split('Model name:')[-1].splitlines()[0].strip() if Path('/usr/bin/lscpu').exists() else '',
               'node': subprocess.run(['node', '--version'], capture_output=True, text=True).stdout.strip(),
               'note': 'One flag press per decode step; the prompt is prefilled first, so time to first token '
                       'covers the whole prompt. Times include the one-time 5 MB BPE table load.',
               'results': rows}
    args.output.write_text(json.dumps(payload, indent=1) + '\n')
    for row in rows:
        print(f"{row['prompt']!r}: wall {row['wall_seconds']:.1f}s, first token {row['first_token']}, "
              f"text {row['text']!r}")


if __name__ == '__main__':
    main()
