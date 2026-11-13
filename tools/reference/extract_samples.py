#!/usr/bin/env python3
"""Extract compact native operator samples from the documented capture runs."""
import argparse
import hashlib
import json
from pathlib import Path

REVISION = 'e3de715222b2667c54a267441159a5729f11155e'
MODEL = 'd5f01a4ecaf49239653308ee732ad1a0c5f802c0ddb997549fc7b694c8b45928'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    sources = {}
    def load(folder):
        if folder not in sources:
            manifest = json.loads((args.capture_root / folder / 'manifest.json').read_text())
            if manifest['model']['sha256'] != MODEL or manifest['build']['revision'] != REVISION or manifest['input_tokens'] != [1, 321]:
                raise ValueError(f'{folder}: capture provenance mismatch')
            sources[folder] = manifest
        return sources[folder]
    def save(folder, entry, name, limit=None):
        data = (args.capture_root / folder / entry['payload']['file']).read_bytes()
        if hashlib.sha256(data).hexdigest() != entry['payload']['sha256']:
            raise ValueError(f'{folder}: capture checksum mismatch')
        if limit is not None:
            if len(data) < limit:
                raise ValueError(f'{folder}: capture is too short')
            data = data[:limit]
        (output / name).write_bytes(data)
        files[name] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                       'source_run': folder, 'step': entry['step']}
    for kind, folder, node, operation, size in [('rms', 'norm0', 'norm-0', 'RMS_NORM', None),
                                                 ('silu', 'silu0-pre', 'conv_output_silu-0', 'UNARY', None),
                                                 ('l2', 'l2-0', 'qk_conv_l2-0', 'L2_NORM', 512)]:
        captures = load(folder)['captures']
        for step in (0, 1):
            for role, label in [('src0', 'input'), ('out', 'output')]:
                source_node, source_op, source_role = (('conv_output_raw-0', 'SSM_CONV', 'out')
                                                      if kind == 'silu' and role == 'src0'
                                                      else (node, operation, role))
                matches = [c for c in captures if c['step'] == step and c['node'] == source_node and
                           c['operation'] == source_op and c['role'] == source_role]
                if not matches:
                    raise ValueError(f'{folder}: missing {step} {role}')
                save(folder, matches[0], f'{kind}-{step}-{label}.f32', size)
    for kind, folder, node in [('q3', 'q3-heldout', 'Qcur_full-3'), ('ffn0', 'ffn0-heldout', 'ffn_out-0')]:
        manifest = load(folder)
        for role, label in [('src1', 'input'), ('out', 'output')]:
            matches = [c for c in manifest['captures'] if c['step'] == 1 and c['node'] == node and c['role'] == role]
            if len(matches) != 1:
                raise ValueError(f'{folder}: ambiguous held-out {role}')
            save(folder, matches[0], f'{kind}-1-{label}.f32')
        matches = [c for c in manifest['cpu_q8_k_staging'] if c['step'] == 1 and c['node'] == node]
        if len(matches) != 1:
            raise ValueError(f'{folder}: ambiguous held-out staging')
        save(folder, matches[0], f'{kind}-1-staging.q8_k')
    (output / 'manifest.json').write_text(json.dumps({'schema': 'bonsai-native-operator-samples/v1',
        'model_sha256': MODEL, 'runtime_revision': REVISION, 'tokens': [1, 321],
        'notes': 'Step 0 calibration and stateful step 1 holdout; L2 samples are the first 128-element row.',
        'files': files}, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
