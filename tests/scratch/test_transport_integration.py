import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import zipfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


def add_loader_block(project, count, scale_list, start=0):
    with zipfile.ZipFile(project) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    graph = json.loads(entries['project.json'])
    blocks = graph['targets'][0]['blocks']
    hat = next(key for key, value in blocks.items() if value['opcode'] == 'event_whenflagclicked')
    following = blocks[hat]['next']
    blocks[hat]['next'] = 'load-resident'
    blocks[following]['parent'] = 'load-resident'
    fields = {'ID': 'resident', 'TENSOR': 'blk.3.attn_q.weight', 'START': start, 'COUNT': count,
              'Q': 'l_packed', 'SCALE': scale_list}
    blocks['load-resident'] = {'opcode': 'bonsailoader_loadGroups', 'next': following, 'parent': hat,
                               'inputs': {key: [1, [4 if isinstance(value, int) else 10, str(value)]]
                                          for key, value in fields.items()},
                               'fields': {}, 'shadow': False, 'topLevel': False}
    graph['extensions'] = ['bonsailoader']
    entries['project.json'] = json.dumps(graph, separators=(',', ':')).encode()
    with zipfile.ZipFile(project, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, contents in entries.items():
            archive.writestr(name, contents)


class TransportIntegrationTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('BONSAI_TURBOWARP_VM') and os.environ.get('BONSAI_MODEL_PACKAGE'),
                         'set BONSAI_TURBOWARP_VM and BONSAI_MODEL_PACKAGE')
    def test_compiled_resident_scale_decode_after_loader_disabled(self):
        raw = (ROOT / 'fixtures/native/layer3-q-first5.pq2').read_bytes()
        expected = [struct.unpack('<e', raw[i * 34:i * 34 + 2])[0] for i in range(5)]
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp) / 'project.sb3'
            generated = subprocess.run([sys.executable, str(ROOT / 'tools/generate_project.py'), str(project)], capture_output=True, text=True)
            self.assertEqual(generated.returncode, 0, generated.stderr)
            add_loader_block(project, 5, 'l_input')
            run = subprocess.run(['node', str(ROOT / 'tests/scratch/run-vm.js'), str(project), '--package'], capture_output=True, text=True, timeout=60)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(json.loads(run.stdout), expected)

    @unittest.skipUnless(os.environ.get('BONSAI_TURBOWARP_VM') and os.environ.get('BONSAI_MODEL_PACKAGE'),
                         'set BONSAI_TURBOWARP_VM and BONSAI_MODEL_PACKAGE')
    def test_compiled_resident_scalar_dot_after_loader_disabled(self):
        expected = struct.unpack('<5f', (ROOT / 'fixtures/native/layer3-q-first5.f32').read_bytes())
        with tempfile.TemporaryDirectory() as temp:
            for row in range(5):
                with self.subTest(row=row):
                    project = Path(temp) / f'project-{row}.sb3'
                    generated = subprocess.run([sys.executable, str(ROOT / 'tools/generate_project.py'), str(project)], capture_output=True, text=True)
                    self.assertEqual(generated.returncode, 0, generated.stderr)
                    add_loader_block(project, 16, 'l_scales', row * 16)
                    run = subprocess.run(['node', str(ROOT / 'tests/scratch/run-vm.js'), str(project), '--dot'],
                                         capture_output=True, text=True, timeout=60)
                    self.assertEqual(run.returncode, 0, run.stderr)
                    actual = json.loads(run.stdout)
                    self.assertEqual(len(actual), 1)
                    self.assertLess(abs(actual[0] - expected[row]), 1e-6)


if __name__ == '__main__':
    unittest.main()
