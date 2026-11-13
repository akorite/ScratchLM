import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(os.environ.get('BONSAI_TURBOWARP_VM'), 'set BONSAI_TURBOWARP_VM to the pinned VM checkout')
class TextBlockTests(unittest.TestCase):
    def test_compiled_case_unicode_and_unchanged_backdrop(self):
        equality = [('', '', 1), ('abc', 'abc', 1), ('z', 'Z', 0), ('AbC', 'abc', 0),
                    ('é', 'e\u0301', 0), ('漢字', '漢字', 1), ('😀', '😁', 0),
                    ('\ud800', '\ud801', 0), ('\ud800', '\ud800', 1), ('x', 'xy', 0)]
        utf8 = [('', []), ('aAzZ', [97, 65, 122, 90]), ('e\u0301', [101, 204, 129]),
                ('漢', [230, 188, 162]), ('😀', [240, 159, 152, 128]),
                ('\ud800\udc00', [240, 144, 128, 128]), ('\udbff\udfff', [244, 143, 191, 191]),
                ('\ud800x\udc00', [239, 191, 189, 120, 239, 191, 189]),
                ('\ud800\ud800\udc00', [239, 191, 189, 240, 144, 128, 128]),
                ('\u007f\u0080\u07ff\u0800', [127, 194, 128, 223, 191, 224, 160, 128])]
        cases = [{'mode': 20, 'input': [left, right]} for left, right, _ in equality]
        cases += [{'mode': 21, 'input': [value]} for value, _ in utf8]
        expected = [[value] for _, _, value in equality] + [value for _, value in utf8]
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp) / 'text.sb3'
            generated = subprocess.run([sys.executable, str(ROOT / 'tools/generate_project.py'), str(project), '--text'],
                                       capture_output=True, text=True)
            self.assertEqual(generated.returncode, 0, generated.stderr)
            with zipfile.ZipFile(project) as source:
                entries = {name: source.read(name) for name in source.namelist()}
            data = json.loads(entries['project.json'])
            stage = data['targets'][0]
            stage['costumes'].append(dict(stage['costumes'][0], name='other backdrop'))
            stage['currentCostume'] = 1
            entries['project.json'] = json.dumps(data, ensure_ascii=True).encode()
            with zipfile.ZipFile(project, 'w', zipfile.ZIP_DEFLATED) as archive:
                for name, content in entries.items():
                    archive.writestr(name, content)
            case_file = Path(temp) / 'cases.json'
            case_file.write_text(json.dumps(cases, ensure_ascii=True))
            result = subprocess.run(['node', str(ROOT / 'tests/scratch/run-text-batch.js'), str(project), str(case_file)],
                                    capture_output=True, text=True, timeout=240)
            self.assertEqual(result.returncode, 0, result.stderr[-4000:])
            actual = json.loads(result.stdout)
            self.assertEqual(len(actual), len(expected))
            for case, got, want in zip(cases, actual, expected):
                with self.subTest(case=ascii(case)):
                    self.assertEqual(got, {'output': want, 'backdrop': 2})


if __name__ == '__main__':
    unittest.main()
