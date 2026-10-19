"""Verify the recovered package without network access, GPU inference, or live submissions."""
from __future__ import annotations

import ast
import base64
import gzip
import io
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    environment = dict(os.environ)
    environment['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
    environment['BIOHUB_ALLOW_LIVE_SUBMISSIONS'] = '0'
    environment['PYTHONPATH'] = os.pathsep.join((str(ROOT / 'src'), str(ROOT)))
    checked = 0
    for folder in ('src', 'tests', 'scripts', 'external'):
        for file in (ROOT / folder).rglob('*.py'):
            ast.parse(file.read_text(encoding='utf-8'), filename=str(file.relative_to(ROOT)))
            checked += 1
    archived_notebooks = []
    for file in sorted((ROOT / 'notebooks').glob('*.ipynb')):
        saved = json.loads(file.read_text())
        for cell in saved['cells']:
            if cell['cell_type'] == 'code':
                if cell.get('outputs') or cell.get('execution_count') is not None:
                    raise RuntimeError('Archived notebook contains execution state: ' + file.name)
                compile(''.join(cell['source']), file.name, 'exec')
        archived_notebooks.append(file.name)
    with tempfile.TemporaryDirectory(prefix='biohub-export-verify-') as temporary:
        temporary = Path(temporary)
        junit = temporary / 'tests.xml'
        tests = subprocess.run(
            [sys.executable, '-m', 'pytest', '-q', '--tb=short', '--junitxml=' + str(junit)],
            cwd=ROOT, env=environment, capture_output=True, text=True, check=False,
        )
        print(tests.stdout)
        if tests.returncode:
            print(tests.stderr, file=sys.stderr)
            return tests.returncode
        tree = ET.parse(junit)
        cases = tree.findall('.//testcase')
        skips = [case.get('name') for case in cases if case.find('skipped') is not None]
        notebook = temporary / 'best-own.ipynb'
        subprocess.run(
            [sys.executable, str(ROOT / 'scripts/build_best_own_notebook.py'), '--output', str(notebook)],
            cwd=temporary, env=environment, check=True, capture_output=True, text=True,
        )
        notebook_data = json.loads(notebook.read_text())
        for cell in notebook_data['cells']:
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), '<generated-best-own-notebook>', 'exec')
        smoke = subprocess.run(
            [sys.executable, '-m', 'biohub_lab.cli', 'run', '--config', str(ROOT / 'config/experiments.json'),
             '--duration-seconds', '2', '--max-experiments', '2', '--output-dir', 'runs/demo'],
            cwd=temporary, env=environment, capture_output=True, text=True, check=False,
        )
        if smoke.returncode:
            raise RuntimeError('Synthetic CLI run failed: ' + smoke.stderr[-2000:])
        records = [json.loads(file.read_text()) for file in (temporary / 'runs/demo').glob('*/result.json')]
        if len(records) != 2 or any(row.get('status') != 'completed' for row in records):
            raise RuntimeError('Expected two completed smoke records; got ' + repr([row.get('status') for row in records]))
        subprocess.run(
            [sys.executable, '-m', 'biohub_lab.cli', 'rank', '--runs-dir', 'runs/demo', '--output-dir', 'outputs/demo'],
            cwd=temporary, env=environment, check=True, capture_output=True, text=True,
        )
        ranking = temporary / 'outputs/demo/ranking.csv'
        if not ranking.is_file():
            raise RuntimeError('Smoke ranking was not generated.')
    sys.path.insert(0, str(ROOT / 'src'))
    from biohub_lab.validation import validate_submission
    from biohub_lab.kaggle_runner import _package_upstream_archive
    with zipfile.ZipFile(io.BytesIO(gzip.decompress(base64.b64decode(_package_upstream_archive(ROOT))))) as archive:
        if b'BSD 3-Clause License' not in archive.read('LICENSE'):
            raise RuntimeError('Upstream BSD license is missing from the generated archive.')
    fixture = validate_submission(ROOT / 'tests/fixtures/submission.csv', include_topology=True)
    if not fixture.valid:
        raise RuntimeError('Synthetic CSV fixture failed validation.')
    report = {
        'checked_at': datetime.now(timezone.utc).isoformat(),
        'python': platform.python_version(),
        'platform': platform.system() + '-' + platform.machine(),
        'verification_packages': {name: importlib.metadata.version(name) for name in ('pytest', 'numpy', 'scipy')},
        'python_files_parsed': checked,
        'archived_notebooks_syntax': archived_notebooks,
        'tests': {'total': len(cases), 'passed': len(cases) - len(skips), 'failed': 0, 'skipped': len(skips), 'skipped_tests': skips},
        'generated_notebook_syntax': 'passed',
        'upstream_license_packaging': 'passed',
        'synthetic_cli_runs_completed': len(records),
        'synthetic_ranking': 'passed',
        'synthetic_csv_topology_validation': 'passed',
        'scope': 'Software and synthetic verification only. No historical dataset/GPU rerun, new Kaggle score, or live submission.',
    }
    destination = ROOT / 'results/validation.json'
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
