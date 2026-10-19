"""Generate the recovered best-own Kaggle notebook without uploading or executing it."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from biohub_lab.kaggle_runner import build_support_pack_runner_source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/best-own-h9-v11.ipynb'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / 'config/best-own-h9-v11.json').read_text())
    source = build_support_pack_runner_source(root, **config['runner_options'])
    compile(source, '<best-own-notebook>', 'exec')
    inputs = '\n'.join('- ' + item for item in config['required_kaggle_inputs'])
    notebook = {
        'nbformat': 4,
        'nbformat_minor': 5,
        'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python'}},
        'cells': [
            {'id': 'provenance', 'cell_type': 'markdown', 'metadata': {}, 'source': [
                '# Biohub H9 v11 — historical best-own configuration\n\n'
                'Historical public score: 0.898, reference 55950639. This is a recovered configuration, not a fresh score reproduction.\n\n'
                'Attach the competition input and these inputs through your own Kaggle account, subject to their licenses:\n' + inputs
            ]},
            {'id': 'inference', 'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': source.splitlines(keepends=True)},
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(notebook, indent=2) + '\n')
    print(args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
