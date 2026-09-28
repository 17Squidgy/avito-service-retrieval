"""Regenerate the selected submission from data and frozen weights, then verify bytes.

No cached predictions are read. Setup downloads are separate; this command makes
no external inference calls. Run with the locked dependencies from the repo root.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'answer.csv')
    args = parser.parse_args()
    output = args.output.resolve()
    manifest = json.loads((ROOT / 'model/manifest.json').read_text())
    for name, expected_input in manifest['input_sha256'].items():
        with (ROOT / 'data/raw' / name).open('rb') as stream:
            actual_input = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual_input != expected_input:
            raise SystemExit(f'Input checksum mismatch: {name}; use the original organizer data.')
    subprocess.run([sys.executable, '-m', 'avito_retrieval.catboost_predict',
                    'model/manifest.json', '--output', str(output)], cwd=ROOT, check=True)
    expected = manifest['answer_sha256']
    actual = hashlib.sha256(output.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f'Submission is structurally valid but differs from the frozen answer.\n'
                         f'Expected SHA-256: {expected}\nActual SHA-256:   {actual}\n'
                         'Check dependency versions, input files, and neural device/cache provenance. '
                         'Do not claim byte-identical reproduction for this run.')
    print(f'Exact submission reproduced: {output}\nSHA-256: {actual}')


if __name__ == '__main__':
    main()
