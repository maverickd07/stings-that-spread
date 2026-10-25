"""Offline checks for copied files, result syntax and exact prompt integrity."""
import ast
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    manifest = json.loads((ROOT / 'COPY_MANIFEST.json').read_text())['files']
    for row in manifest:
        p = ROOT / row['path']
        assert p.stat().st_size == row['bytes'], p
        assert sha(p) == row['sha256'], p
    for p in ROOT.rglob('*'):
        if not p.is_file() or any(x in {'.git', '.venv', '__pycache__', '.pytest_cache', 'logs'} for x in p.relative_to(ROOT).parts):
            continue
        assert not p.is_symlink(), p
        assert p.stat().st_size < 50 * 1024 * 1024, p
        if p.suffix == '.json':
            json.loads(p.read_text())
        elif p.suffix == '.csv':
            with p.open(newline='') as f:
                rows = csv.reader(f)
                header = next(rows)
                assert header, p
                for i, row in enumerate(rows, 2):
                    assert len(row) == len(header), (p, i)
        elif p.suffix == '.py':
            ast.parse(p.read_text())
    examples = json.loads((ROOT / 'prompts/recorded_examples_index.json').read_text())
    examples += json.loads((ROOT / 'prompts/reconstructed_v2_index.json').read_text())
    for e in examples:
        assert sha(ROOT / e['path']) == e['prompt_sha256'], e['path']
    for e in json.loads((ROOT / 'prompts/source_templates.json').read_text()):
        assert sha(ROOT / e['source']) == e['sha256'], e['source']
    sums = ROOT / 'PUBLICATION_SHA256SUMS'
    if sums.exists():
        for line in sums.read_text().splitlines():
            digest, rel = line.split('  ', 1)
            assert sha(ROOT / rel) == digest, rel
    print(f'PASS: {len(manifest)} original copies, {len(examples)} exact prompt examples, result syntax and checksums.')


if __name__ == '__main__':
    main()
