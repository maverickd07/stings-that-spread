"""Export exact prompt constants from source, without importing or running experiments.

With --raw-runs, also copy one representative stored prompt per directory,
condition and offer rung. Responses and inference logs are never copied.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def constants():
    files = sorted((ROOT / 'sting').glob('*.py'))
    files += sorted((ROOT / 'runs').glob('**/source/sting/*.py'))
    result = []
    for path in files:
        tree = ast.parse(path.read_text(encoding='utf-8'))
        entries = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                names = [t.id for t in node.targets if isinstance(t, ast.Name) and t.id.isupper()]
                if not names:
                    continue
                try:
                    val = ast.literal_eval(node.value)
                    json.dumps(val)
                except (ValueError, TypeError, SyntaxError):
                    continue
                for name in names:
                    entries[name] = val
        # Function-local prompt fragments, including f-string source, remain
        # recoverable without executing their builders or loading pickles.
        fragments = [dict(line=n.lineno, source=ast.get_source_segment(path.read_text(encoding='utf-8'), n))
                     for n in ast.walk(tree)
                     if isinstance(n, ast.JoinedStr) or (isinstance(n, ast.Constant)
                         and isinstance(n.value, str) and len(n.value) >= 100)]
        if entries or fragments:
            result.append(dict(source=str(path.relative_to(ROOT)),
                               sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                               constants=entries, long_strings_and_fstrings=fragments))
    dump(ROOT / 'prompts/source_templates.json', result)


def examples(raw):
    groups = set()
    exported = []
    paths = []
    for folder in ('wording', 'additional_Llama31_8B', 'claude_ext', 'robust', 'interp', 'values'):
        paths += sorted((raw / folder).rglob('snap_*.json'))
    for p in paths:
        if any(part in {'logs', 'snapshots', 'branches', 'contagion'} for part in p.relative_to(raw).parts):
            continue
        d = json.loads(p.read_text(encoding='utf-8'))
        if not isinstance(d, dict) or not isinstance(d.get('prompt'), str):
            continue
        cond = d.get('condition') or d.get('memory') or 'N0'
        rung = d.get('offer_rung', '')
        group = (str(p.parent.relative_to(raw)), cond, rung)
        if group in groups:
            continue
        groups.add(group)
        dest = Path('prompts/recorded_examples') / p.relative_to(raw).parent / (cond + ('__' + rung if rung else '') + '.txt')
        target = ROOT / dest
        target.parent.mkdir(parents=True, exist_ok=True)
        # Preserve exact prompt bytes; do not append a newline.
        target.write_bytes(d['prompt'].encode('utf-8'))
        exported.append(dict(path=str(dest), source=str(p.relative_to(raw)),
                             source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                             prompt_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                             condition=cond, offer_rung=rung, model=d.get('model')))
    dump(ROOT / 'prompts/recorded_examples_index.json', exported)
    print(f'Exported {len(exported)} representative recorded prompts; no responses or raw logs copied.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw-runs', type=Path, help='Optional local raw runs tree; read only')
    args = ap.parse_args()
    constants()
    if args.raw_runs:
        examples(args.raw_runs.resolve())
