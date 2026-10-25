"""Reconstruct V2 prompt examples from trusted local snapshots and saved turns.

No inference. Read only the original runs and write prompt-only publication files.
Only load the author's trusted snapshots; pickle is not a safe interchange format.
"""
import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sting import values as V
from sting.interp import SOURCE


def main(raw):
    index = []
    original_chat = V.chat
    for folder in sorted((raw / 'values').iterdir()):
        if not (folder / 'v2').is_dir():
            continue
        seen = set()
        paths = sorted((folder / 'v2').glob('snap_*.json'))
        # Prefer a saved trajectory reaching the furthest re-offer, so each
        # observed turn for the condition has an exact example.
        for p in sorted(paths, key=lambda p: -len(json.loads(p.read_text())['turns'])):
            d = json.loads(p.read_text())
            condition = d['condition']
            if condition in seen:
                continue
            seen.add(condition)
            state = raw / ('main_' + SOURCE[d['model']].split('/')[-1]) / 'snapshots' / (d['snap'] + '.pkl')
            with state.open('rb') as f:
                game = pickle.load(f)
            if d['model'] == V.LLAMA:
                for seat in game.agents():
                    seat.model = V.LLAMA
            calls = []
            turns = iter(d['turns'])
            def replay(raw_client, model, prompt, max_tokens, temperature):
                t = next(turns)
                calls.append((t['turn'], prompt))
                return t['reply'], t.get('error')
            V.chat = replay
            try:
                rebuilt = V.v2_run(None, d['model'], game, game.seat(d['agent']), condition)
            finally:
                V.chat = original_chat
            assert rebuilt == d['turns'], p
            for turn, prompt in calls:
                rel = Path('prompts/reconstructed_v2') / folder.name / f'{condition}__turn{turn}.txt'
                target = ROOT / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(prompt.encode())
                index.append(dict(path=str(rel), condition=condition, turn=turn, model=d['model'],
                                  source=str(p.relative_to(raw)),
                                  source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                                  snapshot=str(state.relative_to(raw)),
                                  snapshot_sha256=hashlib.sha256(state.read_bytes()).hexdigest(),
                                  prompt_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                                  reconstruction='Exact source builder replayed with saved responses; no inference'))
    (ROOT / 'prompts/reconstructed_v2_index.json').write_text(json.dumps(index, ensure_ascii=False, indent=2) + '\n')
    print(f'PASS: {len(index)} V2 prompt examples reconstructed; saved turns match exactly.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw-runs', type=Path, required=True)
    args = ap.parse_args()
    main(args.raw_runs.resolve())
