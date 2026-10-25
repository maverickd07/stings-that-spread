"""Offline, seed-0 free-form reply audit. Never invokes an inference client."""
import hashlib
import json
import random
import re
from pathlib import Path

from .wording import ROOT, request


def main():
    base = (ROOT / 'runs/wording').resolve()
    models = ('Qwen2.5-32B-Instruct', 'Mistral-Small-24B-Instruct-2501')
    patterns = {word: re.compile(pattern, re.I) for word, pattern in {
        'deal': r'\bdeal\b', 'agreed': r'\bagreed\b', 'yes': r'\byes\b',
        'sure': r'\bsure\b', "won't challenge": r"\bwon['’]t\s+challenge\b",
    }.items()}
    hashes, groups, counts, all_l2f_agree = {}, [], [], []

    def read(path):
        raw = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    for model in models:
        cache = read(base/model/'judgments_phi4.json')
        for condition in ('L1F', 'L2F'):
            rows = []
            for path in sorted((base/model/'offers').glob(f'*__{condition}.json')):
                r = read(path)
                jid = request('agree', {'offer': r['offer'], 'reply': r['reply']})['id']
                saved = cache[jid]
                assert saved['judge_model'] == 'microsoft/phi-4'
                agree = saved['judge']['agree']
                assert type(agree) is bool
                rows.append(dict(id=r['id'], model=model, condition=condition,
                    snap=r['snap'], agent=r['agent'], uc=r['uc'], reply=r['reply'],
                    offer=r['offer'], phi4_agree=agree, judge_id=jid,
                    commitment_word_hits=[w for w,p in patterns.items() if p.search(r['reply'])],
                    source=str(path)))
            assert len(rows) == 79
            # Fresh fixed generator in each model × condition stratum; no label filtering.
            sample = random.Random(0).sample(rows, 30)
            groups.append(dict(model=model, condition=condition, n=79, sampled=sample))
            if condition == 'L2F':
                positives = [r for r in rows if r['phi4_agree']]
                all_l2f_agree.extend(positives)
                sampled_positive = [r for r in sample if r['phi4_agree']]
                counts.append(dict(model=model, agree_n=len(positives),
                    any_word_n=sum(bool(r['commitment_word_hits']) for r in positives),
                    per_word={w:sum(w in r['commitment_word_hits'] for r in positives) for w in patterns},
                    sampled_agree_n=len(sampled_positive),
                    sampled_any_word_n=sum(bool(r['commitment_word_hits']) for r in sampled_positive)))
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    output = base/'free_check'
    output.mkdir(exist_ok=True)
    audit = dict(seed=0, sampling='random.Random(0).sample(sorted source filenames, 30), independently reset per model × condition; no stratification by label',
        patterns={w:p.pattern for w,p in patterns.items()}, counts=counts, groups=groups,
        all_l2f_agree=all_l2f_agree, source_sha256=hashes, inputs_unchanged=True,
        inference_calls=0, additional_compute_cost_usd=0)
    (output/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    lines=['# 自由回答人工核查：120 条全文', '',
        '离线读取已有 Phi-4 标签，不重判。每个模型 × L1F/L2F 从 79 条中不放回随机抽 30 条；每组按源文件名排序后，独立使用 random.Random(0)。未按接受标签筛选。', '',
        '词汇计数使用全部 L2F agree=true 回复；忽略大小写，整词匹配 deal / agreed / yes / sure，以及连续短语 won’t challenge（接受直/弯撇号、可变空白）。同一回复可命中多词，任一命中只计一次。不扩展到 deals、agree、I’m in 等未列词。词汇命中不判断否定或引用，也不等同于真实承诺。', '',
        '|模型|L2F agree=true|任一词命中|deal|agreed|yes|sure|won’t challenge|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for c in counts:
        lines.append('|'+c['model']+'|'+str(c['agree_n'])+'|'+str(c['any_word_n'])+'|'+'|'.join(str(c['per_word'][w]) for w in patterns)+'|')
    for group in groups:
        lines += ['',f"## {group['model']} — {group['condition']}（30/79）", '']
        for i,r in enumerate(group['sampled'],1):
            hits=', '.join(r['commitment_word_hits']) or '无'
            lines += [f"### {i:02d}. {r['id']} — Phi-4 agree={str(r['phi4_agree']).lower()}", '',
                      f"指定词命中：{hits}；代理 {r['agent']}，卧底 {r['uc']}。", '', '````text',r['reply'],'````','']
    lines += ['', '## 补充：Mistral 全部两条 L2F 接受回复（不属于随机样本）', '',
              '固定 seed=0 的 Mistral L2F 样本包含 0 条 agree=true。以下补列全部两条接受回复供核查，不替换或改变上述 30 条随机样本。', '']
    for r in all_l2f_agree:
        if r['model'].startswith('Mistral'):
            lines += [f"### {r['id']} — Phi-4 agree=true", '', '````text',r['reply'],'````','']
    (output/'full_replies.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(output=str(output),counts=counts,samples=sum(len(g['sampled']) for g in groups),inputs_unchanged=True),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
