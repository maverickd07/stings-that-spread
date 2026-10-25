"""Offline reanalysis of previously collected contagion and values runs."""
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .judge import key
from .recheck import ROOT, proportion, show, write_json, write_csv
from .wording import sha


def paired(left, right):
    units = sorted(left.keys() & right.keys())
    b = sum(left[u] and not right[u] for u in units)
    c = sum(not left[u] and right[u] for u in units)
    n = b+c
    p = min(1., 2*sum(math.comb(n,k) for k in range(min(b,c)+1))/2**n) if n else 1.
    snaps = sorted({u[0] for u in units})
    cells = np.array([[sum(int(left[u])-int(right[u]) for u in units if u[0]==s),
                       sum(u[0]==s for u in units)] for s in snaps])
    draws = np.random.default_rng(0).integers(0,len(snaps),(2000,len(snaps))) if snaps else None
    bs = cells[draws].sum(axis=1) if snaps else np.empty((0,2))
    ci = np.quantile(bs[:,0]/bs[:,1],[.025,.975]).tolist() if snaps else [None,None]
    return dict(n=len(units),n_snapshots=len(snaps),left_only=b,right_only=c,
        mcnemar_p=p if units else None,left=proportion(sum(left[u] for u in units),len(units)),
        right=proportion(sum(right[u] for u in units),len(units)),difference_ci=ci,
        left_unmatched=len(left.keys()-right.keys()),right_unmatched=len(right.keys()-left.keys()))


def first_offers(root, cond):
    result={}
    for f in sorted((root/'branches').glob(f'*__{cond}__*.json')):
        b=json.loads(f.read_text())
        for o in b['offers']:
            if o['attempt']==1:
                ident=(b['snap'],o['agent'])
                assert ident not in result
                result[ident]=bool(o['accept'])
    return result


def main():
    dest=ROOT/'runs/extensions_recheck';dest.mkdir(exist_ok=True)
    models=('Qwen2.5-32B-Instruct','Mistral-Small-24B-Instruct-2501')
    hashes={};cont=[];values=[]
    for model in models:
        main=ROOT/'runs'/('main_'+model);val=ROOT/'runs'/('values_'+model)
        for root in (main,val):
            for p in root.rglob('*'):
                if p.is_file() and 'recheck' not in p.relative_to(root).parts:
                    hashes[str(p.resolve())]=sha(p)
        cache=json.loads((main/'judgments.json').read_text());groups=defaultdict(dict);accepted=defaultdict(dict)
        host={}
        for f in sorted((main/'contagion').glob('*.json')):
            b=json.loads(f.read_text());cond=b['condition']
            for r in b['contagion']:
                ident=(b['snap'],r['agent'])
                assert ident not in groups[cond]
                if r['sent']:
                    j=cache[key('collusion',r['to'],r['message'])]
                    assert j['judge_model'].endswith(model) and not j['mock']
                    recruit=bool(j['judge']['collusion'])
                else:recruit=False
                groups[cond][ident]=recruit
                accepted[cond][ident]=r['accepted']
                h=(r['host_snap'],r['host_name'])
                assert ident not in host or host[ident]==h
                host[ident]=h
        base=groups['N0'];assert len(base)==84
        for cond in ('N0','L0','L1','L2','L3','PC'):
            g=groups[cond];assert g.keys()==base.keys()
            sub={u:v for u,v in g.items() if accepted[cond][u]}
            cont.append(dict(model=model,condition=cond,all=proportion(sum(g.values()),len(g)),
                paired_vs_N0=paired(g,base),accepted=proportion(sum(sub.values()),len(sub)),
                accepted_paired_vs_N0=paired(sub,base)))
        for cond in ('L1','L2','L3'):
            l,r=first_offers(val,cond),first_offers(main,cond)
            for u in l.keys()&r.keys():
                a=json.loads((val/'snapshots'/(u[0]+'.json')).read_text())
                b=json.loads((main/'snapshots'/(u[0]+'.json')).read_text())
                assert a['seed']==b['seed'] and a['uc']==b['uc']
            values.append(dict(model=model,condition=cond,
                values_all=proportion(sum(l.values()),len(l)),rules_all=proportion(sum(r.values()),len(r)),
                paired=paired(l,r)))
    assert all(sha(Path(f))==h for f,h in hashes.items())
    write_json(dest/'results.json',dict(contagion=cont,values=values,source_sha256=hashes,
        bootstrap_seed=0,bootstrap_B=2000,original_inputs_unchanged=True,inference_calls=0))
    def pv(p):return '—' if p is None else f'{p:.3g}'
    def name(m):return 'Qwen-32B' if m.startswith('Qwen') else 'Mistral-24B'
    lines=['# 已有扩展实验：离线复核','',
        'Contagion：一次下一局 PM 被原自评裁判判为合谋；每条件 28 快照×3 原代理=84，包括已出局代理。括号为 k/n，区间为 Wilson 95% CI。接受者=原游戏累计接受标记（PC 为强制联盟），不是后来 Phi-4 重判。', '',
        '|模型|条件|全部招募率 % [CI]|配对 N0 p|接受者招募率 % [CI]|接受者对应 N0 % [CI]|接受者配对 p|',
        '|---|---|---|---|---|---|---|']
    for r in cont:
        lines.append(f"|{name(r['model'])}|{r['condition']}|{show(r['all'])}|{pv(r['paired_vs_N0']['mcnemar_p']) if r['condition']!='N0' else '—'}|{show(r['accepted'])}|{show(r['accepted_paired_vs_N0']['right'])}|{pv(r['accepted_paired_vs_N0']['mcnemar_p'])}|")
    lines+=['','Values：首次提议接受，沿用两模式原始 parse_accept 标签，未调用新裁判。全体率见下；配对检验只用共同的种子×代理。', '',
        '|模型|条件|values 全体 % [CI]|rules 全体 % [CI]|配对 values/rules 接受数；共同 n|配对 p|',
        '|---|---|---|---|---|---|']
    for r in values:
        p=r['paired'];lines.append(f"|{name(r['model'])}|{r['condition']}|{show(r['values_all'])}|{show(r['rules_all'])}|{p['left']['k']}/{p['right']['k']}; n={p['n']}|{pv(p['mcnemar_p'])}|")
    lines+=['',
        '检验均为双侧精确 McNemar。Contagion 按源快照×原代理配对相同下一局 host 的 N0；接受者子组的 N0 也限制为同一组代理。接受者筛选是处理后选择，不能解释为因果效应。',
        'Values 是独立生成的对局：两模型各有 78 个共同代理，Qwen 排除 values/rules 各 1 个，Mistral 排除 4/1 个；同种子不代表同快照，差异可能包含之前游戏历史的影响。Values 仅测首次回复，无后续行为。',
        'Wilson/McNemar 未校正快照聚类或多重比较；results.json 另存 2000 次快照 bootstrap 差值 CI（seed=0）和全部配对计数。裁判均为原代理同模型（Modal，T=0），未将这些标签混称为 Phi-4 中立判定。原始数据未改，新增费用 $0。']
    (dest/'summary.md').write_text('\n'.join(lines)+'\n')
    print((dest/'summary.md').read_text())


if __name__=='__main__':main()
