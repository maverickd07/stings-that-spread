"""E3: does Llama-3.1-8B's generic refusal direction (Arditi et al. 2024) drive its refusals of the sting offers?

  python -m modal run sting/refusal_dir.py      (one H100 container, HF transformers, then it exits)

Local side builds the game prompts from the saved Qwen snapshots (agents answered by Llama, as in E1) and reads
the harmful/harmless instruction splits from runs/interp/e3_data; results go to runs/interp/Llama8B_refusal/.
"""
import json
import os
import random
from pathlib import Path

import modal

MODEL = 'NousResearch/Meta-Llama-3.1-8B-Instruct'
app = modal.App('sting-refusal')
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('torch==2.6.0', 'transformers==4.55.2', 'accelerate==1.10.0', 'numpy<2.3')
         .env({'HF_HOME': '/cache/hf', 'HF_HUB_OFFLINE': '0'}))
cache = modal.Volume.from_name('sting-hf-cache', create_if_missing=True)
E3_CONDS = ['L0', 'L1', 'L2', 'L2-nobody', 'L2-secret', 'L2-victim', 'L2-allcues', 'L1+nobody', 'L1+secret',
            'D0L1', 'DmonL1', 'DpenL1', 'DplaceboL1', 'VL1', 'VplaceboL1']
ABLATE_CONDS = ['L1', 'L2', 'L2-allcues', 'D0L1', 'VL1']


@app.function(image=image, gpu='H100', volumes={'/cache': cache}, timeout=40 * 60)
def run(payload: dict) -> dict:
    import re
    import time
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = 'left'
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map='cuda')
    model.eval()
    layers = model.model.layers
    nL = len(layers)

    def norm(s):
        return re.sub(r'[\s*"\'`_#:\-\.]', '', s).upper()

    acc_ids, ref_ids = [], []
    for i in range(len(tok)):
        t = norm(tok.decode([i]))
        if len(t) >= 2 and ('ACCEPT'.startswith(t) or t.startswith('ACCEPT')):
            acc_ids.append(i)
        elif len(t) >= 2 and any(w.startswith(t) or t.startswith(w) for w in ('REFUSE', 'REJECT', 'DECLINE')):
            ref_ids.append(i)
    refusal_first = [tok.encode(w, add_special_tokens=False)[0] for w in ('I', 'Sorry', 'As', 'Unfortunately')]

    hooks, state = [], {'ablate': None, 'add': None, 'add_layer': None}

    def pre_hook(mod, args):
        h = args[0]
        if state['ablate'] is not None:
            r = state['ablate']
            h = h - (h @ r).unsqueeze(-1) * r
        return (h,) + args[1:]

    def out_hook(idx):
        def f(mod, args, out):
            h = out[0] if isinstance(out, tuple) else out
            if state['ablate'] is not None:
                r = state['ablate']
                h = h - (h @ r).unsqueeze(-1) * r
            if state['add'] is not None and idx == state['add_layer']:
                h = h + state['add']
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return f
    for i, l in enumerate(layers):
        hooks.append(l.register_forward_pre_hook(pre_hook) if i == 0 else None)
        hooks.append(l.register_forward_hook(out_hook(i)))

    def enc(texts):
        chats = [tok.apply_chat_template([{'role': 'user', 'content': t}], tokenize=False, add_generation_prompt=True) for t in texts]
        return tok(chats, return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')

    @torch.no_grad()
    def forward(texts, bs=8, want_hidden=False):
        hid, logp = [], []
        for i in range(0, len(texts), bs):
            b = enc(texts[i:i + bs])
            out = model(**b, output_hidden_states=want_hidden)
            lp = torch.log_softmax(out.logits[:, -1].float(), -1)
            logp.append(lp.cpu())
            if want_hidden:
                hid.append(torch.stack([h[:, -1].float() for h in out.hidden_states[1:]], 1).cpu())   # [b, nL, d]
        return (torch.cat(hid) if want_hidden else None), torch.cat(logp)

    def ar(lp):
        pa = lp[:, acc_ids].exp().sum(-1); pr = lp[:, ref_ids].exp().sum(-1)
        return pa, pr, (pa.clamp_min(1e-6).log() - pr.clamp_min(1e-6).log())

    # 1. candidate directions: mean difference at the last prompt token, each layer
    harm, harmless = payload['harmful_train'], payload['harmless_train']
    hh, _ = forward(harm, want_hidden=True)
    hl, _ = forward(harmless, want_hidden=True)
    diff = hh.mean(0) - hl.mean(0)                       # [nL, d]
    dirs = diff / diff.norm(dim=-1, keepdim=True)
    # 2. pick the layer whose direction, ablated everywhere, most reduces refusal on held-out harmful prompts
    _, base_lp = forward(payload['harmful_val'])
    base_ref = base_lp[:, refusal_first].exp().sum(-1).mean().item()
    sel = {}
    for li in range(4, int(nL * 0.75)):
        state['ablate'] = dirs[li].to('cuda', torch.bfloat16)
        _, lp = forward(payload['harmful_val'])
        sel[li] = lp[:, refusal_first].exp().sum(-1).mean().item()
    state['ablate'] = None
    best = min(sel, key=sel.get)
    r = dirs[best]
    # sanity: generations on 8 held-out harmful prompts, with and without ablation
    @torch.no_grad()
    def generate(texts, n=48):
        b = enc(texts)
        out = model.generate(**b, max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
        return [tok.decode(o[b['input_ids'].shape[1]:], skip_special_tokens=True) for o in out]
    gen_base = generate(payload['harmful_val'][:8])
    state['ablate'] = r.to('cuda', torch.bfloat16)
    gen_abl = generate(payload['harmful_val'][:8])
    state['ablate'] = None
    proj_harm = (hh[:, best] @ r).tolist(); proj_harmless = (hl[:, best] @ r).tolist()

    # 3. projections + first-token log-odds on the game prompts
    items = payload['items']
    H, LP = forward([it['prompt'] for it in items], want_hidden=True)
    pa, pr, lo = ar(LP)
    proj = (H[:, best] @ r).tolist()
    for it, p, a, b_, l in zip(items, proj, pa.tolist(), pr.tolist(), lo.tolist()):
        it.update(proj=p, pA=a, pR=b_, logit=l)
    # 3b. every-layer projections; an 'L2 minus L1' direction from even snapshots, tested on odd ones
    pall = torch.einsum('nld,ld->nl', H, dirs)
    for it, row in zip(items, pall.tolist()):
        it['proj_layers'] = row
    snapnum = lambda it: int(it['snap'].split('_')[1])
    ev2 = [i for i, it in enumerate(items) if it['condition'] == 'L2' and snapnum(it) % 2 == 0]
    ev1 = [i for i, it in enumerate(items) if it['condition'] == 'L1' and snapnum(it) % 2 == 0]
    dl = H[ev2].mean(0) - H[ev1].mean(0)                               # [nL, d]
    cos_l = torch.nn.functional.cosine_similarity(dl, dirs, dim=-1).tolist()
    sdir = dl[best] / dl[best].norm()
    # 4. causal tests on a subset: ablate r, ablate a random direction, ablate the L2-L1 direction, add r to L1
    rng = torch.Generator().manual_seed(0)
    rand = torch.randn(r.shape, generator=rng); rand = rand / rand.norm()
    sub = [it for it in items if it['condition'] in payload['ablate_conds']]
    for name, vec in (('ablate_r', r), ('ablate_rand', rand), ('ablate_s', sdir)):
        state['ablate'] = vec.to('cuda', torch.bfloat16)
        _, lp = forward([it['prompt'] for it in sub])
        a, b_, l = ar(lp)
        for it, x, y, z in zip(sub, a.tolist(), b_.tolist(), l.tolist()):
            it[name] = dict(pA=x, pR=y, logit=z)
    state['ablate'] = None
    gap = float(np.mean(proj_harm) - np.mean(proj_harmless))
    l1 = [it for it in items if it['condition'] == 'L1']
    for scale in (0.5, 1.0):
        state['add'], state['add_layer'] = (scale * gap * r).to('cuda', torch.bfloat16), best
        _, lp = forward([it['prompt'] for it in l1])
        a, b_, l = ar(lp)
        for it, x, y, z in zip(l1, a.tolist(), b_.tolist(), l.tolist()):
            it[f'add_r_{scale}'] = dict(pA=x, pR=y, logit=z)
    state['add'] = None
    # 5. example generations for L2 with and without ablation
    l2 = [it for it in items if it['condition'] == 'L2'][:12]
    ex_base = generate([it['prompt'] for it in l2], 60)
    state['ablate'] = r.to('cuda', torch.bfloat16)
    ex_abl = generate([it['prompt'] for it in l2], 60)
    state['ablate'] = None
    return dict(layer=best, n_layers=nL, selection=sel, cos_L2L1_vs_refusal_by_layer=cos_l,
                s_norm_by_layer=dl.norm(dim=-1).tolist(), base_harmful_refusal_prob=base_ref,
                proj_harmful_mean=float(np.mean(proj_harm)), proj_harmless_mean=float(np.mean(proj_harmless)),
                add_gap=gap, harmful_gen_base=gen_base, harmful_gen_ablated=gen_abl,
                l2_examples=[dict(snap=it['snap'], agent=it['agent'], base=b_, ablated=a) for it, b_, a in zip(l2, ex_base, ex_abl)],
                n_accept_ids=len(acc_ids), n_refuse_ids=len(ref_ids), items=items, seconds=time.time() - t0)


@app.local_entrypoint()
def main():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from sting import interp as I
    from sting import prompts as P
    data = I.OUT / 'e3_data'
    rd = lambda n: [x['instruction'] for x in json.loads((data / f'{n}.json').read_text())]
    rnd = random.Random(0)
    harmless = rd('harmless_train'); rnd.shuffle(harmless)
    harmful = rd('harmful_train'); rnd.shuffle(harmful)
    items = []
    for path in I.snaps(I.LLAMA):
        g0 = I.load(path, I.LLAMA)
        for s in g0.agents():
            if not s.alive:
                continue
            for c in E3_CONDS:
                g, offer = I.variant(g0, s, c)
                items.append(dict(snap=path.stem, agent=s.name, condition=c,
                                  prompt=P.OFFER_REPLY.format(**g.ctx(s), uc_name=g.uc.name, offer=offer)))
    payload = dict(harmful_train=harmful[:128], harmless_train=harmless[:128], harmful_val=rd('harmful_val')[:32],
                   items=items, ablate_conds=ABLATE_CONDS)
    print(f'[e3] {len(items)} game prompts', flush=True)
    res = run.remote(payload)
    out = I.OUT / 'Llama8B_refusal'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'raw_results.json').write_text(json.dumps(res, ensure_ascii=False))
    print('[e3] layer', res['layer'], 'seconds', round(res['seconds']), flush=True)


@app.function(image=image, gpu='H100', volumes={'/cache': cache}, timeout=30 * 60)
def patch(payload: dict) -> dict:
    """E3c: move L2 toward L1 at the last token of one layer, either fully or only along the refusal direction."""
    import re
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = 'left'; tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map='cuda').eval()
    layers = model.model.layers
    norm = lambda s: re.sub(r'[\s*"\'`_#:\-\.]', '', s).upper()
    acc, ref = [], []
    for i in range(len(tok)):
        t = norm(tok.decode([i]))
        if len(t) >= 2 and ('ACCEPT'.startswith(t) or t.startswith('ACCEPT')):
            acc.append(i)
        elif len(t) >= 2 and any(w.startswith(t) or t.startswith(w) for w in ('REFUSE', 'REJECT', 'DECLINE')):
            ref.append(i)
    state = {'vec': None, 'layer': None}

    def hook(idx):
        def f(mod, args, out):
            if state['vec'] is None or idx != state['layer']:
                return out
            h = out[0] if isinstance(out, tuple) else out
            h = h.clone(); h[:, -1] = h[:, -1] + state['vec']
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return f
    for i, l in enumerate(layers):
        l.register_forward_hook(hook(i))

    def enc(texts):
        chats = [tok.apply_chat_template([{'role': 'user', 'content': t}], tokenize=False, add_generation_prompt=True) for t in texts]
        return tok(chats, return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')

    @torch.no_grad()
    def fwd(texts, hidden=False, bs=8):
        H, P = [], []
        for i in range(0, len(texts), bs):
            b = enc(texts[i:i + bs]); o = model(**b, output_hidden_states=hidden)
            lp = torch.log_softmax(o.logits[:, -1].float(), -1)
            P.append((lp[:, acc].exp().sum(-1) / (lp[:, acc].exp().sum(-1) + lp[:, ref].exp().sum(-1))).cpu())
            if hidden:
                H.append(torch.stack([h[:, -1].float() for h in o.hidden_states[1:]], 1).cpu())
        return (torch.cat(H) if hidden else None), torch.cat(P)
    hh, _ = fwd(payload['harmful_train'], True); hl, _ = fwd(payload['harmless_train'], True)
    dirs = hh.mean(0) - hl.mean(0); dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    it = payload['items']
    even = lambda x: int(x['snap'].split('_')[1]) % 2 == 0
    l1e = [x['prompt'] for x in it if x['condition'] == 'L1' and even(x)]
    l2e = [x['prompt'] for x in it if x['condition'] == 'L2' and even(x)]
    l1o = [x['prompt'] for x in it if x['condition'] == 'L1' and not even(x)]
    l2o = [x['prompt'] for x in it if x['condition'] == 'L2' and not even(x)]
    H1, _ = fwd(l1e, True); H2, _ = fwd(l2e, True)
    d = H2.mean(0) - H1.mean(0)                                         # [nL, D]
    _, base1 = fwd(l1o); _, base2 = fwd(l2o)
    out = dict(base_L1=base1.mean().item(), base_L2=base2.mean().item(), n_L1=len(l1o), n_L2=len(l2o), layers={})
    for li in payload['layers']:
        r = dirs[li]; dv = d[li]; along = (dv @ r) * r; orth = dv - along
        row = dict(diff_norm=dv.norm().item(), refusal_component=(dv @ r).item(), cos=(dv @ r / dv.norm()).item())
        for name, vec, texts in (('L2_minus_full', -dv, l2o), ('L2_minus_refusal_part', -along, l2o),
                                 ('L2_minus_orth_part', -orth, l2o), ('L1_plus_full', dv, l1o),
                                 ('L1_plus_refusal_part', along, l1o)):
            state['vec'], state['layer'] = vec.to('cuda', torch.bfloat16), li
            _, p = fwd(texts); row[name] = p.mean().item()
        state['vec'] = None
        out['layers'][li] = row
    return out


@app.local_entrypoint()
def patch_main():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from sting import interp as I
    from sting import prompts as P
    data = I.OUT / 'e3_data'
    rd = lambda n: [x['instruction'] for x in json.loads((data / f'{n}.json').read_text())]
    rnd = random.Random(0)
    harmless = rd('harmless_train'); rnd.shuffle(harmless)
    harmful = rd('harmful_train'); rnd.shuffle(harmful)
    items = []
    for path in I.snaps(I.LLAMA):
        g0 = I.load(path, I.LLAMA)
        for s in g0.agents():
            if s.alive:
                for c in ('L1', 'L2'):
                    g, offer = I.variant(g0, s, c)
                    items.append(dict(snap=path.stem, agent=s.name, condition=c,
                                      prompt=P.OFFER_REPLY.format(**g.ctx(s), uc_name=g.uc.name, offer=offer)))
    res = patch.remote(dict(harmful_train=harmful[:128], harmless_train=harmless[:128], items=items,
                            layers=[5, 9, 13, 16, 20, 24, 28, 31]))
    out = I.OUT / 'Llama8B_refusal'
    (out / 'patch_results.json').write_text(json.dumps(res, indent=1))
    print('[e3c]', json.dumps(res, indent=1), flush=True)
