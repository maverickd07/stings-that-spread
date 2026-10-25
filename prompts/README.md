# Exact prompts and condition catalogue

Text is extracted from source and recorded requests, not rewritten for publication. Sources and builders are authoritative. Names, state, history, private messages and memories vary by agent; a condition has many rendered prompts.

- `source_templates.json`: exact literal constants, dictionaries, long strings and f-string source with line numbers. Includes current modules and archived run-specific source. Expressions composed from other constants remain visible in source.
- `recorded_examples/`: 299 representative exact stored **request prompts**, one per available source directory × condition × offer rung. UTF-8 strings are preserved without adding a newline. Historical model utterances inside a request are part of its context. No response field or raw log is copied.
- `recorded_examples_index.json`: condition, model, original relative path, source hash and prompt hash. These are representative examples, not every request.
- `reconstructed_v2/` and its index: exact V2 prompts reconstructed from trusted source snapshots and saved response turns. The unchanged builder was replayed without inference; every reconstructed turn matched the saved offer/reply/notice/acceptance record.
- `../sting/rules_vanilla.txt`: exact rules supplied to agents, copied verbatim from upstream.

Regenerate the source catalogue offline with `python scripts/export_prompt_templates.py`. With access to the original run tree, add `--raw-runs /path/to/runs` to regenerate examples. The original raw tree is not distributed.

## Core design

| Condition / task | Exact source and construction |
|---|---|
| N0 | `game.CONDITIONS['N0']`: no offer text. Predisposition/recruiting use `prompts.PM_TOOL` with empty memory or `prompts.MEMORY`. |
| L0, L1, L2, L3 | `prompts.OFFERS` inserted into `prompts.OFFER_REPLY` by the game engine: benign, soft, explicit non-challenge, and hand-sharing/point-splitting offers. |
| Repeated offers 2/3 | `prompts.FOLLOW_UPS`, used after refusal according to `game.py`. |
| PC | `prompts.PC_NOTICE`, `PC_REMINDER`; imposed alliance, not a voluntary reply. |
| D0, D10, D50 | `prompts.DETERRENCE` with `(p,k)=(0,5),(10,5),(50,5)` plus L2. |
| D50k2, D10k20 | Same notice with `(50,2),(10,20)` plus L2; optional engine conditions, not a claim they were all run. |
| rules / values mode | Empty string or `prompts.AGENT_MODES['values']` appended by `Game.rules`. |
| Designer probe | `prompts.PROBE`, adapted from the upstream designer tool offer. |
| Actions / exit | `prompts.PLAY`, `CHALLENGE`, `EXIT`; state assembled by `Game.ctx`. |

## Wording and extensions

| Condition | Exact source and construction |
|---|---|
| L1S, L2P | `wording.WORDING`, in `OFFER_REPLY`. L2P changes multiple phrases, not only secrecy. |
| L1F, L2F | `wording.offer_prompt`: same L1/L2 offer, removes forced ACCEPT/REFUSE instruction. |
| L2SEND | `wording.SEND` and `SEND_JUDGE`; asks whether to send the proposed L2 message. |
| VL1, VL2 | `claude_ext.treated` or `robust.offer_prompt`: values framing plus L1/L2. |
| D0L1, D10L1, D50L1 | Notice with p=0/10/50, k=5 plus **L1**, distinct from core D0/D10/D50's L2. |
| Llama checks | `additional_model.py` uses the wording builders with Llama on Qwen source states. |
| Robustness | `robust.OFFER_CONDS`, `offer_prompt`, `CONTAGION_CONDS`; fresh self-play and fresh-seed replication. |

## Interpretability E1/E2/E3

`interp.E1_CONDS` gives the ordered E1 list; exact replacements are in `L1_EDITS`/`L2_EDITS`, notices in `NOTICE`.

- Base: L0/L1/L2/L3.
- L1 edits: `L1-between`, `L1-uptoyou`, `L1+nobody`, `L1+secret`, `L1+victim`, `L1+explicit`.
- L2 edits: `L2-private`, `L2-secret`, `L2-victim`, `L2-benefit`, `L2-nobody`, `L2-ask`, `L2-allcues`.
- Notice + L1: `D0L1`, `D50L1`, `DmonL1`, `DpenL1`, `DwatchL1`, `DplaceboL1`, `VL1`, `VaffectL1`, `VknewL1`, `VplaceboL1`.
- `mirror`: the agent's own earlier neutral PM, via `own_pm` / `variant`; no single fixed mirror text.
- E2 think-first: `THINK_LINE` replaces `FORCED_LINE`, for L1/L2/VL1/D0L1.
- Monitor estimate: `MONITOR_Q`, for L1/L2/D0L1.
- Memory transplant: `interp.transplant_jobs`: `full`, `none`, `offer_only`, `as_accept`, `as_refuse`. Exact synthetic replies are in that function and the source catalogue.
- E3 uses the same offer builders for `refusal_dir.E3_CONDS` / `ABLATE_CONDS`; activation interventions add no new game prompt wording. External instruction splits are excluded.

## Values V1/V2/V3

| Family | Exact source and construction |
|---|---|
| V1 direct R0–R9 | `values.DIRECTIVE`, `V1_CONDS`, `RULE_NOTICE`, `v1_prompt`, `think_line`; each with L1/L2. R0 is baseline; R9 adds fair-play notice. |
| V1 think-first | Same builder with `think=True`; R1_generic is the baseline. No separate R0 think-first run. |
| V2 C0, CP, CV, CPV, CRid | `V2_CONDS`: no notice, `PEN`, `VAL`, both, or identity reflection. `v2_run` asks L1, applies `LIFT` and `T2_OFFER`, then `T3_OFFER`, stopping on acceptance. |
| V2 C0nolift, CPnolift, CVnolift | Same notices; `T2_PLAIN` replaces `T2_OFFER`, without cancellation notice. |
| V3 M0_none, M1_offer, M2_values, M3_strategy, M4_fear, M5_accept, M6_observe | `V3_MEM`, `REFUSALS`, `v3_jobs`; memories inserted into the next game's `PM_TOOL`. |

V2 records store turns rather than complete request prompts. `reconstructed_v2/` therefore supplies separately labelled reconstructed examples, selecting a saved trajectory reaching the furthest observed turn per model and condition. Exact templates and private-history assembly remain in `values.v2_run` and `P.OFFER_REPLY`. Regenerate with `python scripts/export_v2_prompts.py --raw-runs /path/to/trusted/runs`.

## Judges

`prompts.JUDGE_COLLUSION`, `JUDGE_SUSPICION`, `JUDGE_NEUTRALIZATION`; `recheck.AGREE_PROMPT` / `REASON_PROMPT`; `wording.request` / `SEND_JUDGE`; `claude_ext.agree_request` / `collusion_request`; `interp.THINK_PROMPT` / `RECODE_PROMPT`; and values-specific builders are all preserved in source. Use their exact evidence wrapper and retry logic.

Neutral judging uses `microsoft/phi-4`, temperature 0. Historical same-model labels and parser labels remain distinguished in reports; they are not all Phi-4 labels.
