# Stings That Spread

**Undercover Agents Mismeasure and Propagate Collusion Among LLM Agents**  
Daniel Weiyu Chen · Apart AI Collusion Sprint · Question 3.2 · October 23–25, 2026

This submission asks whether an undercover agent offering a secret alliance detects an existing willingness to collude or creates the behavior it claims to measure. A new Liar's Bar engine branches saved round-2 states into matched offer, format, monitoring, reflection, and memory conditions. Across the tested open-weight models, measured acceptance depends strongly on the wording and response format; spontaneous private-message collusion is a poor proxy for accepting a particular offer. Imposed alliances and some voluntary offers also increase recruiting at a later table. Further experiments separate monitoring notices from stated detection probability, compare values prompts with strategic reflection, and examine refusal directions and the transmission of refusal memories. Effects vary by model; agreement, imposed alliance membership, and subsequent recruiting are reported separately.

## Results

Start with [the consolidated results](runs/RESULTS_ALL_20261024.md), [interpretability report](runs/interp/REPORT.md), and [values report](runs/values/REPORT.md). The October 24 consolidated report predates the final values batch: its approximately $15 account total is historical. The final values report records **$24.28 in metered Modal usage** (approximately $24 for the sprint; credits covered the recorded compute). This is a historical cost, not a guaranteed reproduction price.

The release contains derived Markdown, CSV, and summary JSON results, including agent-level observation tables, scoring tables, and audit tables. Raw API logs, full per-request response files, serialized game states, and model weights are excluded. Published reports are preserved verbatim, including their limitations and earlier results; consult each report's judge and baseline definitions before combining tables. `N0` denotes a no-offer branch in the engine; report references to an N0 predisposition measure denote the neutral private-message probe and may mean one sample or any of three samples.

## Reproduce

Use **Python 3.12**, run commands from the repository root, and configure your own Modal account:

```bash
git clone https://github.com/maverickd07/stings-that-spread.git
cd stings-that-spread
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[analysis,test]' modal numpy
modal setup
mkdir -p runs logs
```

The original packaging metadata is retained. The experiment environment observed at publication used Modal 1.6.2, NumPy 2.5.4, pandas 3.1.0, matplotlib 3.11.2, and pytest 9.1.1 locally (Python 3.13.9). The remote inference image explicitly pins **Python 3.12, vLLM 0.10.1.1, transformers 4.55.2, and huggingface_hub 0.36.0** in `sting/modal_host.py`. The optional refusal-direction experiment has its own pinned transformers/PyTorch image in `sting/refusal_dir.py`.

First check the publication and run the existing tests without inference:

```bash
python scripts/verify_publication.py
STING_MOCK=1 python -m pytest -q tests
```

### Main games and neutral judging

The main models were `Qwen/Qwen2.5-32B-Instruct` and `mistralai/Mistral-Small-24B-Instruct-2501`, each using 30 attempted seeds 1000–1029. Each produced 28 usable round-2 snapshots and 79 eligible agent × snapshot observations. A table has three LLM seats using the same model and one scripted undercover seat. The screening also tested `Qwen/Qwen2.5-72B-Instruct`.

This example regenerates one model's main games and saved states; repeat it with the Mistral model and its matching output directory:

```bash
M=Qwen/Qwen2.5-32B-Instruct
OUT=runs/main_Qwen2.5-32B-Instruct
python -m sting.modal_host --model "$M" --gpu H100 -- \
  python -m sting.run snapshots --out "$OUT" --n 30 --seed 1000 --models "$M"
python -m sting.modal_host --model "$M" --gpu H100 -- \
  python -m sting.run branches --out "$OUT" --conditions N0,L0,L1,L2,L3,PC \
  --after-rounds 3 --save-state
python -m sting.modal_host --model "$M" --gpu H100 -- \
  python -m sting.run contagion --out "$OUT"
python -m sting.modal_host --model "$M" --gpu H100 -- \
  python -m sting.judge --out "$OUT" --model "$M"
python -m sting.analyze --out "$OUT"
```

The main-run `sting.judge` command above reproduces the **historical same-model labels**, which are distinct from later **`microsoft/phi-4`, temperature 0** judging. Neutral rechecking, wording, robustness, interpretability, and values use the Phi-4 pathways below. Keep `STING_RAW_LOG` set for judge commands that enforce provenance logging, for example:

```bash
STING_RAW_LOG="$PWD/logs/recheck_phi4.jsonl" \
python -m sting.modal_host --model microsoft/phi-4 --gpu H100 -- \
  python -m sting.recheck --run-judge
```

`sting.recheck`'s A3/A6 judge is an agreement/refusal-reason recheck. The later baseline relabeling uses `sting.wording --judge`, which also rejudges N0 collusion with the original collusion prompt. Read each report for the exact stage and denominator.

### Wording and additional models

After generating the main snapshots, run each main model's wording offers on its own Modal server, then judge on Phi-4:

```bash
STING_RAW_LOG="$PWD/logs/wording_qwen.jsonl" \
python -m sting.modal_host --model Qwen/Qwen2.5-32B-Instruct --gpu H100 -- \
  python -m sting.run branches --offers-only \
  --source-out runs/main_Qwen2.5-32B-Instruct \
  --out runs/wording/Qwen2.5-32B-Instruct \
  --conditions L1S,L2P,L1F,L2F,L2SEND
# Repeat with the Mistral model, source directory, and output directory.
STING_RAW_LOG="$PWD/logs/wording_phi4.jsonl" \
python -m sting.modal_host --model microsoft/phi-4 --gpu H100 -- \
  python -m sting.wording --judge
python -m sting.wording
```

`sting.additional_model` provides `generate`, `judge`, and `analyze` for `NousResearch/Meta-Llama-3.1-8B-Instruct` responding on Qwen's saved states. Run generation on the Llama server and judging on Phi-4, with `STING_RAW_LOG` configured. This is a cross-model response check, not new Llama self-play. The NousResearch mirror was used in the recorded runs; see the consolidated report for tokenizer/revision caveats.

### Extensions

The original launch scripts are included unchanged. They use account-total budget caps from the sprint: review `CAP`/`STING_BUDGET_CAP` for your own account. The main-plan launcher also performs screening and the older independent-game values comparison; the later matched-state values intervention is in `claude_ext`.

| Stage | Launcher / module | Dependencies |
|---|---|---|
| Main and screening | `scripts/modal_plan.sh` | Own Modal account; choose stage and model |
| Older independent-game values / contagion summary | `sting.extension_report` | Main games and `runs/values_<model>` generated by the main-plan launcher |
| Matched-state values, deterrence, Llama contagion | `scripts/claude_ext.sh` | Both main models, baseline/wording labels, Llama source states |
| More models and fresh-seed replications | `scripts/robust.sh` | Main games and extensions |
| E1/E2 prompt/logprob and memory experiments | `scripts/interp.sh` | Main games, wording/Llama results, robustness labels |
| E3 refusal-direction experiment | `scripts/interp_e3.sh` | Local raw snapshots and external instruction splits; see below |
| V1/V2/V3 reflection, durability, transmission | `scripts/values.sh`; `scripts/values_backfill.sh` | Main states, interpretability helpers and neutral labels |

Robustness includes `Qwen/Qwen2.5-72B-Instruct` (two H100 GPUs), `allenai/OLMo-2-1124-13B-Instruct`, and new seeds 1030–1059 for the two main models. Llama-8B is the mirror ID above. The launcher also lists optional 7B/Llama self-play cases; listing them does not mean every optional case was part of the reported dataset.

The E3 script expects harmful/harmless training and harmful validation instruction splits in `runs/interp/e3_data/`. Those external datasets and raw activation/weight artifacts are not redistributed. The experiment code describes the expected files and selection procedure. Reproducing E3 requires supplying those inputs separately.

**Reproduction limits:** the public summary release supports inspecting results and regenerating experiments from the code, but the original stochastic histories and pickle snapshots are not shipped. The raw-data-dependent analysis commands cannot rerun on summary files alone. Recreate their input stages first, in a separate checkout/output tree; the launchers can otherwise skip existing filenames. Engine seeds, prompt text, model IDs, and server settings are provided, but per-request sampling was not fully seeded and model downloads are not revision-locked everywhere, so new generations need not match the published numbers exactly. No paid inference is required to view results or run the mock tests.

## Final paper figures

`figures/make_figures.py` regenerates all final main-text and appendix plots from frozen, derived summary inputs embedded in the script. It includes the corrected distinctions between first-token probabilities and generated acceptance, the joint PC intervention, and the continuing values cue. The inputs correspond to `runs/RESULTS_ALL_20261024.md`, `runs/interp/results.json`, and `runs/values/results.json`; they are not new experimental runs.

```bash
python -m pip install matplotlib
python figures/make_figures.py --out figures/generated
```

The script writes PDF/PNG figures, derived tables and a font audit. It uses Python's standard library for the exact Fisher calculation, so SciPy is not required. Labels are sized for full-width (6.3-inch) inclusion in the paper. This plotting script was adapted from the author's final figure source for this release; the older engine and run-specific source copies listed in `COPY_MANIFEST.json` remain unchanged.

The public release contains derived summaries, selected observation/audit tables and prompt examples, rather than complete raw logs or serialized states. Full raw logs are available from the author on request.

## Prompts

[The prompt catalogue](prompts/README.md) maps every condition family to its exact source text and builder. `prompts/source_templates.json` exports literal constants and dynamic string fragments from current and archived source without paraphrasing. `prompts/recorded_examples/` contains representative **exact stored request prompts**, including their game context, for each available directory × condition × offer rung. These are prompt-only extracts, not raw response logs. Multi-turn V2 prompt examples in `prompts/reconstructed_v2/` are reconstructed by replaying the exact builder with saved turns and trusted snapshots, without inference; reconstructed turns were checked against their saved originals. Both indices record source and prompt SHA-256 hashes.

## Repo layout

```text
sting/                    Engine, inference, judges, rechecks and experiment modules
scripts/                  Original launch/analysis scripts and publication verification/export tools
figures/make_figures.py    Final paper figure generator from frozen summary inputs
repro/                    Offline probe of the starter's alliance-memory path
tests/                    Original mock and runtime regression tests
runs/                     Published reports, derived CSV tables and summary JSON
  main_*/recheck/          Neutral rechecks and wording statistics
  additional_Llama31_8B/   Llama response recheck and archived source files
  wording/source/         Archived wording/recheck source versions
  claude_ext/, robust/     Extension and replication summaries
  interp/, values/        Final reports and scores/results
prompts/                  Exact text catalogue and representative request prompts
licenses/                 Unmodified upstream licence texts
COPY_MANIFEST.json        Byte-for-byte source-copy provenance
PUBLICATION_SHA256SUMS    Checksums for the public release files
```

## Preexisting code

This project builds on Zeng & Rudzicz's [LLM-Agent-collusion / Liar's Bar code](https://github.com/AgentSafety2025/LLM-Agent-collusion) as distributed in the [poseidon-research/collusion_hackathon starter](https://github.com/poseidon-research/collusion_hackathon/tree/main/llm-agent-collusion-2). The sprint harness is a separate reimplementation; it is not the complete upstream repository.

| Component | Origin and changes |
|---|---|
| `sting/rules_vanilla.txt` | Copied verbatim from the starter's `Collusion_Liars_Bar/prompts/rule_vanilla.txt`; byte-for-byte equality checked against starter commit `e6fb1308dae9cf97ae42aec1c5f7889d9e2291e8`. |
| `sting/prompts.py::PROBE` | Adapted from upstream `prompts/secret_channel/tool_offer_prompt.txt`: retains designer-offer wording, updates the live roster and instructions, and changes the final response/reasoning format. It is not a verbatim reproduction of the whole original prompt. |
| `PC_REMINDER` and imposed alliance control | Adapt the original `src/game/game_secret_channel.py` challenge-time alliance reminder; the sprint restricts that instruction to the positive control. `PC_NOTICE` expresses the corresponding designer-imposed alliance. |
| Rules, scoring and game setup | Based on the original Liar's Bar mechanics; implemented anew in `sting/game.py`, with shuffled seats, snapshots, branching and a scripted undercover player. |
| Modal serving | `sting/modal_host.py` was written for this sprint, informed by the starter's shared `poseidon_compute.modal_serve` approach, adding tensor parallelism, bounded sessions and single-container concurrent serving. |
| New sprint work, October 23–25, 2026 | Python engine and clients; paired neutral probes and offer ladder; persistence and monitoring interventions; wording/format checks and Phi-4 rejudging; cross-game memory, robustness, interpretability/refusal-direction and values experiments; launchers, analysis, tests and resulting data. Archived run-specific versions are retained under `runs/*/source/sting/`. Publication documentation, prompt exports, manifests and verification utilities were added for this release. |

The upstream **LICENSE files contain Apache License 2.0**, verified in both the starter and original repository (original commit `fa7d6718c3c3a1f865e452740d7ac4300c39f172`). The original repository's README says MIT, but its actual [LICENSE](https://github.com/AgentSafety2025/LLM-Agent-collusion/blob/fa7d6718c3c3a1f865e452740d7ac4300c39f172/LICENSE) is Apache-2.0. This release uses **Apache-2.0**, preserves both upstream licence texts in `licenses/`, and describes reused/adapted material in [NOTICE](NOTICE). Model weights and external datasets retain their own terms and are not included.

## Publication scope

Included: all Python/Shell/text code from the local sprint engine and run-specific source folders, original tests and launchers, exact prompts, and selected derived results. All original copies listed in `COPY_MANIFEST.json` are byte-identical to their originals; the manifest distinguishes project files from desktop run files. The original project README is preserved in `docs/ORIGINAL_README.md`.

Excluded: all raw API/console logs (even small ones), full per-request generation/response JSON, pickle snapshots and branch states, model weights/activation tensors, archives, virtual environments, caches, credential files, billing/account dumps, local machine/HPC handoff notes, and external instruction datasets. No file of 50 MiB or larger is included. Generic environment-variable names in code are configuration references; credentials are not supplied. The publication was scanned for credential patterns and hard-coded key assignments before committing.
