#!/usr/bin/env bash
# Full plan on Modal (event credit: $50, Functions only). One model per server, so every table is
# homogeneous: all three LLM seats run the same model. Prices are rough Modal list prices.
#
#   bash scripts/modal_plan.sh screen      # ~$5-10: which models are off the ceiling?
#   bash scripts/modal_plan.sh main MODEL GPU    # ~$5-12 per model
#   bash scripts/modal_plan.sh ceiling     # ~$3: small model of the original study
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -x .venv/bin/python ]]; then
  export PATH="$PWD/.venv/bin:$PATH"
fi
mkdir -p logs/modal
export MPLCONFIGDIR="$PWD/logs/matplotlib-cache"
host() {
  local tag stamp rc=0
  tag=$(basename "$1"); stamp=$(date -u +%Y%m%dT%H%M%SZ)
  python -m modal billing summary --json > "logs/modal/billing_before_${tag}_${stamp}.json"
  python -m sting.budget --billing "logs/modal/billing_before_${tag}_${stamp}.json" --gpu "$2" --cap "${STING_BUDGET_CAP:-40}"
  STING_RAW_LOG="$PWD/logs/modal/raw_${tag}_${stamp}.jsonl" \
  PYTHONUNBUFFERED=1 python -m sting.modal_host --model "$1" --gpu "$2" -- bash -c "$3" \
    2>&1 | tee "logs/modal/console_${tag}_${stamp}.log" || rc=$?
  python -m modal billing summary --json > "logs/modal/billing_after_${tag}_${stamp}.json"
  python -m sting.budget --billing "logs/modal/billing_after_${tag}_${stamp}.json" --gpu "$2" --report-only
  return "$rc"
}

case "${1:-}" in
resume-main)
  # Resume the interrupted Mistral judge; completed snapshots/branches are untouched.
  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  python -m modal billing summary --json > "logs/modal/billing_before_resume_${stamp}.json"
  M=mistralai/Mistral-Small-24B-Instruct-2501; tag=$(basename "$M")
  if python -m sting.budget --billing "logs/modal/billing_before_resume_${stamp}.json" --gpu H100 --max-minutes 90 --cap 35; then
    export STING_BUDGET_CAP=35
    host "$M" H100 "python -m sting.judge --out runs/main_$tag --model $M && \
                     python -m sting.run contagion --out runs/main_$tag && \
                     python -m sting.run snapshots --out runs/values_$tag --n 30 --models $M --agent-mode values && \
                     python -m sting.run branches --out runs/values_$tag --conditions L1,L2,L3 --after-rounds 0 && \
                     python -m sting.judge --out runs/main_$tag --model $M && \
                     python -m sting.judge --out runs/values_$tag --model $M"
    python -m sting.analyze --out "runs/main_$tag" --fig-dir "runs/main_$tag/analysis"
    python -m sting.analyze --out "runs/values_$tag" --fig-dir "runs/values_$tag/analysis"
    M=Qwen/Qwen2.5-32B-Instruct; tag=$(basename "$M")
    host "$M" H100 "python -m sting.run contagion --out runs/main_$tag && \
                     python -m sting.run snapshots --out runs/values_$tag --n 30 --models $M --agent-mode values && \
                     python -m sting.run branches --out runs/values_$tag --conditions L1,L2,L3 --after-rounds 0 && \
                     python -m sting.judge --out runs/main_$tag --model $M && \
                     python -m sting.judge --out runs/values_$tag --model $M"
    python -m sting.analyze --out "runs/main_$tag" --fig-dir "runs/main_$tag/analysis"
    python -m sting.analyze --out "runs/values_$tag" --fig-dir "runs/values_$tag/analysis"
  else
    host "$M" H100 "python -m sting.judge --out runs/main_$tag --model $M"
    python -m sting.analyze --out "runs/main_$tag" --fig-dir "runs/main_$tag/analysis"
    echo '[budget] Add-ons skipped: reserve exceeds $35 total.'
  fi ;;
main-priority)
  # User-approved 2026-10-24: only these two models; core conditions finish first.
  for M in Qwen/Qwen2.5-32B-Instruct mistralai/Mistral-Small-24B-Instruct-2501; do
    tag=$(basename "$M")
    host "$M" H100 "python -m sting.run snapshots --out runs/main_$tag --n 30 --models $M && \
                     python -m sting.run branches --out runs/main_$tag --conditions N0,L0,L1,L2,L3,PC --after-rounds 3 --save-state && \
                     python -m sting.judge --out runs/main_$tag --model $M"
    python -m sting.analyze --out "runs/main_$tag" --fig-dir "runs/main_$tag/analysis"
  done
  # Reserve two full 45-minute sessions under the $35 total add-on ceiling.
  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  python -m modal billing summary --json > "logs/modal/billing_before_addons_${stamp}.json"
  if python -m sting.budget --billing "logs/modal/billing_before_addons_${stamp}.json" --gpu H100 --max-minutes 90 --cap 35; then
    export STING_BUDGET_CAP=35
    for M in Qwen/Qwen2.5-32B-Instruct mistralai/Mistral-Small-24B-Instruct-2501; do
      tag=$(basename "$M")
      host "$M" H100 "python -m sting.run contagion --out runs/main_$tag && \
                       python -m sting.run snapshots --out runs/values_$tag --n 30 --models $M --agent-mode values && \
                       python -m sting.run branches --out runs/values_$tag --conditions L1,L2,L3 --after-rounds 0 && \
                       python -m sting.judge --out runs/main_$tag --model $M && \
                       python -m sting.judge --out runs/values_$tag --model $M"
      python -m sting.analyze --out "runs/main_$tag" --fig-dir "runs/main_$tag/analysis"
      python -m sting.analyze --out "runs/values_$tag" --fig-dir "runs/values_$tag/analysis"
    done
  else
    echo '[budget] Add-ons skipped: conservative reserve would reach the $35 total ceiling.'
  fi ;;
screen)
  # 8 snapshots, L0 and L2 offers only, no follow-up play. Keep models with L2 acceptance in 15-85%.
  for spec in "Qwen/Qwen2.5-72B-Instruct H100:2" "Qwen/Qwen2.5-32B-Instruct H100" \
              "mistralai/Mistral-Small-24B-Instruct-2501 H100"; do
    set -- $spec; tag=$(basename "$1")
    host "$1" "$2" "python -m sting.run snapshots --out runs/screen_$tag --n 8 --models $1 && \
                    python -m sting.run branches --out runs/screen_$tag --conditions L0,L2 --after-rounds 0 && \
                    python -m sting.judge --out runs/screen_$tag --model $1"
    python -m sting.analyze --out "runs/screen_$tag" | sed -n '/Entrapment/,/^$/p'
  done ;;
main)
  M="$2"; G="$3"; tag=$(basename "$M")
  host "$M" "$G" "python -m sting.run snapshots --out runs/main_$tag --n 30 --models $M && \
                  python -m sting.run branches --out runs/main_$tag --conditions N0,L0,L1,L2,L3,PC,D0,D10,D50 --after-rounds 3 --contagion && \
                  python -m sting.run branches --out runs/main_$tag --conditions N0,L2,PC --after-rounds -1 --exit --limit 10 && \
                  python -m sting.run snapshots --out runs/values_$tag --n 30 --models $M --agent-mode values && \
                  python -m sting.run branches --out runs/values_$tag --conditions L1,L2,L3 --after-rounds 0 && \
                  python -m sting.judge --out runs/main_$tag --model $M && python -m sting.judge --out runs/values_$tag --model $M"
  python -m sting.analyze --out "runs/main_$tag" --fig-dir "runs/main_$tag/analysis"
  python -m sting.analyze --out "runs/values_$tag" --fig-dir "runs/values_$tag/analysis" ;;
ceiling)
  M=Qwen/Qwen2.5-7B-Instruct; tag=$(basename "$M")
  host "$M" A10G "python -m sting.run snapshots --out runs/ceiling_$tag --n 20 --models $M && \
                  python -m sting.run branches --out runs/ceiling_$tag --conditions N0,L0,L2,L3,PC --after-rounds 3 && \
                  python -m sting.judge --out runs/ceiling_$tag --model $M"
  python -m sting.analyze --out "runs/ceiling_$tag" ;;
*) sed -n '2,9p' "$0"; exit 1 ;;
esac
