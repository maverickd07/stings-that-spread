#!/usr/bin/env bash
# Robustness run (Daniel 2026-10-24: more models, bigger; stop when done; safety stop at account $38).
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/.venv/bin:$PATH" MPLCONFIGDIR="$PWD/logs/matplotlib-cache"
B="$PWD/runs/robust"; L="$B/logs"; mkdir -p "$L"
export STING_HOST_LOG_DIR="$L" STING_MODAL_MAX_SEQS=64 STING_MODAL_MAX_BATCHED=8192
CAP=${CAP:-18.8}   # Daniel 19:58: size this run so the account lands around $18-20
host() {  # tag model gpu max_minutes max_len command
  python -m modal billing summary --json > "$L/billing_before_$1.json"
  python -m sting.budget --billing "$L/billing_before_$1.json" --gpu "$3" --max-minutes "$4" --cap $CAP || { echo "SAFETY_STOP before $1"; return 3; }
  echo "=== START $1 $(date -u +%FT%TZ)"
  STING_RAW_LOG="$L/raw_$1.jsonl" PYTHONUNBUFFERED=1 \
    python -m sting.modal_host --model "$2" --gpu "$3" --max-minutes "$4" --max-len "$5" -- bash -c "$6" 2>&1 | tee "$L/console_$1.log"
  echo "=== END $1 rc=${PIPESTATUS[0]} $(date -u +%FT%TZ)"
  python -m modal billing summary --json > "$L/billing_after_$1.json"
  python -m modal app list --json > "$L/apps_after_$1.json"
  python -m sting.budget --billing "$L/billing_after_$1.json" --gpu "$3" --report-only --cap $CAP
}
model() {  # tag model gpu max_minutes max_len seed [key-only]
  local extra="&& python -m sting.robust contagion --tag $1"
  [[ -n "${7:-}" ]] && extra=""
  host "$1" "$2" "$3" "$4" "$5" "python -m sting.run snapshots --out $B/$1 --n 30 --seed $6 --models $2 --workers 16 && \
    ROBUST_CONDS=${7:-} python -m sting.robust offers --tag $1 $extra" || echo "SKIPPED $1"
}
for spec in ${ONLY:-72b olmo q32 mis}; do case $spec in
  72b)   model Qwen2.5-72B Qwen/Qwen2.5-72B-Instruct H100:2 35 8192 1000 ;;
  llama) model Llama31-8B NousResearch/Meta-Llama-3.1-8B-Instruct H100 40 8192 1000 ;;
  7b)    model Qwen2.5-7B Qwen/Qwen2.5-7B-Instruct H100 40 8192 1000 ;;
  olmo)  model OLMo2-13B allenai/OLMo-2-1124-13B-Instruct H100 25 4096 1000 ;;
  q32)   model Qwen2.5-32B-new Qwen/Qwen2.5-32B-Instruct H100 20 8192 1030 L1,L2,L2F,VL1 ;;
  mis)   model Mistral24B-new mistralai/Mistral-Small-24B-Instruct-2501 H100 20 8192 1030 L1,L2,L2F,VL1 ;;
esac; done
CAP=20 host phi4 microsoft/phi-4 H100 25 8192 "python -m sting.robust judge"
python -m sting.robust analyze
python -m sting.robust sanity
echo ALL_DONE
