#!/usr/bin/env bash
# Values batch (Daniel 2026-10-24 22:25 "都跑吧"); target account ~$23, safety stop $38.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/.venv/bin:$PATH" MPLCONFIGDIR="$PWD/logs/matplotlib-cache"
B="$PWD/runs/values"; L="$B/logs"; mkdir -p "$L"
export STING_HOST_LOG_DIR="$L" STING_MODAL_MAX_SEQS=64 STING_MODAL_MAX_BATCHED=8192
CAP=${CAP:-38}
host() {  # tag model gpu max_minutes command
  python -m modal billing summary --json > "$L/billing_before_$1.json"
  python -m sting.budget --billing "$L/billing_before_$1.json" --gpu "$3" --max-minutes "$4" --cap $CAP || { echo "SAFETY_STOP before $1"; return 3; }
  echo "=== START $1 $(date -u +%FT%TZ)"
  STING_RAW_LOG="$L/raw_$1.jsonl" PYTHONUNBUFFERED=1 \
    python -m sting.modal_host --model "$2" --gpu "$3" --max-minutes "$4" -- bash -c "$5" 2>&1 | tee "$L/console_$1.log"
  echo "=== END $1 rc=${PIPESTATUS[0]} $(date -u +%FT%TZ)"
  python -m modal billing summary --json > "$L/billing_after_$1.json"
  python -m modal app list --json > "$L/apps_after_$1.json"
}
for spec in ${ONLY:-qwen mistral llama}; do case $spec in
  qwen)    host Qwen32B Qwen/Qwen2.5-32B-Instruct H100 40 "python -m sting.values gen --model Qwen/Qwen2.5-32B-Instruct" ;;
  mistral) host Mistral24B mistralai/Mistral-Small-24B-Instruct-2501 H100 40 "python -m sting.values gen --model mistralai/Mistral-Small-24B-Instruct-2501" ;;
  llama)   host Llama8B NousResearch/Meta-Llama-3.1-8B-Instruct H100 25 "python -m sting.values gen --model NousResearch/Meta-Llama-3.1-8B-Instruct" ;;
esac; done
[[ -z "${SKIP_JUDGE:-}" ]] && host phi4 microsoft/phi-4 H100 45 "python -m sting.values judge"
python -m sting.values analyze > "$B/analyze.log" 2>&1
echo ALL_DONE
