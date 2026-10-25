#!/usr/bin/env bash
# Claude add-on experiments 1-3 (approved 2026-10-24, total cap +$8 over $9.78 => absolute $17.8).
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/.venv/bin:$PATH" MPLCONFIGDIR="$PWD/logs/matplotlib-cache"
L=runs/claude_ext/logs; mkdir -p "$L"
export STING_HOST_LOG_DIR="$L"
CAP=17.8
host() {  # tag model max_minutes command
  python -m modal billing summary --json > "$L/billing_before_$1.json"
  python -m sting.budget --billing "$L/billing_before_$1.json" --gpu H100 --max-minutes "$3" --cap $CAP
  STING_RAW_LOG="$PWD/$L/raw_$1.jsonl" PYTHONUNBUFFERED=1 \
    python -m sting.modal_host --model "$2" --gpu H100 --max-minutes "$3" -- bash -c "$4" 2>&1 | tee "$L/console_$1.log"
  python -m modal billing summary --json > "$L/billing_after_$1.json"
  python -m modal app list --json > "$L/apps_after_$1.json"
  python -m sting.budget --billing "$L/billing_after_$1.json" --gpu H100 --report-only --cap $CAP
}
host qwen Qwen/Qwen2.5-32B-Instruct 15 "python -m sting.claude_ext offers --model Qwen/Qwen2.5-32B-Instruct"
host mistral mistralai/Mistral-Small-24B-Instruct-2501 15 "python -m sting.claude_ext offers --model mistralai/Mistral-Small-24B-Instruct-2501"
host llama NousResearch/Meta-Llama-3.1-8B-Instruct 25 "python -m sting.claude_ext contagion"
host phi4 microsoft/phi-4 15 "python -m sting.claude_ext judge"
python -m sting.claude_ext analyze
echo ALL_DONE
