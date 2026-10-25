#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/.venv/bin:$PATH"
L="$PWD/runs/interp/logs"; mkdir -p "$L"
python -m modal billing summary --json > "$L/billing_before_e3b.json"
python -m sting.budget --billing "$L/billing_before_e3b.json" --gpu H100 --max-minutes 40 --cap 38 || exit 3
echo "=== START e3b $(date -u +%FT%TZ)"
PYTHONUNBUFFERED=1 python -m modal run sting/refusal_dir.py 2>&1 | tee "$L/console_e3b.log"
echo "=== END e3b rc=${PIPESTATUS[0]} $(date -u +%FT%TZ)"
python -m modal billing summary --json > "$L/billing_after_e3b.json"
python -m modal app list --json > "$L/apps_after_e3b.json"
