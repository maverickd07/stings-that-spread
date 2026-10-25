#!/usr/bin/env bash
# Lightweight login-node inspection only. Does not allocate GPUs or run inference.
set -u
base="$HOME/inward/results/apart_sting_hpc_preflight_20261024"
mkdir -p "$base"
report=$(mktemp "$base/preflight_XXXXXX.txt")
exec > >(tee "$report") 2>&1
check() {
  printf '\n$'
  printf ' %q' "$@"
  printf '\n'
  "$@" || printf 'Inspection unavailable (exit %s); continue.\n' "$?"
}
check date -u
check hostname
check id -un
check lquota
check squeue -h -u "$(id -un)" -o '%i|%j|%T|%P|%b|%l'
check sinfo -h -N -p stud -o '%N|%P|%G|%m|%f'
check scontrol show partition stud
check sacctmgr -n -P show assoc where "user=$(id -un)"
check sacctmgr -n -P show qos stud
printf '\nSCRATCH=%s\n' "${SCRATCH:-unset}"
check df -h "$HOME" "${SCRATCH:-$HOME}"
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  if conda activate inward; then
    check python - <<'PY'
import importlib.metadata as m
import json
import sys
from pathlib import Path
print('Python:', sys.version)
for name in ('torch', 'transformers', 'vllm', 'huggingface_hub', 'pandas', 'matplotlib'):
    try:
        print(name, m.version(name))
    except m.PackageNotFoundError:
        print(name, 'NOT INSTALLED')
root = Path.home() / 'inward' / 'models'
for p in sorted(root.glob('*/config.json')):
    c = json.loads(p.read_text())
    print('Existing model:', p.parent, 'dtype:', c.get('torch_dtype'), 'architecture:', c.get('architectures'))
PY
  fi
fi
printf '\nSaved report: %s\nNo job submitted; no model loaded or downloaded.\n' "$report"
