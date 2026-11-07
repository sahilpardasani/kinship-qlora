#!/bin/bash
set -eu
cd "$(dirname "$0")"
if [[ "$(uname -m)" != arm64 ]]; then
  echo "Use a native Apple Silicon Terminal (arm64), then run this again." >&2
  exit 1
fi
if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install 'mlx-lm[train]==0.32.2'
mkdir -p results
python -m pip freeze > results/packages.txt
python evaluate.py --output results/before.json
bash train.sh
python evaluate.py --adapter adapters --output results/after.json
python - <<'PY'
import json
for label in ("before", "after"):
    with open(f"results/{label}.json") as f:
        s = json.load(f)["summary"]
    print(f"{label}: {s['correct']}/{s['total']} terms, {s['pairs_correct']}/{s['pairs']} complete pairs")
PY
