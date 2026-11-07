#!/bin/bash
set -eu
cd "$(dirname "$0")"
kit=$(pwd)
previous=$(dirname "$kit")
bin="$previous/.venv315/bin"
if [[ ! -x "$bin/mlx_lm.lora" ]]; then bin="$previous/.venv/bin"; fi
if [[ ! -x "$bin/mlx_lm.lora" || ! -f "$previous/adapters/adapters.safetensors" || ! -f "$previous/evaluate.py" ]]; then
  echo "Extract kinship-v2 inside the trained kinship-mac folder, alongside adapters and the existing Python environment." >&2
  exit 1
fi
mkdir -p results
"$bin/python" check.py --adapter "$previous/adapters" --output results/v1-expanded.json
"$bin/mlx_lm.lora" --train --model mlx-community/Qwen3-0.6B-4bit --data data \
  --adapter-path adapters --mask-prompt --batch-size 4 --num-layers 12 \
  --iters 300 --learning-rate 0.0001 --max-seq-length 256 --seed 42 \
  --steps-per-report 10 --steps-per-eval 60 --val-batches 5 --save-every 60
"$bin/python" check.py --adapter adapters --output results/v2-expanded.json
cd "$previous"
"$bin/python" evaluate.py --adapter "$kit/adapters" --output "$kit/results/v2-original30.json"
cd "$kit"
"$bin/python" - <<'PY'
import json
from pathlib import Path
for path in [Path('results/v1-expanded.json'), Path('results/v2-expanded.json'), Path('../results/after.json'), Path('results/v2-original30.json')]:
    if path.exists():
        print(path, json.loads(path.read_text())['summary'])
PY
