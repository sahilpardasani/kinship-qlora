#!/bin/bash
set -eu
cd "$(dirname "$0")"
mlx_lm.lora --model mlx-community/Qwen3-0.6B-4bit --train --data data \
  --adapter-path adapters --mask-prompt --batch-size 4 --num-layers 12 \
  --iters 120 --learning-rate 0.0001 --max-seq-length 256 --seed 42 \
  --steps-per-report 10 --steps-per-eval 30 --val-batches 2 --save-every 30
