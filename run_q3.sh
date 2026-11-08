#!/bin/bash
# usage: run_q3.sh MODE SEED  (task subset, train P 2-4, test up to 16)
m=$1; s=$2; x=""
[ "$s" = "0" ] && x="--save ckpt/q3_${m}_s0.pt --trace traces/q3_${m}_s0.json"
exec ~/venv-baseline/bin/python train.py --mode $m --seed $s --steps 4000 --task subset $x --out results/q3_${m}_s$s.json > logs/q3_${m}_s$s.log 2>&1
