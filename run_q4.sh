#!/bin/bash
# usage: run_q4.sh MODE SEED : curriculum query->subset, train P 2-4, test up to 16
m=$1; s=$2; x=""
[ "$s" = "0" ] && x="--save ckpt/q4_${m}_s0.pt --trace traces/q4_${m}_s0.json"
exec ~/venv-baseline/bin/python train.py --mode $m --seed $s --curriculum query:2500,subset:2500 $x --out results/q4_${m}_s$s.json > logs/q4_${m}_s$s.log 2>&1
