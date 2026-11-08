#!/bin/bash
# q6: occupancy-gated returns, K=4 (as in q2), fixed oracle mapping, query task, train P 2-3, test to 16, 8000 steps
m=$1; s=$2; x=""
[ "$s" = "0" ] && x="--save ckpt/q6_${m}_s0.pt --trace traces/q6_${m}_s0.json"
exec ~/venv-baseline/bin/python train.py --mode $m --seed $s --K 4 --occ 1 --task query --train_P 2,3 --test_P 2,3,4,6,8,12,16 --steps 8000 $x --out results/q6_${m}_s$s.json > logs/q6_${m}_s$s.log 2>&1
