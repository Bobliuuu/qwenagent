#!/bin/bash
# q9: q7 setting (random names) + normalised slot return (mean over occupied slots)
m=$1; s=$2; x=""
[ "$s" = "0" ] && x="--save ckpt/q9_${m}_s0.pt --trace traces/q9_${m}_s0.json"
exec ~/venv-baseline/bin/python train.py --mode $m --seed $s --K 4 --occ 1 --ret_norm 1 --rename 1 --task query --train_P 2,3 --test_P 2,3,4,6,8,12,16 --steps 8000 $x --out results/q9_${m}_s$s.json > logs/q9_${m}_s$s.log 2>&1
