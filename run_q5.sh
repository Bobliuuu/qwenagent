#!/bin/bash
# usage: run_q5.sh MODE SEED : query task (the regime that learns), train P 2-3, test up to 16, K grows at test
m=$1; s=$2; x=""
[ "$s" = "0" ] && x="--save ckpt/q5_${m}_s0.pt --trace traces/q5_${m}_s0.json"
exec ~/venv-baseline/bin/python train.py --mode $m --seed $s --task query --train_P 2,3 --test_P 2,3,4,6,8,12,16 --steps 6000 $x --out results/q5_${m}_s$s.json > logs/q5_${m}_s$s.log 2>&1
