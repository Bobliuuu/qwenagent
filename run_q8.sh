#!/bin/bash
# q8: PERMUTATION TRACKING (pure state tracking). train P=4, 8-16 swaps; test longer sequences (to ~96 swaps) and P=5,6
m=$1; s=$2; x=""
[ "$s" = "0" ] && x="--save ckpt/q8_${m}_s0.pt --trace traces/q8_${m}_s0.json"
exec ~/venv-baseline/bin/python train.py --mode $m --seed $s --K 4 --occ 1 --task perm --train_P 4 --test_P 3,4,5,6 --steps 8000 $x --out results/q8_${m}_s$s.json > logs/q8_${m}_s$s.log 2>&1
