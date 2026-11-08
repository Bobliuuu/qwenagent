#!/bin/bash
# q7: RANDOM register names + shuffled init order in train and eval (content-based binding must be learned)
m=$1; s=$2; x=""
[ "$s" = "0" ] && x="--save ckpt/q7_${m}_s0.pt --trace traces/q7_${m}_s0.json"
exec ~/venv-baseline/bin/python train.py --mode $m --seed $s --K 4 --occ 1 --rename 1 --task query --train_P 2,3 --test_P 2,3,4,6,8,12,16 --steps 8000 $x --out results/q7_${m}_s$s.json > logs/q7_${m}_s$s.log 2>&1
