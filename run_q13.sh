#!/bin/bash
# q13: robustness sweep. q9 setting at other injection layers / model sizes.
# usage: MODEL=Qwen/Qwen3-1.7B LAYER=12 ./run_q13.sh MODE SEED     (or args 3,4 = model layer)
m=$1; s=$2; M=${3:-${MODEL:-Qwen/Qwen3-0.6B}}; L=${4:-${LAYER:-12}}
short=$(basename $M | sed 's/Qwen3-//')
tag=q13_${short}_L${L}_${m}_s${s}
mkdir -p results logs
exec ~/venv-baseline/bin/python train.py --model $M --layer $L --mode $m --seed $s --K 4 --occ 1 --ret_norm 1 --rename 1 --task query --train_P 2,3 --test_P 2,3,4,6,8,12,16 --steps 8000 --out results/${tag}.json > logs/${tag}.log 2>&1
