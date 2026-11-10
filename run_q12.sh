#!/bin/bash
# q12: realistic interleaved-entity task. usage: run_q12.sh TASK MODE SEED   (TASK = ledger|files)
t=$1; m=$2; s=$3; ex=${EXTRA:-}
exec ~/venv-baseline/bin/python train.py --mode $m --seed $s --K 4 --occ 1 --ret_norm 1 --task $t --train_P 2,3 --test_P 2,3,4,6,8,12 --opr 2,4 --steps ${STEPS:-6000} --n_eval 100 $ex --out results/q12_${t}_${m}_s$s.json > logs/q12_${t}_${m}_s$s.log 2>&1
