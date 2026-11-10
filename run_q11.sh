#!/bin/bash
# q11 (Track B): capacity-controlled sweep + ops-per-register wall.  usage: run_q11.sh <cell> <seed>
# cell = cap_<variant>_d<ds>   variant: multi|oracle|uniform|single_eq|single_ds|single_max|adapter
#      | opr_<mode>            mode: single|multi|oracle
# ALL cells: --match_params 0 (widths set explicitly, see results/q11_notes.md), K=4 train, random 2-letter names.
c=$1; s=$2; K=4
PY=~/venv-baseline/bin/python
COMMON="--seed $s --K $K --occ 1 --ret_norm 1 --rename 1 --long_names 1 --task query --match_params 0"
case $c in
cap_*)
  ds=${c##*_d}; v=${c#cap_}; v=${v%_d*}
  case $v in
    multi|oracle|uniform) mode=$v; sw=0;;
    single_eq)  mode=single;  sw=$((K*ds));;
    single_ds)  mode=single;  sw=$ds;;
    single_max) mode=single;  sw=$((64*ds));;
    adapter)    mode=adapter; sw=$((K*ds));;
  esac
  exec $PY train.py --mode $mode --d_s $ds --single_width $sw $COMMON --train_P 2,3 --test_P 2,3,4,8,16,32,64 --steps ${Q11_STEPS:-6000} --n_eval ${Q11_NEVAL:-200} \
    --out results/q11_${c}_s$s.json > logs/q11_${c}_s$s.log 2>&1;;
opr_*)
  mode=${c#opr_}; sw=0; [ $mode = single ] && sw=$((K*64))
  exec $PY train.py --mode $mode --d_s 64 --single_width $sw $COMMON --train_P 2,3,4 --test_P 2,3,4,6 \
    --opr_curriculum "1,1:800;1,2:800;2,4:1200;4,8:1600;8,12:1800" \
    --out results/q11_${c}_s$s.json > logs/q11_${c}_s$s.log 2>&1;;
esac
