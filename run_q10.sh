#!/bin/bash
# q10: q9 setting with parameter-matched LoRA baseline. $1 = lora_pre | lora_post | none, $2 = seed
v=$1; s=$2; m=lora; pl=pre
[ "$v" = "lora_post" ] && pl=post
[ "$v" = "none" ] && m=none
exec ~/venv-baseline/bin/python train.py --mode $m --lora_place $pl --seed $s --K 4 --occ 1 --ret_norm 1 --rename 1 --task query --train_P 2,3 --test_P 2,3,4,6,8,12,16 --steps 8000 --out results/q10_${v}_s$s.json > logs/q10_${v}_s$s.log 2>&1
