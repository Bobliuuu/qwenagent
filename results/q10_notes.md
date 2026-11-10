# q10: parameter-matched LoRA baseline

Setting (identical to q9): `--K 4 --occ 1 --ret_norm 1 --rename 1 --task query --train_P 2,3 --test_P 2,3,4,6,8,12,16 --steps 8000`, lr 1e-3, AdamW, bs 32, n_eval 200, seeds 0-4. Frozen Qwen3-0.6B.

## How params were matched
Hand-rolled LoRA (no peft) on q/k/v/o of selected decoder layers; delta = 2.0 * x A^T B^T (fp32 params, B zero-init). Rank is a uniform base rank per layer plus +1 on enough layers to land closest to the multi/single budget of 441,281. Result: 440,320 trainable params for both placements (0.2% under).
- `lora_pre`: layers 0-12 (all layers up to and including the injection layer 12), ranks 3/4.
- `lora_post`: layers 13-27 (everything after the injection layer), ranks 2/3.

Zero-return interventions are n/a for LoRA (skipped). `none` has 0 params and no training; the eval RNG is fixed so all 5 seeds are identical up to bf16 noise.

## Results (accuracy, mean±std over seeds; chance = 0.10)
q9 rows are all 5 seeds (q9 adapter/single are complete). q7 rows (earlier setting) are for reference only.

| config | n | params | P2 | P3 | P4 | P6 | P8 | P12 | P16 | mean(P2-16) | final train loss |
|---|---|---|---|---|---|---|---|---|---|---|---|
| q10 lora_pre (L0-12) | 5 | 440320 | 0.114±0.017 | 0.116±0.015 | 0.150±0.021 | 0.120±0.013 | 0.109±0.027 | 0.126±0.011 | 0.122±0.013 | 0.122±0.004 | 2.25 |
| q10 lora_post (L13-27) | 5 | 440320 | 0.560±0.094 | 0.491±0.099 | 0.441±0.053 | 0.438±0.048 | 0.322±0.031 | 0.299±0.028 | 0.204±0.037 | 0.394±0.044 | 1.15 |
| q10 none | 5 | 0 | 0.096±0.005 | 0.133±0.003 | 0.128±0.003 | 0.095±0.000 | 0.058±0.003 | 0.100±0.000 | 0.103±0.003 | 0.102±0.002 | n/a |
| q9 adapter | 5 | 507873 | 0.615±0.024 | 0.580±0.015 | 0.541±0.023 | 0.473±0.018 | 0.393±0.030 | 0.318±0.057 | 0.284±0.039 | 0.458±0.015 | 0.96 |
| q9 single | 5 | 441281 | 0.588±0.019 | 0.580±0.036 | 0.514±0.029 | 0.443±0.036 | 0.387±0.052 | 0.359±0.077 | 0.273±0.048 | 0.449±0.030 | 1.08 |
| q9 multi | 5 | 441281 | 0.554±0.043 | 0.516±0.051 | 0.492±0.035 | 0.392±0.078 | 0.348±0.097 | 0.312±0.131 | 0.256±0.092 | 0.410±0.069 | 1.14 |
| q7 adapter | 5 | 507873 | 0.601±0.019 | 0.569±0.023 | 0.526±0.024 | 0.476±0.026 | 0.417±0.020 | 0.340±0.034 | 0.278±0.022 | 0.458±0.007 | 0.99 |
| q7 single | 5 | 441281 | 0.609±0.024 | 0.576±0.020 | 0.534±0.014 | 0.450±0.016 | 0.383±0.040 | 0.340±0.045 | 0.234±0.034 | 0.447±0.020 | 0.95 |
| q7 multi | 5 | 441281 | 0.513±0.120 | 0.457±0.114 | 0.460±0.116 | 0.359±0.101 | 0.362±0.090 | 0.330±0.085 | 0.263±0.073 | 0.392±0.097 | 1.29 |

## Verdict
- **LoRA after the injection layer (lora_post, 440k params) roughly matches the multi workers and loses to adapter/single.** Mean P2-16 is 0.394±0.044, against q9 multi 0.410±0.069 (within noise), q9 adapter 0.458±0.015 and q9 single 0.449±0.030. The gap to adapter/single is about 0.055-0.065, larger than the seed std. So the workers (multi) do not beat a parameter-matched LoRA here, and the simple stateless adapter / single-state worker beat both.
- **LoRA on layers 0-12 (lora_pre) failed to learn**: training loss stays near 2.25 (ln 10 = 2.30) and accuracy is about 0.12, the same as `none` (0.102). All 5 seeds are at chance, so this is a consistent failure, not a dead seed.
- Caveats: one hyperparameter setting (lr 1e-3, scale 2.0) taken from the workers' recipe and not tuned for LoRA. lora_pre's failure may be an lr/scale issue (gradients through 13 modified layers) rather than a capacity limit. A fair "best LoRA" claim needs an lr sweep, which was not done.
