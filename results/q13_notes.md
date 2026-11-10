# q13: robustness sweep (injection layer and model scale)

Setting for every row: the q9 setting (`--K 4 --occ 1 --ret_norm 1 --rename 1 --task query --train_P 2,3 --test_P 2,3,4,6,8,12,16 --steps 8000`), accuracy % on the query task, mean ± population std over n seeds, chance 10%. Launcher `run_q13.sh`, Modal app `modal_scale.py` (app `qwenagent-scale`, volume `qwenagent-scale-res`), aggregation `collect_q13.py`. Raw results `results/q13_<model>_L<layer>_<mode>_s<seed>.json`, logs `logs/q13_*.log`. Qwen weights run in bf16 everywhere (the train.py default on cuda); worker params stay fp32. Hidden size is read from the model config, so no code change was needed for 1.7B (d=2048) or 4B (d=2560). Injection layer for scale runs: 12 of 28 for 1.7B, 15 of 36 for 4B (about 43% depth, the same fraction as 0.6B layer 12 of 28).

## Reference: 0.6B, layer 12 (q9, seeds 0-2 only, same settings)

| mode | P2 | P3 | P4 | P6 | P8 | P12 | P16 |
|---|---|---|---|---|---|---|---|
| adapter | 62.5 ± 1.1 | 58.0 ± 1.5 | 54.3 ± 2.6 | 46.7 ± 1.7 | 38.0 ± 2.7 | 28.3 ± 3.2 | 26.2 ± 2.8 |
| single | 58.3 ± 1.2 | 55.8 ± 2.3 | 52.8 ± 1.2 | 42.7 ± 2.3 | 36.3 ± 4.7 | 32.0 ± 6.5 | 24.7 ± 3.5 |
| multi | 54.5 ± 4.6 | 49.7 ± 4.4 | 50.0 ± 3.9 | 41.3 ± 7.1 | 35.2 ± 11.2 | 31.8 ± 15.0 | 25.0 ± 10.5 |

## Full q13 table (layers on 0.6B, scale at about 43% depth)

| model | layer | mode | n | P2 | P3 | P4 | P6 | P8 | P12 | P16 | P4 zero-ablate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.6B | 4 | adapter | 3 | 63.5 ± 1.9 | 60.3 ± 1.7 | 47.2 ± 2.7 | 46.0 ± 3.7 | 38.7 ± 3.0 | 32.3 ± 4.1 | 27.8 ± 3.5 | 12.5 ± 0.0 |
| 0.6B | 4 | single | 3 | 64.7 ± 1.9 | 62.2 ± 0.8 | 51.3 ± 2.7 | 46.8 ± 6.4 | 44.2 ± 8.0 | 34.3 ± 9.5 | 32.0 ± 6.8 | 12.7 ± 0.2 |
| 0.6B | 4 | multi | 3 | 58.7 ± 2.3 | 54.2 ± 3.3 | 49.5 ± 1.5 | 38.8 ± 3.8 | 35.2 ± 5.7 | 28.8 ± 9.7 | 23.0 ± 7.0 | 12.7 ± 0.2 |
| 0.6B | 8 | adapter | 3 | 63.2 ± 2.4 | 63.3 ± 0.6 | 38.3 ± 4.4 | 30.5 ± 3.5 | 28.2 ± 5.7 | 25.5 ± 7.4 | 22.8 ± 4.0 | 12.5 ± 0.0 |
| 0.6B | 8 | single | 3 | 49.7 ± 21.0 | 44.5 ± 22.3 | 34.2 ± 14.2 | 25.0 ± 10.4 | 23.8 ± 10.8 | 24.0 ± 11.9 | 16.2 ± 6.0 | 12.5 ± 0.0 |
| 0.6B | 8 | multi | 3 | 59.0 ± 4.6 | 55.7 ± 4.2 | 45.7 ± 1.6 | 32.3 ± 4.5 | 26.7 ± 2.7 | 18.5 ± 3.6 | 16.3 ± 4.0 | 12.8 ± 0.2 |
| 0.6B | 18 | adapter | 3 | 28.3 ± 2.8 | 27.5 ± 3.3 | 25.0 ± 1.1 | 24.3 ± 2.7 | 23.2 ± 3.9 | 20.5 ± 1.2 | 22.5 ± 1.9 | 12.5 ± 0.0 |
| 0.6B | 18 | single | 3 | 26.2 ± 5.6 | 23.2 ± 4.1 | 19.5 ± 2.3 | 19.3 ± 3.1 | 18.8 ± 0.6 | 17.2 ± 3.4 | 20.5 ± 4.5 | 12.5 ± 0.0 |
| 0.6B | 18 | multi | 3 | 24.5 ± 6.4 | 22.0 ± 4.3 | 20.3 ± 1.7 | 19.2 ± 1.7 | 20.7 ± 3.4 | 18.0 ± 4.7 | 17.2 ± 4.8 | 12.5 ± 0.0 |
| 0.6B | 24 | adapter | 3 | 16.0 ± 0.7 | 16.5 ± 3.2 | 18.0 ± 2.5 | 11.8 ± 1.0 | 14.0 ± 2.9 | 11.8 ± 3.3 | 11.7 ± 1.2 | 12.5 ± 0.0 |
| 0.6B | 24 | single | 3 | 13.3 ± 0.2 | 13.2 ± 0.2 | 13.8 ± 2.1 | 14.0 ± 1.1 | 10.8 ± 0.9 | 11.3 ± 2.2 | 11.3 ± 0.5 | 12.7 ± 0.2 |
| 0.6B | 24 | multi | 3 | 14.0 ± 1.4 | 13.0 ± 0.4 | 14.0 ± 0.7 | 14.7 ± 0.5 | 9.7 ± 2.1 | 11.2 ± 1.9 | 11.2 ± 0.2 | 12.5 ± 0.0 |
| 1.7B | 12 | none | 3 | 11.0 ± 0.0 | 12.5 ± 0.0 | 9.0 ± 0.0 | 11.5 ± 0.0 | 8.0 ± 0.0 | 11.5 ± 0.0 | 11.5 ± 0.0 | - |
| 1.7B | 12 | adapter | 2 | 65.2 ± 0.8 | 65.2 ± 0.8 | 57.8 ± 2.2 | 53.8 ± 0.8 | 44.8 ± 1.8 | 40.5 ± 1.0 | 29.0 ± 0.0 | 9.0 ± 0.0 |
| 1.7B | 12 | single | 2 | 65.8 ± 1.8 | 62.5 ± 0.5 | 57.2 ± 0.8 | 53.0 ± 2.5 | 52.0 ± 4.0 | 46.5 ± 3.5 | 36.0 ± 1.5 | 9.0 ± 0.0 |
| 1.7B | 12 | multi | 2 | 61.2 ± 2.8 | 58.0 ± 2.0 | 51.0 ± 3.0 | 44.2 ± 4.2 | 40.2 ± 0.2 | 32.2 ± 3.3 | 26.8 ± 1.3 | 9.0 ± 0.0 |
| 4B | 15 | none | 1 | 10.0 ± 0.0 | 11.5 ± 0.0 | 11.0 ± 0.0 | 12.0 ± 0.0 | 6.5 ± 0.0 | 10.5 ± 0.0 | 9.5 ± 0.0 | - |
| 4B | 15 | adapter | 3 | 82.0 ± 0.4 | 73.8 ± 3.3 | 71.5 ± 1.8 | 75.3 ± 1.9 | 68.3 ± 1.8 | 61.2 ± 4.6 | 56.3 ± 5.0 | 11.0 ± 0.0 |
| 4B | 15 | single | 3 | 81.3 ± 2.4 | 72.8 ± 0.6 | 71.7 ± 0.6 | 75.2 ± 2.1 | 66.7 ± 1.2 | 59.7 ± 2.7 | 46.5 ± 6.9 | 11.0 ± 0.0 |
| 4B | 15 | multi | 3 | 76.7 ± 4.5 | 69.0 ± 3.3 | 67.0 ± 1.5 | 69.7 ± 5.3 | 61.2 ± 4.0 | 46.2 ± 7.0 | 30.2 ± 7.0 | 11.0 ± 0.0 |

`P4 zero-ablate` is accuracy at P=4 with the worker return zeroed at test time (about the no-worker floor, 9-13%), so every trained mode relies on its injected signal.

Seed counts: layer sweep n=3 everywhere. 1.7B: `none` n=3, adapter/single/multi n=2 (seed-2 runs were cancelled for budget; GPU concurrency was shared with other tracks, and one 1.7B multi run took about 2 h). 4B: `none` n=1, adapter/single/multi n=3. No OOMs or crashes; all runs fit on A10G/L4 (0.6B, 1.7B) and L40S/A100 (4B).

Router purity at P=4 (multi only): 0.6B layers 4/8/18/24 range 0.47-1.00 (bimodal: some seeds fully bound, some not); 1.7B 0.97 and 1.00; 4B 0.61, 0.58, 0.99. Even seeds with purity near 1.0 do not beat `single` or `adapter`.

## Verdict

**Layer depth (0.6B).** Injection depth matters far more for absolute accuracy than the choice of mode. Layers 4 and 12 are comparable (P4 about 50%), layer 8 is lower and noisier (single has two weak seeds, std 14-21 points), layer 18 drops to about 20-25% and layer 24 is near the floor (11-18%, barely above the no-worker level), presumably because too few frozen layers remain to read the injected state. The slots-vs-shared verdict does not change at any layer where the workers work at all: at layer 4 `single` (51.3 at P4, 32.0 at P16) >= `adapter` (47.2, 27.8) > `multi` (49.5, 23.0); at layer 8 `adapter` and `multi` are close and `single` is unstable; at 18 and 24 `adapter` is best or tied and `multi` is not better. `multi` never beats both baselines at any layer, and it has the largest seed spread.

**Model scale.** Bigger models make the workers much more effective (adapter P4: 54 / 58 / 72 for 0.6B / 1.7B / 4B; P16 at 4B reaches 56), and the no-worker model is at chance at every size (`none` 9-12%), so the signal comes from the workers. The mode ordering is the same at every scale: `adapter` and `single` are tied within noise at short range (4B P4: 71.5 vs 71.7), `single` leads at long range at 1.7B (P8 52.0 vs 44.8, P16 36.0 vs 29.0), `adapter` leads at 4B P16 (56.3 vs 46.5), and `multi` is behind both at 1.7B and 4B (4B P12 46.2 vs 59.7 / 61.2, P16 30.2 vs 46.5 / 56.3). If anything the multi gap grows at 4B at long range.

**Does the conclusion hold?** Yes, within the range tested. Workers work (large gains over `none`; zeroing the return destroys them), private slots do not beat a shared state or a stateless adapter at any layer where the module works, and nothing at 1.7B or 4B makes slots win. The `--ret_norm` fix holds: there is no collapse at 8-16 registers at 1.7B or 4B, and accuracy declines smoothly.

**Caveats.** Only 2-3 seeds per row, bimodal `multi` seeds, one task family (`query`), training on P in {2,3} with a fixed 8000-step budget for all sizes (larger models may be under-trained), and one layer per scale (about 43% depth) for 1.7B and 4B, so layer sensitivity at larger scale was not swept. The stateless `adapter` matching or beating persistent states, including at 4B P16, again suggests the frozen upper layers do most of the tracking.
