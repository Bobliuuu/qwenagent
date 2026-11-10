# q11 (Track B): do private slots win when state capacity binds?

## Design (written before running)

Question. At P<=16 and d_s=64 a single shared state is not capacity-limited, so `multi` ~ `single` ~ `adapter` is expected. Shrink the per-slot width and push test P to 32/64 so a fixed-size shared state must hold P digits, and see whether slots (state that grows with P) win.

Setup (all cap cells): Qwen3-0.6B frozen, workers at layer 12, K=4 at train, train P in {2,3}, 6000 steps, ops/register 2-4, query task, random distinct 2-letter register names (`--long_names 1`, new flag, so names have the same format at P=2 and P=64; 676 names available), `occ=1`, `ret_norm=1` (the q9 setting), 3 seeds (0,1,2), 200 eval programs per P, test P in {2,3,4,8,16,32,64}. Chance 10%.

**`--match_params 0` for the whole sweep.** The usual flag widens single/adapter to multi's parameter count, which would hide a capacity bottleneck, so widths are set explicitly (new flag `--single_width`):

| variant | state | note |
|---|---|---|
| `multi` / `oracle` / `uniform` | K slots x d_s; at test K_eff = max(K, P), so total state = P*d_s | slots scale with P by construction (that is the claimed advantage) |
| `single_eq` | ONE state, width K*d_s (=4*d_s) | equal total state width to multi *at train* (K=4) |
| `single_ds` | ONE state, width d_s | the shrunken single (capacity-starved control) |
| `single_max` | ONE state, width 64*d_s | equal total width to multi *at P=64*, fixed at train. Only d_s in {4,8,16} (64*64=4096 too big) |
| `adapter` | stateless MLP, hidden 4*K*d_s | stateless reference |

So the fair comparison for "slots win when capacity binds" is multi vs single_max (equal state numbers at the largest P) and vs single_eq/single_ds (equal at train). Caveat: equal-at-test is not equal in parameters (GRU/phi params scale with width^2, so single_max has far more).

Sweep d_s in {4,8,16,64}. Pruned (cost cap ~$30): uniform only at d_s=4; single_max only d_s<=16; adapter only d_s=4; d_s=16 has no single_max in batch A; no multi_rr. Sequences at P=64 are ~1000 tokens; eval batch is 5 for P>16 (code change only affects P>16).

Part 2, ops-per-register wall: `--opr_curriculum "1,1:800;1,2:800;2,4:1200;4,8:1600;8,12:1800"` (up to 8-12 ops per register in training), train P in {2,3,4}, d_s=64, `single` (width 256 = K*d_s) / `multi` / `oracle`, match_params 0, same names/ret_norm setting. Eval via existing `P4_ops{5-6,7-12,13-24}` and P in {2,3,4,6}. Reference without curriculum (q9, 1-letter names): P4 at 5-6 ops ~0.19, 7-12 ~0.11, 13-24 ~0.12, i.e. near chance.

Code changes (additive only): `task.make(long_names=)`, `train.py --long_names --single_width`, eval batch 5 for P>16, `run_q11.sh`, `modal_cap.py` (app qwenagent-cap, volume qwenagent-cap-res).

## Results
(filled in as batches finish)
