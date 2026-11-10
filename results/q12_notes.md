# q12: realistic interleaved-entity task (ledger)

Task `--task ledger` (task.py `make_ledger`; samples in `results/q12_samples.txt`, test in `test_task_q12.py`):
random first names (2-16 entities), each with a 0-9 balance (no wraparound), natural-language updates
("mira deposits 3 .", "bo spends 2 .") interleaved in one stream, query "how much does mira have ?" -> digit.
Ground-truth entity id per sentence (all tokens of an update sentence carry it; init sentence too; query tokens -1).
Setup = q9 recipe: K=4, occ gate, ret_norm, random names, train P in {2,3}, 2-4 updates per entity, 6000 steps,
A10G/L4 on Modal, n_eval=100 per P (so +-0.05 binomial noise per point). Launcher `run_q12.sh`, `modal_real.py`
(app qwenagent-real, volume qwenagent-real-res). Logs `logs/q12_*.log`, results `results/q12_ledger_*.json`,
table by `q12_table.py`.

## Results (mean +- population std over seeds; test P in entities; train P = 2,3)

| mode | seeds | P2 | P3 | P4 | P6 | P8 | P12 | P4 zero-return | router purity P4 / P8 |
|---|---|---|---|---|---|---|---|---|---|
| none (frozen) | 4 | 0.06 | 0.16 | 0.14 | 0.13 | 0.07 | 0.06 | - | - |
| adapter | 3 | 0.96+-0.01 | 0.97+-0.02 | 0.87+-0.01 | 0.74+-0.09 | 0.54+-0.08 | 0.43+-0.08 | 0.14 | - |
| single | 3 | 0.94+-0.02 | 0.96+-0.01 | 0.89+-0.02 | 0.80+-0.02 | 0.61+-0.04 | 0.53+-0.01 | 0.14 | - |
| uniform | 2 | 0.91+-0.01 | 0.90+-0.01 | 0.85+-0.04 | 0.75+-0.04 | 0.57+-0.01 | 0.51+-0.01 | 0.14 | 1.00* / 1.00* |
| multi | 3 | 0.90+-0.01 | 0.88+-0.04 | 0.82+-0.01 | 0.66+-0.07 | 0.49+-0.06 | 0.39+-0.04 | 0.14 | 0.97 / 0.96 |
| oracle | 2 | 0.89+-0.03 | 0.91+-0.04 | 0.84+-0.02 | 0.70+-0.01 | 0.55+-0.01 | 0.33+-0.05 | 0.14 | 1.00 / 1.00 |

Per-seed P8/P12: adapter (.46/.37, .66/.54, .51/.39), single (.60/.54, .66/.54, .57/.52), uniform (.58/.51, .56/.50),
multi (.52/.44, .40/.34, .55/.40), oracle (.54/.38, .57/.29). `lora` not run (track-a-lora not merged when launched).
Seeds are fewer than the 3-5 asked (budget trim: none 4, adapter/single/multi 3, uniform/oracle 2); the
`files` variant (same generator, commit-counter vocabulary) is implemented and sampled but NOT run.

## Findings

1. The task works as a testbed: frozen `none` is at chance (6-16%, never above the ~15% majority-value rate),
   every trained mode reaches 0.9+ at P2/P3 from scratch. So there is real learnable signal that needs the module.
2. Zero-return intervention: zeroing the injected return drops P4 from 0.82-0.89 to 0.14 for every mode, i.e. exactly
   the frozen baseline. The frozen host is not solving it by itself; the module carries the state computation.
   (Shuffle-across-batch also drops to ~chance.) Dropping slot 0 of multi: P4 0.83 -> 0.71-0.76, so slots hold
   distinct information but not one slot = indispensable.
3. Router purity: learned router (multi) is ~0.96-0.97, near the oracle's 1.00, so on natural-name entities the router
   does recover entity identity. (*Purity for `uniform` is an artefact: all slots get equal weight so argmax is slot 0
   everywhere; ignore it.) Purity is high but does not translate to accuracy gains, same as q9.
4. Slots vs shared state: does NOT change the conclusion. Ranking at P8/P12 is single (.61/.53) >= uniform (.57/.51)
   > adapter (.54/.43) ~ oracle (.55/.33) ~ multi (.49/.39). Even oracle routing (perfect separation) is not better
   than a single shared state at P12 (it is worse, .33 vs .53, beyond seed spread), and learned multi is within seed
   noise of or below the stateless adapter (seeds overlap heavily: adapter P12 .37-.54, multi .34-.44). Private
   slots never beat the shared-state or adapter baseline on this more realistic task either. In-distribution (P2/P3)
   the adapter/single are slightly better than multi (about +4-6 pts, small n).
5. Persistence advantage is weak too: single beats adapter by ~+10 pts at P12 (0.53 vs 0.43), consistent in sign
   but with 3 seeds and one adapter seed at .54 this is suggestive, not established. Length-in-operations
   extrapolation collapses for all modes as in q7 (see `P4_ops*` in the JSONs).

## Honest caveats
- 2-4 seeds, n_eval 100: differences below ~8 points are noise. Conclusions are "no evidence slots help", not "slots hurt".
- Only the ledger task was run; the `files` variant and `lora` were not. Single train distribution (P 2-3).
- Oracle worse than uniform/single at P12 is unexplained (possible slot-index collision r%K with K=4 when P>K,
  as in q7's oracle/uniform break from ~6 entities). Oracle is therefore not a clean upper bound beyond P=4.
