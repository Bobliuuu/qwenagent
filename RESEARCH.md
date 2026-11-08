# qwenagent: prior art and novelty analysis

Date: 2026-11-07. Scope: read-only review of `worker.py`, `train.py`, `task.py`, `results/q1_*`, `results/q2_*`, plus a web sweep.
Verification status: links marked (V) had the abstract/summary confirmed through search results this session; (U) = cited from memory or only secondhand, check before citing in anything public.

## Results so far (updated 2026-11-08)

All numbers live in `results/*.json` and the generated tables in `results/TABLE.md`; this section is qualitative on purpose. Sections 0-6 below are the earlier prior-art and planning analysis and predate q3-q7.

| run | what changed | outcome |
|---|---|---|
| q2 | task `query`, K=4, 3 seeds, no occupancy gating, modes adapter/single/multi | all learn above chance; persistent-state modes decay more slowly at 5-6 registers than the adapter, but single and multi are not separable and single beats multi in some seeds |
| q3 | task `subset` (sum of three queried registers), K=8, from scratch | never left chance; results not kept |
| q4 | curriculum `query` -> `subset` | stayed at chance even for the adapter; task abandoned |
| q5 | task `query`, K=8, all rungs, 2 seeds, no occupancy gating | adapter and single learn; slot modes (uniform, multi, multi_rr) under-train near chance; oracle partly learns |
| q6 | occupancy-gated returns, K=4, 8000 steps, all rungs incl. multi_rr, 2 seeds | headline: slot modes learn in at least one seed; frozen Qwen ~chance -> workers well above chance at 2-3 registers; zeroing the return -> chance; no mode beats adapter/single at extrapolating; all near chance by 8-12 registers; seed variance large for multi and multi_rr |
| q6 probes | `eval_ckpt.py` on seed-0 checkpoints | every learned mode collapses when register names are randomised, oracle routing mostly survives; multi is insensitive to the slot count K at test (does not use its slots), oracle needs them; all modes collapse with more ops per register |
| q7 | random register names in train and eval (`--rename 1`) | running at the time of writing; see `results/q7_*.json` and the README q7 section once present |

### The binding finding

The control the earlier analysis asked for (section 4, item 3: renamed registers) was run on the q6 checkpoints. It suggests the fixed-letter task lets every learned module memorise "letter -> behaviour", so q2-q6 never tested whether slots can bind a register *name* to a *state*. The oracle router never reads names, which is why it survives renaming. Caveat: one seed and one checkpoint per mode, no error bars.

### What would make a novel claim

The mechanism itself is RIMs / EntNet / Slot Attention in a frozen-LLM host, so the claim has to be empirical. The strongest honest claim available is:

> A frozen pretrained LM can host trainable recurrent slot modules whose router learns **content-based binding** of entities to private states (random names, no positional or letter shortcut), and this extrapolates to more entities than seen in training in a way a single shared state of equal size, and a stateless adapter, cannot match.

That needs q7-style runs (random names) in which `multi` or `multi_rr` beats `single` and `adapter` at P beyond the trained sizes with non-overlapping intervals over more than two seeds, router purity staying high, and accuracy sensitive to K at test time (the K probe above is the check that the slots are really used). If q7 shows all learned modes at chance with random names, the defensible report is the negative one (C5): embedding works, binding is not learned at this scale, and the oracle shows what correct routing would buy.

## 0. What the code and data actually say

Mechanism: at layer 12 of frozen Qwen3-0.6B a forward hook adds `sum_k phi(LN(s_k))` to the residual. Each slot is a GRU state, written token by token with a gate `a_k * wg(h_t)`, where `a` = softmax(3 * q(h_t) . k(s_k)) (content-addressed, Slot-Attention-like but one-step and with no iterations). Slots share weights, slot identity is only the random initial state `randn*0.5`, and K can be raised at test time (`Kmin=P`).

Facts worth keeping in view when judging novelty:

1. The read side is not routed. The return is an unweighted sum over all slots, so "multi" has no query-conditioned read. Whatever selects the queried register must happen downstream, in frozen attention, from a summed vector. This is a big design gap and also a big confound: the router only affects writes.
2. Persistence is only in the adapter. Frozen Qwen still attends across all tokens in layers 13-28, so "stateless adapter" is not stateless computation. It is a per-token MLP perturbation that frozen attention can use. That an adapter reaches 55-70% at P=2,3 shows Qwen can partly do the task in context once nudged. Persistent state is therefore a modest addition on top of an already-capable host.
3. q2 numbers (3 seeds, query task, train P in {2,3}, 6000 steps, K=4, d_s=64; chance = 0.10):

| mode | P2 | P3 | P4 | P5 | P6 |
|---|---|---|---|---|---|
| adapter (stateless) | .56 | .70 | .47 | .36 | .29 |
| multi (learned router) | .40 | .58 | .48 | .44 | .39 |
| single (param-matched), s0/s1/s2 | .47/.49/.51 | .65/.60/.62 | .41/.58/.55 | .30/.56/.46 | .26/.57/.43 |

   Multi is worse than adapter in-distribution (P2 .40 vs .56) and better only at P5-P6, i.e. it degrades more slowly, partly because it starts lower. Single seed 1 and 2 beat multi at every P>=4. Seed variance of single (P6 .26-.57) is as large as the multi-vs-adapter gap. Verdict: persistence-helps-extrapolation is suggestive (n=3, wide spread), private-vs-shared is not established, and "multi > single" is currently contradicted by 2 of 3 seeds.
4. Ablations in the logs show the module is used (zero-ablation drops to 0.11) but also that multi is only partly slot-specific: dropping slot 0 costs .10-.15, and shuffling across the batch leaves .21-.36, i.e. a good share of the accuracy survives a batch-shuffled return, so the adapter carries sequence-independent signal (task-format priors, digit-marginal bias). Single collapses to ~chance under shuffle (.10-.14), which means single relies more on sequence-specific state than multi does. That is the opposite of what a "slots store registers" story predicts and worth probing.
5. "P_long" (more ops per register) is near 0.15-0.30 for all modes: no mode solves long chains. Accuracy is far from saturation even at P=2, so the project is currently measuring optimization quality of a hard-to-train recurrent adapter, not clean generalization.
6. Router quality has not been measured against the ground-truth register ids (the trace file exists, the metric does not). `oracle` and `uniform` modes exist but have not been run in q2.

## 1. Prior art table

| Work | Link | What it already shows | How we differ |
|---|---|---|---|
| Slot Attention, Locatello et al. 2020 (V) | https://arxiv.org/abs/2006.15055 | Permutation-symmetric slots with competitive attention bind to objects; slots generalise to unseen compositions and (in practice) more objects. | Same core idea (shared-weight slots, random init, softmax competition). We apply it to token streams in a frozen LM, with GRU persistence and a one-shot router. Not a novel mechanism; the novelty can only be the setting. |
| Recurrent Independent Mechanisms (RIMs), Goyal et al. 2019 (V) | https://arxiv.org/abs/1909.10893 | K sparsely-activated recurrent modules with attention-based selection and communication; claims better OOD generalisation when factors change. | Almost exactly our "K private recurrent slots, attention routing" structure. We differ by (a) frozen pretrained host, (b) random per-sequence identity with shared weights (RIMs have per-module params), (c) summed return into a residual. This is the closest prior art; any claim "separation helps generalisation" must be positioned against RIMs. |
| Recurrent Entity Networks (EntNet), Henaff et al. 2016 (V) | https://arxiv.org/abs/1612.03969 | Parallel gated memory cells, one per entity, tracked bAbI world state and generalised in length; content+key addressing. | Same idea for text. Their single-state baseline was an LSTM with more params, so slot-vs-single was not cleanly isolated (noted in the sweep). We add a param-matched single state, but the question is old. |
| Neural Turing Machines / DNC, Graves 2014/2016 (U) | https://arxiv.org/abs/1410.5401 | External addressable read/write memory improves algorithmic length generalisation. | We have no differentiable external memory with separate read/write heads; our "memory" is the per-slot hidden state, with write-only routing. |
| Recurrent Memory Transformer, Bulatov et al. 2022; AAAI 2024 follow-up (V) | https://arxiv.org/abs/2207.06881 | Memory tokens passed segment to segment; works as a wrapper over pretrained transformers. | One flat memory, segment recurrence, memory tokens, usually trainable backbone. We are token-level, multiple private slots, frozen backbone, mid-layer residual injection. |
| Block-Recurrent Transformer, Hutchins et al. 2022 (V) | https://arxiv.org/abs/2203.07852 | A recurrent transformer layer over blocks with a state vector set, long-range gains. | Trained from scratch, one recurrent layer, no entity binding claim. |
| Memorizing Transformers, Wu et al. 2022 (U) | https://arxiv.org/abs/2203.08913 | kNN lookup over past KV; non-differentiable external memory. | Retrieval memory, not learned recurrent state. |
| Gist tokens, Mu et al. 2023 (V) | https://arxiv.org/abs/2304.08467 | Frozen/lightly-tuned LM compresses prompts into learned tokens. | Compression of context, not mutable state tracking. A 2025 study (see ACL 2025 gist study listed in search) reports gist fails on synthetic recall. |
| AutoCompressors, Chevalier et al. 2023 (U) | https://arxiv.org/abs/2305.14788 | Pretrained LM recurrently produces summary vectors passed across segments. | Segment-level summary, no slots, no routing. |
| Ladder Side-Tuning, Sung et al. 2022 (V) | https://arxiv.org/abs/2206.06522 | A small side network reading frozen backbone activations gives parameter- and memory-efficient transfer. | Our module is a side network, but recurrent and injected back mid-network; LST is stateless across tokens. Architecturally our module is "LST with a recurrent cell"; reviewers will say so. |
| LatentMAS, Zou et al. 2025 (V) | https://arxiv.org/abs/2511.20639 | Training-free latent collaboration among separate LLM agents through shared KV working memory; accuracy +14.6% and large token savings. | Between-model, inference-time, text-task agents. Our workers live inside one model's layer, are tiny and trained. The shared "agent" framing is the only overlap; we make no multi-agent task claim. |
| RecursiveMAS (found in search, 2604.25917) (V) | https://arxiv.org/abs/2604.25917 | Lightweight RecursiveLink modules connect agents in latent space. | Same distinction as LatentMAS. |
| Mixture-of-Recursions, Bae et al. 2025 (V) | https://arxiv.org/abs/2507.10524 | Token-level routers choose recursion depth with shared layers. | Routes tokens to depth, not to persistent private states. |
| AsyncThink / "Era of Agentic Organization", Chi et al. 2025 (V) | https://arxiv.org/abs/2510.26658 | One LLM backbone as organizer plus workers with Fork/Join in text, trained by SFT+RL; generalises to unseen tasks. | Text-level fork/join. Ours is non-linguistic and internal. Different level of abstraction. |
| "Latent Agents" (U) | not verified | Mentioned in the brief; no confirming source found in this sweep. | Do not cite until located. |
| Kim and Schuster 2023, entity tracking in LMs (U for details) | https://aclanthology.org/2023.acl-long.213/ | Large LMs track state changes of entities in text only when trained on code/enough scale; sensitive to format. | Diagnoses; does not add memory. Gives a benchmark style (boxes and operations) our register task resembles. |
| Prakash et al. 2024; Feng and Steinhardt 2023 binding IDs; Dai et al. EMNLP 2024 (V via 2409.05448) | https://arxiv.org/abs/2409.05448 | LMs use abstract binding IDs, with an order-encoding low-rank subspace, to bind entities to attributes. | Mechanistic evidence that frozen Qwen already binds by order/ID in-context. Our slots would be an explicit bypass. Tests of "bind by name vs order" below connect directly. |
| Merrill, Petty, Sabharwal 2024, Illusion of State in SSMs (V) | https://arxiv.org/abs/2404.08819 | Fixed-depth parallel models (transformers and SSMs) are in TC0 and cannot do permutation-composition state tracking. | Motivation for state. Note our register tasks are commutative modular arithmetic per register (mod 10 add/sub/mul) with independent registers; mul by non-units is not a group, but each register is a tiny finite automaton and the P registers interleave. Hard state tracking in their sense (S5) is not tested. |
| Liu et al. 2022, Transformers Learn Shortcuts to Automata (V) | https://arxiv.org/abs/2210.10749 | Transformers shortcut automata and fail length generalisation OOD. | Baseline expectation: frozen Qwen fails at long chains (matches our P_long ~chance). |
| Linear-RNN/DeltaNet state tracking line, plus length-generalisation theory (NeurIPS 2025, algebraic structure) (U) | search surfaced titles only | Non-diagonal, input-dependent transitions raise state-tracking power. | Our GRU transition is expressive enough in principle; we do not test expressivity. |

Not found in the sweep: any paper that (a) injects K shared-weight, random-identity recurrent slots into a frozen pretrained decoder layer and (b) compares them to a parameter-matched single state on slot-count extrapolation. Absence in a short sweep is weak evidence; search again before claiming priority (queries to try: "slot memory adapter frozen LLM", "entity-centric recurrent adapter", "RIMs pretrained language model", "persistent state adapter frozen transformer state tracking").

## 2. Open questions (where multi-vs-single could come from)

The reported effect could be due to any of these, and the current modes do not separate them:

1. Persistence: does cross-token state help at all, given Qwen attention already sees context? (adapter vs single answers this; single is sometimes ahead of adapter at P>=4 but with large variance.)
2. Separation / binding: does having K independent states with a shared update function help over one wide state? (single vs oracle; requires the controls in 4.)
3. Routing quality: does the learned router recover register identity, and does accuracy track routing purity? (not measured.)
4. Capacity / width: single is matched on parameters, but a width-W GRU has W^2 recurrent weights while multi shares a d_s=64 cell K times; the capacity of the state itself (K*64 numbers in both cases) is matched but compute is not (multi runs the cell K times per token). Matching state size and FLOPs separately is needed.
5. Optimization: multi is slower to start (q1 multi sat at chance for ~1000 steps, lr 3e-3 diverged), single is high variance; much of the difference may be training dynamics and early stopping at 6000 steps. Curves have not converged (in-distribution accuracy is only 40-70%).
6. Read side: summed unrouted return means "which slot is read" is not modelled. Slot-select read could change the picture.
7. Host shortcut: frozen Qwen might use the return as a soft hint of the answer rather than as a mid-network computation. The batch-shuffle ablation (multi retains .21-.36) suggests a large non-sequence-specific component.

## 3. Candidate claims, ranked

Ranking = (defensibility given cost) x (novelty). Honest note: none is a strong novelty claim; the architecture is RIMs/EntNet/Slot Attention in a new host.

### C1 (best): Slot-count extrapolation of a frozen host with random-identity slots
Hypothesis: with shared-weight slots and random per-sequence identity, the adapter trained on P in {2,3} registers and K=4 extrapolates to P=6..8 by raising K at test, while a single state of equal state size cannot, because its binding capacity is fixed by width and by what it learned in training.
Isolating comparison: multi (K raised at test) vs multi (K fixed at 4, P>K so slots must be shared) vs single with width matched to K_test*d_s (train separately at K_train*d_s; evaluate P beyond) vs oracle routing (upper bound) at P in 2..12.
Confirm: multi with K_test>=P beats single and fixed-K multi at P>K with non-overlapping 5-seed intervals, and routing purity (see E3) stays high at P>K_train. Kill: single matches multi at every P (current data: 2 of 3 seeds do).
Cost: ~6 modes x 5 seeds x ~25 min = 12 GPU-hours; trim to 4 modes x 4 seeds, ~6h on one A100 (6000 steps at bs 32 took roughly this order in earlier runs).
Novelty risk: high. RIMs/Slot Attention/EntNet claim slot-count and object-count generalisation. The only fresh element is "frozen LLM host, adapter-only", so the claim is empirical, not conceptual.

### C2: Persistence helps length-in-registers generalisation over a stateless adapter, in a frozen host
Hypothesis: any recurrent state (single or multi) beats a param-matched stateless adapter at P > train P, because it supplies a mutable scratchpad the frozen model lacks.
Comparison: adapter vs single vs multi, plus a "stateless but with a lookback" control: an adapter that also sees a causal depthwise conv or a small window of previous adapter outputs (so persistence, not just context mixing, is isolated), plus none.
Confirm: gap at P5-P6 > seed spread with n>=5 seeds and the gap vanishes in the windowed control. Kill: windowed adapter equals single/multi.
Cost: ~3h. Novelty risk: medium-high; this is "recurrence helps state tracking", established in general. Defensible as a narrow empirical finding on frozen hosts.

### C3: Routing is the load-bearing component (route purity predicts accuracy)
Hypothesis: multi's advantage tracks how well the learned router recovers register identity; oracle >> learned >> uniform, and uniform == single in effect.
Comparison: oracle vs multi vs uniform vs single, with a router-vs-ground-truth purity metric (token->register assignment accuracy up to slot permutation) computed from existing `s.stats`.
Confirm: monotone ordering and per-sequence correlation of accuracy with purity. Kill: oracle not better than learned (routing is not the bottleneck) or uniform ~ multi (separation unused).
Cost: ~2h (oracle and uniform already implemented, only running them and adding a metric). Novelty: low-medium; gives the diagnostic that the field's slot papers often skip for LM hosts.

### C4: Binding by name vs by order (mechanistic)
Hypothesis: slots bind registers by the name token (a, b, ...), not by arrival order; a frozen host without slots binds mainly by order/position (Dai et al.).
Comparison: at test, rename registers with unseen letters or digits, permute the order of initial assignments, interleave patterns shifted from train; compare multi, single, adapter, none.
Confirm: multi invariant to name permutation and order, adapter degrades. Kill: all modes degrade similarly or router keys on position.
Cost: ~1.5h eval-only on saved checkpoints (needs `--save` re-runs; current q2 runs did not save weights).
Novelty: medium (connects to the binding-ID literature); results are diagnostic.

### C5 (negative-result claim): "Private slots are not needed; a single param-matched state is as good on this task"
Comparison: as q2 with 5+ seeds and larger single width, plus single with multi-head readout.
This is a defensible, honest claim if C1 fails, and should be reported as such. Value: low-novelty but protects against overclaiming.

## 4. Missing controls (add before any claim)

1. Unrouted read is a design confound. Add a `read=routed` variant: return = sum_k a_k(query) * phi(...) (or slot-select at the query token). Without it, "multi" cannot even express "read register c".
2. State-size and compute matching: (a) single with width K*d_s (same state numbers, current), (b) single with K cell evaluations (e.g. K heads with a shared sum, no routing = uniform), (c) multi with params matched via per-slot width changes. Also single with several GRU heads reading from one concatenated state.
3. Shuffled register names at test (letters permuted, unseen letters) and shuffled order of initial assignments: checks binding by name vs position.
4. Slot-count: K_test > K_train and K_test < P (forced sharing); per-slot sparsity (how many slots does the router use?).
5. Interference test: after a register is last written, inject distractor writes to other registers and check whether the queried register's answer is unaffected; compare multi vs single by the number of intervening other-register ops (distance).
6. Cross-slot read at return: ablate the sum with per-slot return gating or give phi access to the mean state, to test whether slots communicate usefully.
7. Router purity metric (token->register agreement up to permutation) and slot-usage entropy; compare to oracle.
8. Seeds: 3 seeds are too few given single's spread (.26-.57 at P6). Use >=5 and report bootstrap intervals; use paired seeds (same data order) for modes.
9. Training length: curves are still rising for several runs. Equalise by converged training or fixed budget with early stopping on a held-out in-distribution P, and report in-distribution accuracy so extrapolation is not confounded with a lower starting point (multi starts at .40 vs adapter .56).
10. Easier or harder task variants: the task is not saturated even in-distribution; consider the `sum`/`subset` modes, and a task with explicit group structure (cyclic add only) so errors are cleanly attributable to state tracking.
11. Layer and injection-site sweep (layer 12 only so far), and a "frozen random router" control to check that learning routes matters.
12. Normalise accuracy by in-distribution accuracy (relative retention P_test / P_train) and report both.

## 5. Skeptical flags

- "Private slots with a learned router" = RIMs/EntNet/Slot Attention. Do not claim the mechanism.
- "Persistent state helps beyond a stateless adapter" is weakly supported: single seeds range wide, adapter is competitive at P<=4, and the multi advantage is largely lower decay from a lower start.
- "Multi > single" is not supported; two of three single seeds exceed multi at P>=4.
- The frozen host does most of the arithmetic; the adapter is a nudge. Interpreting results as "agents inside Qwen" or "workers solving subproblems" overstates what is shown: no worker computes a sub-answer, the router is write-only, and the return is a sum.
- The framing with LatentMAS/AsyncThink is rhetorical only; neither tests state tracking.

## 6. Recommended next 3 experiments

1. Controls-first rerun of q2 with 5 paired seeds and saved checkpoints: modes multi, oracle, uniform, single (state-size matched), single with K heads, adapter, adapter-with-window, plus routed-read variant of multi. Report per-P accuracy, retention ratio, and router purity. About 8 GPU-hours on one A100 (cut seeds to 4 if needed). Decides whether separation, routing, or read gating matters at all.
2. Slot-count and name-binding eval on the saved checkpoints: K_test in {P, 2P, 4} at P in 2..12; rename and reorder tests; interference with distractor distance. Eval-only, about 1.5 hours.
3. Train-longer/convergence control: 20000 steps, two modes (multi-routed-read, best single), with and without curriculum on P, to see whether the gap is an optimisation-speed artifact. About 3 hours.

Decision rule: if experiment 1 shows oracle ~ uniform ~ single within seed spread, report C5 (shared state suffices) and stop claiming private-slot benefit. If oracle >> learned >= single, pursue C1/C3 with routed read as the headline.


### Final q7 verdict (5 seeds each for adapter / single / multi)
Random register names in train and eval make tracking generalise from 2-3 to 16 registers for all three, but `multi` (private slots, learned routing) does not beat the stateless adapter or the single shared state: paired by seed at 12 and 16 registers it wins 2-3 of 5 with mean differences of about +-3 points, and its seed variance is several times larger. The 2-seed edge disappeared with 5 seeds. The defensible empirical claims are (1) stateful worker modules embedded in a frozen LLM work, and (2) name randomisation, not slot separation, drives extrapolation in register tracking. Open: why oracle/uniform break from ~6 registers; the `subset` combination task; longer programs (more ops per register) collapse every mode.
