# qwenagent
Subagents *inside* Qwen: K persistent-state neural workers injected into a frozen Qwen3 decoder's residual
stream, assigned subproblems by a learned slot router, returning results before the remaining layers.

Research question: at matched params/compute, does separating internal working memory into private worker
states let Qwen compose more concurrent subproblems than seen in training (train 2 interleaved programs,
test 3-4) than one big shared state, a stateless adapter, or frozen Qwen?

Files: `worker.py` module, `task.py` interleaved register programs, `train.py` train+eval+ablations.
Results in `results/`, logs in `logs/`.

## Status log
- q1 (train P=2, query task, 1 seed): adapter 52/29/23 on P=2/3/4, multi 40/36/26, single+none ~chance. See results/q1_*.json.
- q2 (running): 3 modes x 3 seeds, train P in {2,3}, test P 2..6, 6000 steps. results/q2_*.json.
