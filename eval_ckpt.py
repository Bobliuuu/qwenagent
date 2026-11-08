"""Eval-only probes on a saved checkpoint (no training). Answers 'what do the workers actually rely on?'
  python eval_ckpt.py ckpt/q5_multi_s0.pt --out results/probe_q5_multi_s0.json
Probes: (a) baseline  (b) renamed registers (random letters)  (c) shuffled init order
        (d) slot count K_eff at test (multi-slot modes)  (e) more ops per register (distractor interference)"""
import sys, json, random, argparse, torch
ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("--out", required=True)
ap.add_argument("--task", default="query"); ap.add_argument("--n", type=int, default=200)
ap.add_argument("--P", default="2,3,4,6,8"); b = ap.parse_args()
ck = torch.load(b.ckpt, map_location="cpu"); sa = ck["args"]
sys.argv = ["train.py", "--mode", sa["mode"], "--K", str(sa["K"]), "--d_s", str(sa["d_s"]), "--task", b.task,
            "--init", b.ckpt, "--steps", "0", "--n_eval", str(b.n), "--out", "/dev/null", "--seed", "123"]
src = open("train.py").read().split("res = {")[0]       # reuse model/hook/eval definitions, skip training
exec(src)
import task as T
@torch.no_grad()
def acc(P, opr_, K=None, scale=1.0, **kw):
    rng = random.Random(5000 + P); c = 0
    for i in range(0, b.n, 25):
        xs, ys, sp = T.batch(25, [P], opr_, b.task, rng, **kw)
        if W: W.ret_scale = scale; W.K_force = K            # None -> default (max(train K, P)); int -> force slot count
        c += (run(xs, sp, Kmin=P).argmax(-1).cpu() == torch.tensor(ys)).sum().item()
    return c / b.n
out = {}
for P in map(int, b.P.split(",")):
    base = (2, 4)
    out[f"P{P}"] = acc(P, base)
    out[f"P{P}_renamed"] = acc(P, base, rename=True)
    out[f"P{P}_shuffled_init"] = acc(P, base, shuffle_init=True)
    out[f"P{P}_more_ops"] = acc(P, (6, 10))
    if P > 3: out[f"P{P}_retscale"] = acc(P, base, scale=2.5 / P)   # undo the sum-over-more-slots magnitude growth
    if W and W.mode in ("multi", "oracle", "uniform", "multi_rr"):
        for K in sorted({max(1, P // 2), P, 2 * P}): out[f"P{P}_K{K}"] = acc(P, base, K=K)
json.dump(out, open(b.out, "w"), indent=1); print(json.dumps(out, indent=1))
