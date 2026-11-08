import argparse, json, random, time, os, torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import worker as wk
import task

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="multi")          # multi|multi_rr|oracle|uniform|single|adapter|none
ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
ap.add_argument("--layer", type=int, default=12)
ap.add_argument("--K", type=int, default=8); ap.add_argument("--d_s", type=int, default=64)
ap.add_argument("--match_params", type=int, default=1)   # single/adapter widened to multi's param count
ap.add_argument("--task", default="subset")
ap.add_argument("--curriculum", default="", help="e.g. query:2500,subset:2500 (overrides --steps/--task for training; final eval on last task + query)")         # query|sum|subset
ap.add_argument("--train_P", default="2,3,4"); ap.add_argument("--test_P", default="2,4,6,8,12,16")
ap.add_argument("--opr", default="2,4", help="ops per register range")
ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n_eval", type=int, default=200)
ap.add_argument("--out", default="results/run.json"); ap.add_argument("--save", default="")
ap.add_argument("--trace", default=""); ap.add_argument("--device", default="cuda")
a = ap.parse_args()
random.seed(a.seed); torch.manual_seed(a.seed)
dev = a.device; dt = torch.bfloat16 if dev == "cuda" else torch.float32
tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"
lm = AutoModelForCausalLM.from_pretrained(a.model, dtype=dt).to(dev).eval()
for p in lm.parameters(): p.requires_grad_(False)
digit_ids = [tok.encode(str(i), add_special_tokens=False)[0] for i in range(10)]
dm = lm.config.hidden_size
W = None
if a.mode != "none":
    sw = None
    if a.match_params and a.mode == "single": sw = wk.matched_single_width(dm, a.K, a.d_s)
    if a.match_params and a.mode == "adapter": sw = wk.matched_adapter_width(dm, a.K, a.d_s)
    W = wk.Workers(dm, a.K, a.d_s, a.mode, single_width=sw).to(dev)
    cur = {"mask": None}
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        d = W(h, cur["mask"]).to(h.dtype)
        return (h + d,) + tuple(out[1:]) if isinstance(out, tuple) else h + d
    lm.model.layers[a.layer].register_forward_hook(hook)
NP = wk.nparams(W) if W else 0
print("mode", a.mode, "trainable params", NP, flush=True)

def reg_tensor(enc, spans):
    """token -> ground-truth register id via char spans (-1 = none)."""
    B, T = enc.input_ids.shape; out = torch.full((B, T), -1, dtype=torch.long)
    for b in range(B):
        for t, (c0, c1) in enumerate(enc.offset_mapping[b].tolist()):
            if c1 > c0:
                for (s0, s1, r) in spans[b]:
                    if s0 <= c0 < s1: out[b, t] = r; break
    return out

def run(xs, spans=None, Kmin=1):
    enc = tok(xs, return_tensors="pt", padding=True, return_offsets_mapping=True)
    if W:
        W.reg = reg_tensor(enc, spans).to(dev) if spans is not None else None
        if W.mode in ("multi", "oracle", "uniform", "multi_rr"): W.K_eff = max(a.K, Kmin)
        cur["mask"] = enc.attention_mask.float().to(dev)
    enc = {k: v.to(dev) for k, v in enc.items() if k != "offset_mapping"}
    return lm(**enc).logits[:, -1, :][:, digit_ids].float()

opr = tuple(map(int, a.opr.split(","))); trP = [int(x) for x in a.train_P.split(",")]
teP = [int(x) for x in a.test_P.split(",")]

@torch.no_grad()
def evaluate(P, n, opr_, ablate=None, tname=None):
    if W: W.ablate = ablate
    rng = random.Random(4321 + P); c = 0; bs = 25
    for i in range(0, n, bs):
        xs, ys, sp = task.batch(bs, [P], opr_, tname or a.task, rng)
        c += (run(xs, sp, Kmin=P).argmax(-1).cpu() == torch.tensor(ys)).sum().item()
    if W: W.ablate = None
    return c / n

@torch.no_grad()
def purity(P, n=50):
    """router purity: per sequence, for each true register, share of its tokens sent to its majority slot"""
    if not W or W.mode in ("single", "adapter"): return None
    rng = random.Random(99 + P); xs, ys, sp = task.batch(n, [P], opr, a.task, rng); run(xs, sp, Kmin=P)
    reg = W.reg.cpu(); asg = W.stats.argmax(-1).cpu(); sc = []
    for b in range(n):
        for r in range(P):
            m = (reg[b] == r)
            if m.sum() >= 2: sc.append(torch.bincount(asg[b][m], minlength=W.K_eff).max().item() / m.sum().item())
    return sum(sc) / max(len(sc), 1)

res = {"args": vars(a), "params": NP, "log": []}
if W:
    opt = torch.optim.AdamW(W.parameters(), lr=a.lr, weight_decay=0.01); t0 = time.time()
    phases = [(p.split(":")[0], int(p.split(":")[1])) for p in a.curriculum.split(",")] if a.curriculum else [(a.task, a.steps)]
    sched = [n for n, c in phases for _ in range(c)]; a.task = phases[-1][0]
    for step in range(1, len(sched) + 1):
        xs, ys, sp = task.batch(a.bs, trP, opr, sched[step - 1])
        loss = F.cross_entropy(run(xs, sp), torch.tensor(ys, device=dev))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(W.parameters(), 1.0); opt.step()
        if step % 200 == 0:
            acc = evaluate(trP[0], 100, opr, tname=sched[step - 1])
            res["log"].append((step, loss.item(), acc)); print(step, round(loss.item(), 3), acc, round(time.time() - t0), flush=True)
    if a.save:
        os.makedirs(os.path.dirname(a.save) or ".", exist_ok=True)
        torch.save({"state": W.state_dict(), "args": vars(a)}, a.save)
final = {}
for P in teP:
    final[f"P{P}"] = evaluate(P, a.n_eval, opr)
if a.curriculum:
    for P in teP: final[f"q_P{P}"] = evaluate(P, a.n_eval, opr, tname="query")
final["purity_P4"] = purity(4); final["purity_P8"] = purity(8)
final["P4_long"] = evaluate(4, a.n_eval, (opr[1] + 3, opr[1] + 8))
if W and a.mode not in ("adapter",):
    for ab in ["zero", "shuffle", "drop:0"]:
        final[f"P4_{ab}"] = evaluate(4, a.n_eval, opr, ablate=ab)
if W and a.mode == "adapter":
    for ab in ["zero", "shuffle"]: final[f"P4_{ab}"] = evaluate(4, a.n_eval, opr, ablate=ab)
res["final"] = final; print(json.dumps(final))
if a.trace and W and a.mode != "adapter":
    rng = random.Random(7); traces = []
    for P in [3, 5]:
        xs, ys, sp = task.batch(1, [P], opr, a.task, rng)
        with torch.no_grad(): lg = run(xs, sp, Kmin=P)
        enc = tok(xs, return_offsets_mapping=True)
        toks = [tok.decode([i]) for i in enc.input_ids[0]]
        traces.append({"text": xs[0], "answer": ys[0], "pred": int(lg.argmax(-1)), "tokens": toks,
                       "reg": reg_tensor(tok(xs, return_tensors="pt", return_offsets_mapping=True), sp)[0].tolist(),
                       "assign": W.stats[0, -len(toks):, :P].cpu().tolist(), "P": P})
    json.dump(traces, open(a.trace, "w"))
json.dump(res, open(a.out, "w"), indent=1)
