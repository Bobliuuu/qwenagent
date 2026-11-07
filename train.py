import argparse, json, random, time, torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from worker import Workers
import task

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="multi")          # multi|single|adapter|none
ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
ap.add_argument("--layer", type=int, default=12)
ap.add_argument("--K", type=int, default=4); ap.add_argument("--d_s", type=int, default=64)
ap.add_argument("--task", default="query")
ap.add_argument("--train_P", default="2"); ap.add_argument("--test_P", default="2,3,4")
ap.add_argument("--ops", default="4,10")
ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="results/run.json"); ap.add_argument("--device", default="cuda")
a = ap.parse_args()
random.seed(a.seed); torch.manual_seed(a.seed)
dev = a.device; dt = torch.bfloat16 if dev == "cuda" else torch.float32
tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"
lm = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dt).to(dev).eval()
for p in lm.parameters(): p.requires_grad_(False)
digit_ids = [tok.encode(str(i), add_special_tokens=False)[0] for i in range(10)]
W = None
if a.mode != "none":
    W = Workers(lm.config.hidden_size, a.K, a.d_s, a.mode).to(dev)
    cur = {"mask": None}
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        d = W(h, cur["mask"]).to(h.dtype)
        return (h + d,) + tuple(out[1:]) if isinstance(out, tuple) else h + d
    lm.model.layers[a.layer].register_forward_hook(hook)
print("trainable params", sum(p.numel() for p in W.parameters()) if W else 0, flush=True)

def logits_for(xs):
    enc = tok(xs, return_tensors="pt", padding=True).to(dev)
    if W: cur["mask"] = enc.attention_mask.float()
    lg = lm(**enc).logits[:, -1, :]
    return lg[:, digit_ids].float()

@torch.no_grad()
def evaluate(Ps, n=300, nops=(4, 10), ablate=None, mode=None):
    if W: W.ablate = ablate
    rng = random.Random(1234); c = 0
    for i in range(0, n, 50):
        xs, ys = task.batch(50, [Ps], nops, mode or a.task, rng)
        c += (logits_for(xs).argmax(-1).cpu() == torch.tensor(ys)).sum().item()
    if W: W.ablate = None
    return c / n

ops = tuple(map(int, a.ops.split(","))); trP = [int(x) for x in a.train_P.split(",")]
teP = [int(x) for x in a.test_P.split(",")]
res = {"args": vars(a), "log": []}
if W:
    opt = torch.optim.AdamW(W.parameters(), lr=a.lr, weight_decay=0.01)
    t0 = time.time()
    for step in range(1, a.steps + 1):
        xs, ys = task.batch(a.bs, trP, ops, a.task)
        loss = F.cross_entropy(logits_for(xs), torch.tensor(ys, device=dev))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(W.parameters(), 1.0); opt.step()
        if step % 100 == 0:
            acc = evaluate(trP[0], 100, ops)
            res["log"].append((step, loss.item(), acc)); print(step, round(loss.item(), 3), acc, round(time.time() - t0), flush=True)
final = {}
for P in teP:
    final[f"P{P}"] = evaluate(P, 300, ops)
    final[f"P{P}_long"] = evaluate(P, 300, (ops[1] + 4, ops[1] + 12))
if W and a.mode != "adapter":
    for ab in ["zero", "shuffle", "drop:0"]:
        final[f"P{teP[0]}_{ab}"] = evaluate(teP[0], 300, ops, ablate=ab)
res["final"] = final; print(json.dumps(final, indent=1))
json.dump(res, open(a.out, "w"), indent=1)
