#!/usr/bin/env python3
"""Derive README / video numbers from results/*.json -> results/derived/readme_data.json.
Everything is computed from the per-run result files; nothing is typed by hand."""
import glob, json, os, statistics as st

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
PS = ["P2", "P3", "P4", "P6", "P8", "P12", "P16"]


def load(pat):
    out = []
    for f in sorted(glob.glob(os.path.join(R, pat))):
        out.append(json.load(open(f))["final"])
    return out


def mean(xs):
    return sum(xs) / len(xs)


def ms(xs):
    return mean(xs), (st.pstdev(xs) if len(xs) > 1 else 0.0)


d = {}
modes = ["adapter", "single", "uniform", "multi", "oracle"]
curve, tbl, abl = [], [], []
for m in modes:
    runs = load(f"q9_{m}_s[0-9].json")
    row = {"mode": m, "n": len(runs)}
    for p in PS:
        mu, sd = ms([r[p] * 100 for r in runs])
        curve.append({"P": int(p[1:]), "mode": m, "acc": mu})
        row[p] = f"{mu:.1f} ± {sd:.1f}"
    row["mean_all_P"] = mean([mean([r[p] for p in PS]) for r in runs])
    tbl.append(row)
    abl.append({"mode": m, "intact": ms([r["P4"] * 100 for r in runs])[0],
                "zero": ms([r["P4_zero"] * 100 for r in runs])[0]})
    if m == "multi":
        d["purity_P4_mean"] = mean([r["purity_P4"] for r in runs]) * 100
        d["purity_P8_mean"] = mean([r["purity_P8"] for r in runs]) * 100
        d["purity_P4_min"] = min(r["purity_P4"] for r in runs) * 100
d["q9_curve"] = curve
d["q9_table"] = tbl
d["q9_ablation"] = abl
d["chance"] = 10
none = load("q10_none_s[0-9].json")
d["frozen_mean_P2"] = mean([r["P2"] for r in none]) * 100
d["frozen_mean_all"] = mean([mean([r[p] for p in PS]) for r in none]) * 100

# LoRA baselines (q10) versus workers (q9), mean accuracy over P=2..16
lora = load("q10_lora_post_s[0-9].json")
d["lora_post_mean"] = mean([mean([r[p] for p in PS]) for r in lora]) * 100
d["lora_post_n"] = len(lora)
d["q9_means"] = {t["mode"]: t["mean_all_P"] * 100 for t in tbl}

# scale and depth (q13): P=4 and P=16, mean over seeds
scale = []
for model, layer in [("0.6B", 12), ("1.7B", 12), ("4B", 15)]:
    for m in ["adapter", "single", "multi"]:
        pat = f"q9_{m}_s[0-2].json" if model == "0.6B" else f"q13_{model}_L{layer}_{m}_s[0-9].json"
        runs = load(pat)
        for p in ["P4", "P16"]:
            scale.append({"model": f"Qwen3-{model}", "mode": m, "P": p, "acc": mean([r[p] for r in runs]) * 100, "n": len(runs)})
d["scale"] = scale
d["scale_rows"] = [{"model": f"Qwen3-{mo}", "mode": m,
                    "P4": next(s["acc"] for s in scale if s["model"] == f"Qwen3-{mo}" and s["mode"] == m and s["P"] == "P4"),
                    "P16": next(s["acc"] for s in scale if s["model"] == f"Qwen3-{mo}" and s["mode"] == m and s["P"] == "P16"),
                    "n": next(s["n"] for s in scale if s["model"] == f"Qwen3-{mo}" and s["mode"] == m)}
                   for mo in ["0.6B", "1.7B", "4B"] for m in ["adapter", "single", "multi"]]
depth = []
for L in [4, 8, 12, 18, 24]:
    pat = f"q13_0.6B_L{L}_adapter_s[0-9].json" if L != 12 else "q9_adapter_s[0-2].json"
    runs = load(pat)
    depth.append({"layer": f"L{L}", "acc": mean([mean([r[p] for p in PS]) for r in runs]) * 100})
d["depth"] = depth
d["n_result_files"] = len(glob.glob(os.path.join(R, "*.json")))
flat = {}
for r in curve:
    flat[f"acc_{r['mode']}_P{r['P']}"] = r["acc"]
for r in tbl:
    flat[f"mean_{r['mode']}"] = r["mean_all_P"] * 100
for r in abl:
    flat[f"zero_{r['mode']}"] = r["zero"]
for r in scale:
    flat[f"scale_{r['model'].split('-')[1].replace('.', '')}_{r['mode']}_{r['P']}"] = r["acc"]
d["flat"] = flat
d["zero_rows"] = [{"mode": r["mode"], "kind": k, "acc": r[key]} for r in abl for k, key in (("intact", "intact"), ("zeroed", "zero"))]
d["scale_p16"] = [{"model": s["model"], "mode": s["mode"], "acc": s["acc"]} for s in scale if s["P"] == "P16"]
os.makedirs(os.path.join(R, "derived"), exist_ok=True)
json.dump(d, open(os.path.join(R, "derived", "readme_data.json"), "w"), indent=1)
print("ok")
