"""results/*.json -> figures/*.png. matplotlib+numpy only."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25, "font.size": 10})
FIG = os.path.join(ROOT, "figures"); os.makedirs(FIG, exist_ok=True)

def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(FIG, name), dpi=110); plt.close(fig); print("wrote", name)

def acc_vs_P(pre, res):
    a = info(res); trP = [int(x) for x in a["train_P"].split(",")]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for m in MODES:
        if m not in res: continue
        Ps = pkeys(res[m]); S = [stat(res[m], f"P{p}") for p in Ps]
        mu = np.array([s[0] for s in S]); sd = np.array([s[1] for s in S]); n = S[0][2]
        ax.plot(Ps, mu, "-o", color=COLOR[m], lw=2.4 if m in ("multi", "oracle") else 1.6, ms=4, label=f"{LABEL[m]} (n={n})")
        ax.fill_between(Ps, mu - sd, mu + sd, color=COLOR[m], alpha=.12, lw=0)
        for i, p in enumerate(Ps):   # per-seed dots (n is small, show them)
            v = seed_vals(res[m], f"P{p}"); ax.scatter([p] * len(v), v, s=14, color=COLOR[m], alpha=.55, zorder=3, edgecolors="none")
    ax.axhline(.1, color="k", ls=":", lw=1); ax.text(Ps[-1], .105, "chance 10%", ha="right", va="bottom", fontsize=8)
    ax.axvspan(min(trP) - .4, max(trP) + .4, color="gray", alpha=.12); ax.text(np.mean(trP), .98, "trained", ha="center", fontsize=8, va="top")
    ax.set_xscale("log", base=2) if max(Ps) >= 12 else None
    ax.set_xticks(Ps); ax.set_xticklabels(Ps)
    ax.set_xlabel("number of interleaved registers P at test time"); ax.set_ylabel("accuracy (line = mean, band = ±std, dots = seeds)")
    ax.set_ylim(0, 1); ax.set_title(f"{pre}: task '{a['task']}', trained on P in {{{a['train_P']}}}, tested up to P={max(Ps)}")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1)); save(fig, f"{pre}_accuracy_vs_registers.png")

def curves(pre, res):
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for m in MODES:
        runs = [r for r in res.get(m, []) if r.get("log")]
        if not runs: continue
        L = min(len(r["log"]) for r in runs)
        st = [r["log"][i][0] for r in runs[:1] for i in range(L)]
        for ax, j, nm in ((axs[0], 1, "training loss"), (axs[1], 2, "train-distribution accuracy")):
            M = np.array([[r["log"][i][j] for i in range(L)] for r in runs])
            if j == 1:  # smooth the noisy per-batch loss
                k = 5; M = np.array([np.convolve(np.pad(x, (k // 2, k // 2), mode="edge"), np.ones(k) / k, "valid") for x in M])
            mu = M.mean(0); sd = M.std(0)
            ax.plot(st, mu, color=COLOR[m], lw=1.8, label=m); ax.fill_between(st, mu - sd, mu + sd, color=COLOR[m], alpha=.15, lw=0)
            ax.set_xlabel("step"); ax.set_ylabel(nm)
    axs[1].axhline(.1, color="k", ls=":", lw=1); axs[0].legend(fontsize=8); fig.suptitle(f"{pre}: learning curves (mean ± std, loss smoothed)")
    save(fig, f"{pre}_learning_curves.png")

def ablations(pre, res):
    modes = [m for m in MODES if m in res and abl_prefix(res[m])]
    if not modes: return
    kinds = [("", "intact"), ("zero", "zero return"), ("shuffle", "shuffle across batch"), ("drop:0", "drop slot 0")]
    fig, ax = plt.subplots(figsize=(8, 4)); w = .8 / len(kinds)
    for j, (k, nm) in enumerate(kinds):
        mu, sd = [], []
        for m in modes:
            p = abl_prefix(res[m]); key = (p + k) if k else p.rstrip("_")
            s = stat(res[m], key) or (np.nan, 0, 0); mu.append(s[0]); sd.append(s[1])
        ax.bar(np.arange(len(modes)) + j * w, mu, w, yerr=sd, capsize=2, label=nm, color=["#4a4a4a", "#c0463b", "#d98c1f", "#7a5cb8"][j], alpha=.85)
    ax.axhline(.1, color="k", ls=":", lw=1)
    ax.set_xticks(np.arange(len(modes)) + 1.5 * w); ax.set_xticklabels(modes)
    p0 = abl_prefix(res[modes[0]]).rstrip("_"); ax.set_ylabel(f"accuracy at {p0}")
    ax.set_title(f"{pre}: intervention ablations at {p0} (zeroing the return must hit chance)"); ax.legend(fontsize=8)
    save(fig, f"{pre}_ablations.png")

def params(pre, res):
    modes = [m for m in MODES if m in res and res[m][0].get("params")]
    if not modes: return
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.barh(modes, [res[m][0]["params"] / 1e6 for m in modes], color=[COLOR[m] for m in modes])
    ax.set_xlabel("trainable parameters (millions); Qwen3-0.6B itself is frozen"); ax.set_title(f"{pre}: parameter budgets")
    save(fig, f"{pre}_params.png")

PROBE_KINDS = [("", "base (fixed names)", "#4a4a4a"), ("_renamed", "renamed registers", "#c0463b"),
               ("_shuffled_init", "shuffled init order", "#d98c1f"), ("_more_ops", "more ops / register", "#7a5cb8")]

def probe_bars(pre, probes):
    modes = [m for m in MODES if m in probes]
    if not modes: return
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, P in zip(axs, (2, 3)):
        w = .8 / len(PROBE_KINDS)
        for j, (suf, nm, c) in enumerate(PROBE_KINDS):
            v = [probes[m].get(f"P{P}{suf}", np.nan) for m in modes]
            ax.bar(np.arange(len(modes)) + j * w, v, w, label=nm, color=c, alpha=.88)
        ax.axhline(.1, color="k", ls=":", lw=1); ax.set_xticks(np.arange(len(modes)) + 1.5 * w)
        ax.set_xticklabels([m.replace("_rr", "\n_rr") for m in modes], fontsize=8); ax.set_title(f"P={P} registers (a trained size)")
    axs[0].set_ylabel("accuracy (200 programs, seed-0 checkpoint)"); axs[0].set_ylim(0, 1); axs[0].legend(fontsize=8)
    fig.suptitle(f"{pre} probes: what breaks the models? (one checkpoint per mode, no error bars)")
    save(fig, f"{pre}_probes.png")

def k_at_test(pre, probes):
    modes = [m for m in ("uniform", "multi", "multi_rr", "oracle") if m in probes]
    if not modes: return
    Ps = [2, 3, 4, 6]
    fig, axs = plt.subplots(1, len(Ps), figsize=(13, 3.6), sharey=True)
    for ax, P in zip(axs, Ps):
        for m in modes:
            ks = sorted(int(k.split("_K")[1]) for k in probes[m] if k.startswith(f"P{P}_K"))
            if ks: ax.plot(range(len(ks)), [probes[m][f"P{P}_K{k}"] for k in ks], "-o", color=COLOR[m], label=m, ms=5)
            ax.set_xticks(range(len(ks))); ax.set_xticklabels([f"K={k}" for k in ks], fontsize=8)
        ax.axhline(.1, color="k", ls=":", lw=1); ax.set_title(f"P={P}")
    axs[0].set_ylabel("accuracy"); axs[0].set_ylim(0, 1); axs[-1].legend(fontsize=8)
    fig.suptitle(f"{pre}: slot count K changed at test time (trained with K=4); a flat line means the slots are not being used")
    save(fig, f"{pre}_k_at_test.png")

def purity(pre, res):
    modes = [m for m in ("multi", "multi_rr") if m in res]
    if not modes: return
    fig, ax = plt.subplots(figsize=(6, 3.8)); w = .35
    for j, key in enumerate(("purity_P4", "purity_P8")):
        for i, m in enumerate(modes):
            v = seed_vals(res[m], key); x = i + j * w
            if v: ax.bar(x, np.mean(v), w, color=COLOR[m], alpha=.45 + .35 * j, label=None)
            ax.scatter([x] * len(v), v, color="k", s=16, zorder=3)
    ax.set_xticks([i + w / 2 for i in range(len(modes))]); ax.set_xticklabels(modes)
    ax.set_ylim(0, 1.05); ax.set_ylabel("router purity vs ground-truth register ids")
    ax.set_title(f"{pre}: router purity (light = P4, dark = P8; dots = seeds)", fontsize=9)
    save(fig, f"{pre}_router_purity.png")

if __name__ == "__main__":
    data = load(); hl = headline(data)
    for pre, res in data.items():
        acc_vs_P(pre, res); curves(pre, res); ablations(pre, res); params(pre, res)
        if pre == hl or any(os.path.exists(os.path.join(ROOT, "results", f"probe_{pre}_{m}_s0.json")) for m in MODES):
            pr = load_probes(pre); probe_bars(pre, pr); k_at_test(pre, pr)
        if pre == hl: purity(pre, res)
