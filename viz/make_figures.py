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
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for m in MODES:
        if m not in res: continue
        Ps = pkeys(res[m]); S = [stat(res[m], f"P{p}") for p in Ps]
        mu = np.array([s[0] for s in S]); sd = np.array([s[1] for s in S]); n = S[0][2]
        ax.plot(Ps, mu, "-o", color=COLOR[m], lw=2.4 if m in ("multi", "oracle") else 1.6, ms=4, label=f"{LABEL[m]} (n={n})")
        ax.fill_between(Ps, mu - sd, mu + sd, color=COLOR[m], alpha=.15, lw=0)
    ax.axhline(.1, color="k", ls=":", lw=1); ax.text(Ps[-1], .105, "chance 10%", ha="right", va="bottom", fontsize=8)
    ax.axvspan(min(trP) - .4, max(trP) + .4, color="gray", alpha=.12); ax.text(np.mean(trP), .98, "trained", ha="center", fontsize=8, va="top")
    ax.set_xscale("log", base=2) if max(Ps) >= 12 else None
    ax.set_xticks(Ps); ax.set_xticklabels(Ps)
    ax.set_xlabel("number of interleaved registers P at test time"); ax.set_ylabel("accuracy (mean ± std over seeds)")
    ax.set_ylim(0, 1); ax.set_title(f"{pre}: task '{a['task']}', trained on P in {{{a['train_P']}}}, tested up to P={max(Ps)}")
    ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1, .93)); save(fig, f"{pre}_accuracy_vs_registers.png")

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

if __name__ == "__main__":
    for pre, res in load().items():
        acc_vs_P(pre, res); curves(pre, res); ablations(pre, res); params(pre, res)
