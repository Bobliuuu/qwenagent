"""traces/q3_<mode>_s0.json -> figures/gif_q3_<mode>_P<P>.gif (router assignment streaming over tokens).
usage: make_gifs.py [--trace FILE --out DIR]   (default: all traces/q3_*_s0.json -> figures/)"""
import os, sys, json, glob, argparse, io
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from common import ROOT, COLOR

PAL = plt.get_cmap("tab20")
ORDER = list(range(0, 20, 2)) + list(range(1, 20, 2))
def rcol(r): return "#9a9a9a" if r < 0 else PAL(ORDER[r % 20])
BG, FG = "#fbfbfd", "#1c1c28"

def render(tr, title, path, wrap=64, dur=110):
    toks, reg, A, P = tr["tokens"], np.array(tr["reg"]), np.array(tr["assign"]), tr["P"]
    T = len(toks); P = A.shape[1]
    # ground-truth ownership accumulator: mass slot k puts on tokens of register r
    fig = plt.figure(figsize=(8.4, 5.2), dpi=72, facecolor=BG)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.35, 1.7, 1.2], width_ratios=[3.2, 1], hspace=.55, wspace=.12,
                          left=.08, right=.97, top=.88, bottom=.09)
    axt = fig.add_subplot(gs[0, :]); axh = fig.add_subplot(gs[1, :]); axb = fig.add_subplot(gs[2, 0]); axo = fig.add_subplot(gs[2, 1])
    for ax in (axt, axh, axb, axo): ax.set_facecolor(BG)
    fig.text(.08, .955, title, fontsize=12, weight="bold", color=FG, va="center")
    if tr.get("_synthetic"):
        fig.text(.97, .915, "SYNTHETIC FIXTURE (not a real trace)", fontsize=9, color="#c0463b", ha="right", va="center", weight="bold")
    # token layout
    axt.set_xlim(0, wrap + 1); axt.axis("off"); lines = [0]; pos = []; x = 0; ln = 0
    for t in toks:
        s = t.strip() or t
        if x + len(s) + 1 > wrap: ln += 1; x = 0
        pos.append((x, ln)); x += len(s) + 1
    nl = ln + 1; axt.set_ylim(nl + .6, -.6)
    texts = []
    for (x, ln), t, r in zip(pos, toks, reg):
        texts.append(axt.text(x, ln, t.strip() or t, family="monospace", fontsize=10.5, color=rcol(r), va="center", alpha=0, weight="bold"))
    cur = Rectangle((0, 0), 1, .8, fill=False, ec=FG, lw=1.5); axt.add_patch(cur)
    verdict = axt.text(wrap, nl - .1, "", ha="right", va="center", fontsize=11, weight="bold", family="monospace")
    # heatmap
    M = np.full((P, T), np.nan); im = axh.imshow(M, aspect="auto", cmap="Greys", vmin=0, vmax=max(.6, A.max()), interpolation="nearest")
    axh.set_yticks(range(P)); axh.set_yticklabels([f"slot {k}" for k in range(P)], fontsize=8)
    axh.set_xlabel("token position (router weight of each slot; strip on top = ground-truth register of the token)", fontsize=8)
    axh.set_xlim(-.5, T - .5)
    for t in range(T): axh.add_patch(Rectangle((t - .5, -1.25), 1, .7, color=rcol(reg[t]), clip_on=False, lw=0))
    own = [axh.add_patch(Rectangle((-3.2, k - .5), 2.2, 1, color="#dddddd", clip_on=False, lw=0)) for k in range(P)]
    axh.text(-1.9, -.95, "owner", fontsize=7, ha="center", va="center", color=FG)
    for s in axh.spines.values(): s.set_visible(False)
    # bars
    bars = axb.bar(range(P), np.zeros(P), color="#cccccc"); axb.set_ylim(0, 1); axb.set_xticks(range(P)); axb.set_xticklabels([f"s{k}" for k in range(P)], fontsize=8)
    axb.set_ylabel("router weight now", fontsize=8); axb.tick_params(labelsize=8)
    for s in ("top", "right"): axb.spines[s].set_visible(False)
    axo.axis("off"); axo.set_xlim(0, 1); axo.set_ylim(0, 1)
    info = axo.text(0, .95, "", va="top", fontsize=9, color=FG, family="monospace")
    O = np.zeros((P, max(reg.max() + 1, 1)))
    frames = []
    for t in range(T):
        M[:, t] = A[t]; im.set_data(M)
        texts[t].set_alpha(1); x, ln = pos[t]; cur.set_xy((x - .25, ln - .4)); cur.set_width(len(toks[t].strip() or toks[t]) + .5)
        if reg[t] >= 0: O[:, reg[t]] += A[t]
        owner = O.argmax(1) if O.sum() > 0 else np.zeros(P, int)
        for k in range(P):
            c = rcol(owner[k]) if O[k].sum() > 0 else "#dddddd"; own[k].set_color(c); bars[k].set_color(c); bars[k].set_height(A[t, k])
        # purity: fraction of each register's mass captured by its best slot
        reg_best = [O[:, r].max() / O[:, r].sum() for r in range(O.shape[1]) if O[:, r].sum() > 0]
        info.set_text(f"token {t+1}/{T}\nregister mass held by\nits best slot: {100*np.mean(reg_best):.0f}%\n(uniform = {100/P:.0f}%)" if reg_best else f"token {t+1}/{T}")
        if t == T - 1:
            ok = tr["pred"] == tr["answer"]; verdict.set_text(f"pred {tr['pred']}  answer {tr['answer']}  {'CORRECT' if ok else 'wrong'}")
            verdict.set_color("#1a9a5a" if ok else "#c0463b")
        fig.canvas.draw(); frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
    plt.close(fig)
    frames += [frames[-1]] * 14
    q = [f.quantize(colors=48, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f in frames]
    q[0].save(path, save_all=True, append_images=q[1:], duration=[dur] * (T) + [140] * 14, loop=0, optimize=True, disposal=1)
    print("wrote", path, f"{os.path.getsize(path)/1e6:.2f} MB", f"{T} tokens")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--trace"); ap.add_argument("--out", default=os.path.join(ROOT, "figures")); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = [a.trace] if a.trace else sorted(glob.glob(os.path.join(ROOT, "traces", "q3_*_s0.json")))
    if not files: print("no traces found (traces/q3_*_s0.json); nothing to render"); return
    for f in files:
        base = os.path.splitext(os.path.basename(f))[0]
        for tr in json.load(open(f)):
            mode = base.split("_")[1] if base.startswith("q3_") else "synthetic"
            render(tr, f"{mode}: router slot assignment over a P={tr['P']} program", os.path.join(a.out, f"gif_{base}_P{tr['P']}.gif"))
if __name__ == "__main__": main()
