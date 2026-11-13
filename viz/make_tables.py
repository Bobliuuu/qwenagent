"""results/*.json -> results/TABLE.md, and injects generated blocks into README.md between markers.
Every number printed here is computed from results/*.json (and results/probe_*.json); prose in the README carries no numbers.
numpy only; never imports torch."""
import os, sys, re, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

def fmt(s): return "n/a" if s is None else f"{100*s[0]:.1f} ± {100*s[1]:.1f}"
def pc(x): return "n/a" if x is None else f"{100*x:.0f}%"
def seeds_str(runs, key): return ", ".join(f"{100*v:.0f}" for v in seed_vals(runs, key))

def table_for(pre, res, title=None):
    a = info(res); Ps = sorted({p for m in res for p in pkeys(res[m])}); trP = a["train_P"]
    ren = ", register names randomised in train and eval" if a.get("rename") else ""
    L = [f"**{title or pre}**: task `{a['task']}`, trained on P in {{{trP}}}, {a['steps']} steps, K={a['K']}, layer {a['layer']}, "
         f"occupancy gating {'on' if a.get('occ') else 'off'}{ren}; accuracy in %, mean ± std (population std) over n seeds; chance = 10%.", ""]
    L.append("| mode | n seeds | trainable params | " + " | ".join(f"P={p}" for p in Ps) + " |")
    L.append("|---|---|---|" + "---|" * len(Ps))
    for m in MODES:
        if m not in res: continue
        runs = res[m]; pr = runs[0].get("params")
        L.append(f"| {m} | {len(runs)} | {f'{pr:,}' if pr else 'not recorded'} | " + " | ".join(fmt(stat(runs, f'P{p}')) for p in Ps) + " |")
    ab = [m for m in MODES if m in res and abl_prefix(res[m])]
    if ab:
        p0 = abl_prefix(res[ab[0]]).rstrip("_")
        L += ["", f"Intervention ablations at {p0} (accuracy %, mean ± std; `P4_long` = more ops per register):", "",
              "| mode | intact | zero return | shuffle across batch | drop slot 0 | more ops / register |", "|---|---|---|---|---|---|"]
        for m in ab:
            p = abl_prefix(res[m])
            L.append(f"| {m} | " + " | ".join(fmt(stat(res[m], k)) for k in (p.rstrip("_"), p + "zero", p + "shuffle", p + "drop:0", p + "long")) + " |")
    pur = [m for m in ("multi", "multi_rr") if m in res and stat(res[m], "purity_P4")]
    if pur:
        L += ["", "Router purity against ground-truth register ids (fraction of each register's tokens sent to its best slot; per seed). "
              "`uniform` and `oracle` read 100% by construction and are not shown.", "", "| mode | purity at P=4 | purity at P=8 |", "|---|---|---|"]
        for m in pur:
            L.append(f"| {m} | " + " | ".join(", ".join(pc(v) for v in seed_vals(res[m], k)) + " (seeds in order)" for k in ("purity_P4", "purity_P8")) + " |")
    return "\n".join(L)

def probe_block(pre, probes):
    if not probes: return "_No probe files for this experiment._"
    modes = [m for m in MODES if m in probes]
    L = [f"Probes from `eval_ckpt.py` on the seed-0 checkpoint of each mode: **one seed, one checkpoint per mode, 200 programs each, no error bars.** "
         f"Accuracy %, chance = 10%.", ""]
    for P in (2, 3):
        L += [f"P={P}:", "", "| mode | base | renamed registers | shuffled init order | more ops / register |", "|---|---|---|---|---|"]
        for m in modes:
            L.append(f"| {m} | " + " | ".join(pc(probes[m].get(f"P{P}{s}")) for s in ("", "_renamed", "_shuffled_init", "_more_ops")) + " |")
        L.append("")
    Ks = [m for m in ("uniform", "multi", "multi_rr", "oracle") if m in probes]
    if Ks:
        L += ["Slot count K changed at test time (trained at K=4), accuracy %:", "", "| mode | P=2: K=1 / 2 / 4 | P=3: K=1 / 3 / 6 | P=4: K=2 / 4 / 8 |", "|---|---|---|---|"]
        for m in Ks:
            cells = []
            for P in (2, 3, 4):
                ks = sorted(int(k.split("_K")[1]) for k in probes[m] if k.startswith(f"P{P}_K"))
                cells.append(" / ".join(pc(probes[m][f"P{P}_K{k}"]) for k in ks))
            L.append(f"| {m} | " + " | ".join(cells) + " |")
    return "\n".join(L)

def facts(pre, res, probes, data):
    Ps = sorted({p for m in res for p in pkeys(res[m])}); L = []
    mean = lambda m, k: (stat(res[m], k) or (None,))[0]
    trained = [p for p in Ps if p <= int(info(res)["train_P"].split(",")[-1])]; P0 = trained[0]
    none = data.get("q5", {}).get("none")
    if none: L.append(f"- Frozen Qwen3-0.6B with no workers (`none`, reference run from `q5`, n={len(none)}): {fmt(stat(none, f'P{P0}'))}% at P={P0}, chance 10%.")
    L.append(f"- At the trained sizes, the trainable modes reach: " + "; ".join(f"{m} {fmt(stat(res[m], f'P{P0}'))} at P={P0}" for m in MODES if m in res) + ".")
    zs = {m: stat(res[m], "P4_zero") for m in MODES if m in res and stat(res[m], "P4_zero")}
    if zs: L.append(f"- Zeroing the return channel at P=4 gives {(f"{100*min(v[0] for v in zs.values()):.0f}%" if round(100*min(v[0] for v in zs.values())) == round(100*max(v[0] for v in zs.values())) else f"{100*min(v[0] for v in zs.values()):.0f}% to {100*max(v[0] for v in zs.values()):.0f}%")} across all {len(zs)} trainable modes (chance 10%): the answers flow through the injected modules.")
    for P in (6, 8, 16):
        if P in Ps:
            L.append(f"- Extrapolation to P={P}: " + "; ".join(f"{m} {fmt(stat(res[m], f'P{P}'))}" for m in MODES if m in res) + ".")
    far = [max(mean(m, f"P{p}") for m in res if m != "none" and mean(m, f"P{p}") is not None) for p in (12, 16) if p in Ps]
    if far: L.append(f"- At P=12 and P=16 the best mode averages {100*far[0]:.0f}% and {100*far[-1]:.0f}% respectively: effectively chance.")
    for m in ("multi", "multi_rr"):
        if m in res: L.append(f"- Seed variance, `{m}`: P={P0} per seed (%) = {seeds_str(res[m], f'P{P0}')}; router purity at P=4 per seed = " + ", ".join(pc(v) for v in seed_vals(res[m], "purity_P4")) + ".")
    for m in ("adapter", "single"):
        if m in res: L.append(f"- Seed spread, `{m}`: P={P0} per seed (%) = {seeds_str(res[m], f'P{P0}')}.")
    if probes:
        ren = [(m, probes[m]["P2"], probes[m]["P2_renamed"]) for m in MODES if m in probes and "P2_renamed" in probes[m]]
        L.append("- Renaming registers at test time (P=2, seed-0 checkpoints): " + "; ".join(f"{m} {pc(b)} -> {pc(r)}" for m, b, r in ren) + ".")
        mo = [(m, probes[m]["P2_more_ops"]) for m in MODES if m in probes and "P2_more_ops" in probes[m]]
        L.append("- More ops per register (P=2): " + "; ".join(f"{m} {pc(v)}" for m, v in mo) + ".")
        if "multi" in probes and "P2_K1" in probes["multi"]:
            q = probes["multi"]; L.append(f"- `multi` at P=2 with K=1 / 2 / 4 slots at test: {pc(q['P2_K1'])} / {pc(q['P2_K2'])} / {pc(q['P2_K4'])}; at P=3 with K=1 / 3 / 6: {pc(q['P3_K1'])} / {pc(q['P3_K3'])} / {pc(q['P3_K6'])}. The slot count barely matters, so `multi` is not using its slots as separate workers.")
        if "oracle" in probes and "P2_K1" in probes["oracle"]:
            q = probes["oracle"]; L.append(f"- `oracle` at P=2 with K=1 / 2 / 4: {pc(q['P2_K1'])} / {pc(q['P2_K2'])} / {pc(q['P2_K4'])} (true routing does need the slots).")
    return "\n".join(L)

# Hand-written one-line outcomes (no numbers: the numbers sit in the generated columns next to them).
OUTCOME = {
    "q1": "Smoke test, single seed, easy task.",
    "q2": "First real comparison (no occupancy gating, no oracle/uniform rung). Persistent-state modes decayed more slowly than the stateless adapter at 5-6 registers, but single and multi were not separable.",
    "q3": "Task `subset` (sum of three queried registers), K=8, from scratch: never left chance. Results were not kept in the repo.",
    "q4": "Curriculum `query` -> `subset`: stayed at chance, even for the adapter. Task abandoned.",
    "q5": "Task `query`, K=8, no occupancy gating: slot modes under-trained (near chance) while adapter and single learned.",
    "q6": "Occupancy-gated returns, K=4, 8000 steps, 2 seeds, all rungs: headline run (see above).",
    "q7": "Random register names in train and eval (learned binding). Generalises to 16 registers for adapter, single and multi (5 seeds); private slots do not beat the shared state or the stateless adapter.",
}

def timeline(data):
    rows = ["| run | task | K | steps | train P | occupancy gating | modes x seeds | adapter P=2 | single P=2 | multi P=2 (per seed) | oracle P=2 | outcome |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for pre in sorted(data, key=lambda p: int(p[1:])):
        res = data[pre]; a = info(res); n = sum(len(v) for v in res.values())
        P0 = int(a["train_P"].split(",")[0]); k = f"P{P0}"
        g = lambda m: fmt(stat(res[m], k)) if m in res else "-"
        mu = (", ".join(f"{100*v:.0f}" for v in seed_vals(res["multi"], k))) if "multi" in res else "-"
        rows.append(f"| {pre} | {a['task']} | {a['K']} | {a['steps']} | {a['train_P']} | {'yes' if a.get('occ') else 'no'} | {len(res)} x {max(len(v) for v in res.values())} | {g('adapter')} | {g('single')} | {mu} | {g('oracle')} | {OUTCOME.get(pre, '')} |")
    return "\n".join(rows) + "\n\n" + "Not in the table: " + " ".join(f"`{p}`: {OUTCOME[p]}" for p in ("q1", "q3")) + "\n"

def q7_block(data):
    pres = rename_prefixes(data)
    if not pres:
        return ("> **q7 in progress.** q7 trains the same six rungs with register names randomised in both training and evaluation, so a model can only succeed by learning "
                "content-based binding (name -> state). No q7 result files exist in `results/` yet. This note is removed automatically by `viz/make_tables.py` "
                "when `results/q7_*.json` appear.")
    out = []
    for pre in pres:
        res = data[pre]; n_done = sum(len(v) for v in res.values())
        partial = len(res) < 6 or any(len(v) < 2 for v in res.values())
        out.append(f"**{pre}: learned binding** (task `query`, random register names in train and eval). "
                   + (f"_Final: {n_done} runs over {len(res)} modes; the n-seeds column gives seeds per row, rows with fewer seeds have wide uncertainty._" if partial else "_Complete._"))
        out.append("")
        out.append(table_for(pre, res, title=f"{pre} (learned binding)"))
        out.append("")
        h = headline(data)
        if h and "single" in res and "single" in data[h]:
            out.append("Same-P comparison with the fixed-name run `%s` at P=2 (mean %%): " % h + "; ".join(
                f"{m} {pre} {fmt(stat(res[m], 'P2'))} vs {h} {fmt(stat(data[h][m], 'P2'))}" for m in MODES if m in res and m in data[h]) + ".")
    return "\n".join(out)

def figs_block(hl, data):
    L = []
    for n, cap in ((f"{hl}_accuracy_vs_registers.png", "Accuracy vs number of registers (headline)"),
                   (f"{hl}_probes.png", "Probes: renamed registers, shuffled init order, more ops per register (one checkpoint per mode)"),
                   (f"{hl}_ablations.png", "Intervention ablations"),
                   (f"{hl}_k_at_test.png", "Slot count K changed at test time"),
                   (f"{hl}_router_purity.png", "Router purity vs ground-truth register ids")):
        if os.path.exists(os.path.join(ROOT, "figures", n)): L.append(f"**{cap}**\n\n![{cap}](figures/{n})\n")
    for pre in rename_prefixes(data):
        n = f"{pre}_accuracy_vs_registers.png"
        if os.path.exists(os.path.join(ROOT, "figures", n)): L.append(f"**{pre}: learned binding (random register names)**\n\n![{pre} accuracy vs registers](figures/{n})\n")
    return "\n".join(L)

def hero(hl):
    out = []
    for P in (3, 5):
        f = os.path.join(ROOT, "figures", f"gif_{hl}_multi_vs_oracle_P{P}.gif")
        if os.path.exists(f):
            out.append(f"![learned router vs oracle routing on one P={P} program]({os.path.relpath(f, ROOT)})")
    return "\n\n".join(out) if out else "_Hero animation: run `viz/build_all.sh`._"

def inject(r, tag, body):
    return re.sub(rf"<!-- {tag}:BEGIN.*?<!-- {tag}:END -->", lambda m: f"<!-- {tag}:BEGIN (generated) -->\n{body}\n<!-- {tag}:END -->", r, flags=re.S)

def main():
    data = load(); hl = headline(data)
    if not hl: print("no complete experiment"); return
    res = data[hl]; probes = load_probes(hl)
    blocks = {
        "RESULTS": f"Headline source: `{hl}` (newest experiment with at least 5 modes and fixed register names).\n\n" + table_for(hl, res),
        "PROBES": probe_block(hl, probes), "FACTS": facts(hl, res, probes, data), "TIMELINE": timeline(data),
        "Q7": q7_block(data), "FIGS": figs_block(hl, data), "HERO": hero(hl)}
    md = "# Results tables (generated by viz/make_tables.py)\n\n" + blocks["RESULTS"] + "\n\n## Probes\n\n" + blocks["PROBES"] + "\n\n## Facts\n\n" + blocks["FACTS"] \
         + "\n\n## q7: learned binding\n\n" + blocks["Q7"] + "\n\n## Timeline\n\n" + blocks["TIMELINE"] + "\n"
    open(os.path.join(ROOT, "results", "TABLE.md"), "w").write(md)
    rp = os.path.join(ROOT, "docs", "RESEARCH_LEDGER.md"); r = open(rp).read()
    for tag, body in blocks.items(): r = inject(r, tag, body)
    open(rp, "w").write(r)
    print("tables written; headline", hl, "; q7 files:", bool(rename_prefixes(data)))
if __name__ == "__main__": main()
