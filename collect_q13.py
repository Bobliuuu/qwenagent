"""Aggregate results/q13_*.json into markdown tables (mean +- pop-std over seeds, accuracy %)."""
import json, glob, re, collections, statistics as st
R = collections.defaultdict(list)
for f in sorted(glob.glob("results/q13_*.json")):
    m = re.match(r"results/q13_(.+?)_L(\d+)_(\w+?)_s(\d+)\.json", f)
    model, L, mode, s = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
    R[(model, L, mode)].append((s, json.load(open(f))["final"]))
cols = ["P2", "P3", "P4", "P6", "P8", "P12", "P16"]
print("| model | layer | mode | n | " + " | ".join(cols) + " | P4 zero-ablate |\n|" + "---|" * (len(cols) + 5))
for (model, L, mode), rs in sorted(R.items(), key=lambda kv: (kv[0][0], kv[0][1], ["none", "adapter", "single", "multi"].index(kv[0][2]))):
    cells = []
    for c in cols + ["P4_zero"]:
        v = [100 * r[c] for _, r in rs if r.get(c) is not None]
        cells.append(f"{st.mean(v):.1f} ± {st.pstdev(v):.1f}" if v else "-")
    print(f"| {model} | {L} | {mode} | {len(rs)} | " + " | ".join(cells) + " |")
