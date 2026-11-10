"""Print the q11 table from results/q11_*_s[0-9].json (mean +- pop-std over seeds; per-seed list for small n).
usage: python q11_table.py [results_dir]"""
import json, glob, sys, re, statistics as st, collections
rd = sys.argv[1] if len(sys.argv) > 1 else "results"
cells = collections.defaultdict(dict)
for f in sorted(glob.glob(f"{rd}/q11_*_s[0-9].json")):
    m = re.search(r"q11_(.+)_s(\d)\.json", f); c, s = m.group(1), int(m.group(2))
    if s == 9: continue   # smoke
    cells[c][s] = json.load(open(f))
def fmt(vals):
    v = [100 * x for x in vals]
    return f"{st.mean(v):.0f}±{st.pstdev(v):.0f}" if len(v) > 1 else f"{v[0]:.0f}"
def block(prefix, keys, title):
    print(f"\n{title}\n| cell | n | params | " + " | ".join(keys) + " |\n|---|---|---|" + "---|" * len(keys))
    for c in sorted(cells):
        if not c.startswith(prefix): continue
        rs = cells[c]; p = list(rs.values())[0]["params"]
        print(f"| {c} | {len(rs)} | {p} | " + " | ".join(fmt([r['final'][k] for r in rs.values() if k in r['final']]) for k in keys) + " |")
block("cap_", ["P2", "P3", "P4", "P8", "P16", "P32", "P64"], "capacity sweep (accuracy %, chance 10)")
block("opr_", ["P2", "P3", "P4", "P6", "P4_ops5-6", "P4_ops7-12", "P4_ops13-24"], "ops-per-register wall (accuracy %, P4 at 5-6 / 7-12 / 13-24 ops)")
print("\nper-seed train-P acc (P2,P3):")
for c in sorted(cells): print(c, {s: (r['final']['P2'], r['final']['P3']) for s, r in cells[c].items()})
