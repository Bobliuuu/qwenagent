"""aggregate results/q12_ledger_*.json -> markdown table"""
import json, glob, statistics as st
modes = ["none", "adapter", "single", "uniform", "multi", "oracle"]
keys = ["P2", "P3", "P4", "P6", "P8", "P12", "P4_zero", "purity_P4", "purity_P8"]
print("| mode | n | " + " | ".join(keys) + " |\n|" + "---|" * (len(keys) + 2))
for m in modes:
    rs = [json.load(open(f))["final"] for f in sorted(glob.glob(f"results/q12_ledger_{m}_s*.json"))]
    cells = []
    for k in keys:
        v = [r[k] for r in rs if r.get(k) is not None]
        cells.append("-" if not v else f"{st.mean(v):.2f}" + (f"±{st.pstdev(v):.2f}" if len(v) > 1 and not k.startswith("purity") else ""))
    print(f"| {m} | {len(rs)} | " + " | ".join(cells) + " |")
print("\nper-seed P12 / P8:")
for m in modes:
    print(m, [(round(json.load(open(f))["final"]["P8"], 2), round(json.load(open(f))["final"]["P12"], 2)) for f in sorted(glob.glob(f"results/q12_ledger_{m}_s*.json"))])
