"""Shared loading/aggregation. numpy only; never imports torch."""
import json, glob, os, re
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODES = ["none", "adapter", "single", "uniform", "multi", "multi_rr", "oracle"]
LABEL = {"none": "frozen Qwen (none)", "adapter": "adapter (stateless)", "single": "single (1 shared state)",
         "uniform": "uniform (private, no routing)", "multi": "multi (learned routing)", "multi_rr": "multi_rr (routing + routed read)", "oracle": "oracle (true routing)"}
COLOR = {"none": "#8a8a8a", "adapter": "#d98c1f", "single": "#c0463b", "uniform": "#7a5cb8",
         "multi": "#1f7ae0", "multi_rr": "#17a2b8", "oracle": "#1a9a5a"}
PAT = re.compile(r"^(q\d+)_([a-z_]+?)_s(\d+)\.json$")

def load():
    """-> {prefix: {mode: [result dict,...]}} (only plain <prefix>_<mode>_s<seed>.json; tagged variants skipped)."""
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "*.json"))):
        m = PAT.match(os.path.basename(f))
        if not m or m.group(2) not in MODES: continue
        d = json.load(open(f))
        if "final" not in d: continue
        d["_seed"] = int(m.group(3))
        out.setdefault(m.group(1), {}).setdefault(m.group(2), []).append(d)
    return out

def pkeys(runs):
    ks = set()
    for r in runs:
        for k in r["final"]:
            if re.fullmatch(r"P\d+", k): ks.add(int(k[1:]))
    return sorted(ks)

def stat(runs, key):
    v = [r["final"][key] for r in runs if r["final"].get(key) is not None]
    if not v: return None
    return float(np.mean(v)), float(np.std(v)), len(v)

def info(res):
    a = next(iter(next(iter(res.values()))))["args"]
    return a

def abl_prefix(runs):
    """ablation keys are P4_* in v2 harness, P2_* in v1 (q1/q2)."""
    for p in ("P4_", "P2_"):
        if any(p + "zero" in r["final"] for r in runs): return p
    return None

def headline(data):
    """newest experiment prefix with >=5 modes whose runs did NOT use --rename (q7+ is reported as its own section)."""
    ok = []
    for pre, res in data.items():
        if len(res) >= 5 and not any(r["args"].get("rename") for rs in res.values() for r in rs): ok.append(pre)
    return max(ok, key=lambda p: int(p[1:])) if ok else None

def rename_prefixes(data):
    return sorted([p for p, res in data.items() if any(r["args"].get("rename") for rs in res.values() for r in rs)], key=lambda p: int(p[1:]))

def load_probes(pre):
    """-> {mode: dict} from results/probe_<pre>_<mode>_s0.json (eval_ckpt.py output, seed 0 only)."""
    out = {}
    for m in MODES:
        f = os.path.join(ROOT, "results", f"probe_{pre}_{m}_s0.json")
        if os.path.exists(f): out[m] = json.load(open(f))
    return out

def seed_vals(runs, key):
    return [r["final"][key] for r in sorted(runs, key=lambda r: r["_seed"]) if r["final"].get(key) is not None]
