"""Shared loading/aggregation. numpy only; never imports torch."""
import json, glob, os, re
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODES = ["none", "adapter", "single", "uniform", "multi", "oracle"]
LABEL = {"none": "frozen Qwen (none)", "adapter": "adapter (stateless)", "single": "single (1 shared state)",
         "uniform": "uniform (private, no routing)", "multi": "multi (learned routing)", "oracle": "oracle (true routing)"}
COLOR = {"none": "#8a8a8a", "adapter": "#d98c1f", "single": "#c0463b", "uniform": "#7a5cb8",
         "multi": "#1f7ae0", "oracle": "#1a9a5a"}
PAT = re.compile(r"^(q\d+)_([a-z]+)_s(\d+)\.json$")

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
    v = [r["final"][key] for r in runs if key in r["final"]]
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
