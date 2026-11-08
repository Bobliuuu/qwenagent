"""Generates viz/fixtures/synthetic_trace.json: SYNTHETIC, same schema as traces/q3_*_s0.json. Never used in README figures."""
import json, os, random
import numpy as np
here = os.path.dirname(os.path.abspath(__file__)); os.makedirs(os.path.join(here, "fixtures"), exist_ok=True)
rng = random.Random(1); nr = np.random.default_rng(1); out = []
for P in (3, 5):
    names = "abcdefghijklmnop"; toks, reg = [], []
    for i in range(P):
        for piece in (names[i], "=", str(rng.randrange(10))): toks.append(piece if piece != names[i] else piece); reg.append(i)
    for _ in range(P * 3):
        i = rng.randrange(P)
        for piece in (" " + names[i], rng.choice("+-*"), str(rng.randrange(1, 9))): toks.append(piece); reg.append(i)
    for piece in (" |", " a+b+c", "?", " ="): toks.append(piece); reg.append(-1)
    perm = nr.permutation(P); T = len(toks); A = []
    for t in range(T):
        lo = np.full(P, 1.0 / P)
        if reg[t] >= 0:
            conf = min(.85, .25 + t / T); lo = lo * (1 - conf); lo[perm[reg[t]]] += conf
        a = np.abs(lo + nr.normal(0, .03, P)); A.append((a / a.sum()).round(4).tolist())
    out.append({"text": " ".join(toks), "answer": 7, "pred": 7, "tokens": toks, "reg": reg, "assign": A, "P": P, "_synthetic": True})
json.dump(out, open(os.path.join(here, "fixtures", "synthetic_trace.json"), "w"))
print("fixture written")
