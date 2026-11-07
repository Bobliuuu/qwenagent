"""Interleaved register programs. P registers, each with its own op stream, interleaved.
Ops mod 10. Query 'query': value of one register. Query 'sum': sum of all finals mod 10."""
import random
NAMES = "abcdefgh"

def make(P, n_ops, mode="query", rng=random):
    init = [rng.randrange(10) for _ in range(P)]
    val = init[:]
    toks = [f"{NAMES[i]}={init[i]}" for i in range(P)]
    for _ in range(n_ops):
        i = rng.randrange(P)
        op = rng.choice("+-*")
        d = rng.randrange(10 if op != "*" else 1, 10 if op != "*" else 5) if op != "*" else rng.randrange(1, 5)
        val[i] = (val[i] + d) % 10 if op == "+" else (val[i] - d) % 10 if op == "-" else (val[i] * d) % 10
        toks.append(f"{NAMES[i]}{op}{d}")
    if mode == "query":
        q = rng.randrange(P); ans = val[q]; qs = f"{NAMES[q]}?"
    else:
        ans = sum(val) % 10; qs = "sum?"
    return " ".join(toks) + " | " + qs + " =", ans

def batch(B, Ps, nops, mode, rng=random):
    xs, ys = [], []
    for _ in range(B):
        P = rng.choice(Ps); n = rng.randrange(nops[0], nops[1] + 1)
        x, y = make(P, n, mode, rng); xs.append(x); ys.append(y)
    return xs, ys
