"""Interleaved register programs (mod 10). P registers (letters a..p), each with its own op stream, interleaved.
Modes: 'query' value of one register | 'sum' sum of all finals | 'subset' sum of 3 queried registers.
make() also returns word-level register ids (for oracle routing / visualization)."""
import random
NAMES = "abcdefghijklmnop"

def make(P, n_ops, mode="query", rng=random, rename=False, shuffle_init=False):
    names = NAMES if not rename else "".join(rng.sample("abcdefghijklmnopqrstuvwxyz", 16))
    init = [rng.randrange(10) for _ in range(P)]
    val = init[:]
    order = list(range(P)); shuffle_init and rng.shuffle(order)
    words = [(f"{names[i]}={init[i]}", i) for i in order]
    for _ in range(n_ops):
        i = rng.randrange(P); op = rng.choice("+-*")
        d = rng.randrange(1, 5) if op == "*" else rng.randrange(1, 10)
        val[i] = (val[i] + d) % 10 if op == "+" else (val[i] - d) % 10 if op == "-" else (val[i] * d) % 10
        words.append((f"{names[i]}{op}{d}", i))
    words.append(("|", -1))
    if mode == "query":
        q = rng.randrange(P); ans = val[q]; words.append((f"{names[q]}?", -1))
    elif mode == "subset":
        k = min(3, P); qs = rng.sample(range(P), k); ans = sum(val[i] for i in qs) % 10
        words.append(("+".join(names[i] for i in qs) + "?", -1))
    else:
        ans = sum(val) % 10; words.append(("sum?", -1))
    words.append(("=", -1))
    return words, ans

def render(words):
    text, regs, pos = "", [], 0
    spans = []
    for w, r in words:
        if text: text += " "
        spans.append((len(text), len(text) + len(w), r)); text += w
    return text, spans

def batch(B, Ps, nops_per_reg, mode, rng=random, **kw):
    xs, ys, sp = [], [], []
    for _ in range(B):
        P = rng.choice(Ps); n = P * rng.randrange(nops_per_reg[0], nops_per_reg[1] + 1)
        words, y = make(P, n, mode, rng, **kw); t, s = render(words); xs.append(t); ys.append(y); sp.append(s)
    return xs, ys, sp
