"""Interleaved register programs (mod 10). P registers (letters a..p), each with its own op stream, interleaved.
Modes: 'query' value of one register | 'sum' sum of all finals | 'subset' sum of 3 queried registers.
make() also returns word-level register ids (for oracle routing / visualization)."""
import random
NAMES = "abcdefghijklmnop"

def make_perm(P, n_swaps, rng=random):
    """pure state tracking: P distinct digits in positions 1..P, n_swaps swaps 'sIJ', answer = digit at queried position"""
    arr = rng.sample(range(10), P); cur = arr[:]
    words = [("".join(map(str, arr)), -1)]
    for _ in range(n_swaps):
        i, j = rng.sample(range(P), 2); cur[i], cur[j] = cur[j], cur[i]
        words.append((f"s{i+1}{j+1}", -1))
    q = rng.randrange(P); words.append(("|", -1)); words.append((f"q{q+1}?", -1)); words.append(("=", -1))
    return words, cur[q]

def make(P, n_ops, mode="query", rng=random, rename=False, shuffle_init=False):
    if mode == "perm": return make_perm(P, n_ops, rng)
    names = NAMES if not rename else "".join(rng.sample("abcdefghijklmnopqrstuvwxyz", 16))
    if mode == "perm": return make_perm(P, n_ops, rng)
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
