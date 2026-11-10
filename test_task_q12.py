"""Sample dump + sanity test for ledger/files tasks: python test_task_q12.py > results/q12_samples.txt"""
import random, task
rng = random.Random(0)
for mode in ("ledger", "files"):
    for P, n in ((2, 4), (3, 6), (6, 12)):
        for _ in range(2):
            w, y = task.make(P, n, mode, rng); t, sp = task.render(w)
            assert all(t[a:b] == x for (a, b, _), (x, _) in zip(sp, w)) and 0 <= y <= 9
            print(f"[{mode} P={P}] ans={y}\n{t}\nentity id per sentence: {[r for _, r in w]}\n")
