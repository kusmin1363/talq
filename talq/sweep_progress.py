"""Main sweep progress -- counts the **artifacts**, not the log.

The earlier sweep runner's "done" log cannot be trusted. talq.alloc catches a
per-task OOM, saves an empty result and exits with rc=0 (on 2026-09-05 six cells
went empty that way). So all that is looked at here is whether each cell JSON's
allocs actually contains the task.

Cell = (quantizer, backbone, lambda, task). ER is 5x because each fold has its
own out-dir.

  python -m talq.sweep_progress
  python -m talq.sweep_progress --detail        # list of the empty cells
"""
import argparse, glob, json, os, re

from talq.paths import RESULTS_ROOT

BKS = ["w2v2", "wavlm", "hubert", "wavlmL", "hubertL"]
T5 = ["pr", "asr", "ks", "ic", "asv"]
LAMS = [0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 5]
FOLDS = [1, 2, 3, 4, 5]


def done_tasks(out, bk, lam):
    fp = f"{RESULTS_ROOT}/{out}/{bk}/l{lam:g}/{bk}_b3.json"
    try:
        return set(json.load(open(fp)).get("allocs", {}))
    except Exception:
        return set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detail", action="store_true")
    a = ap.parse_args()
    print("main sweep progress  (cell = quantizer x backbone x lambda x task)\n")
    gt = gd = 0
    miss = []
    for q in ("gptq", "awq"):
        print(f"--- {q.upper()}")
        print(f"{'':10}" + "".join(f"{b:>11}" for b in BKS) + f"{'subtotal':>12}")
        for lab, out, ts in ([("non-ER", f"sw_{q}", T5)] +
                             [(f"ER f{n}", f"sw_{q}_erf{n}", ["er"]) for n in FOLDS]):
            row, tot, dn = "", 0, 0
            for bk in BKS:
                n = len(LAMS) * len(ts)
                c = sum(len(done_tasks(out, bk, l) & set(ts)) for l in LAMS)
                tot += n; dn += c
                row += f"{f'{c}/{n}':>11}"
                if a.detail and c < n:
                    for l in LAMS:
                        for t in set(ts) - done_tasks(out, bk, l):
                            miss.append(f"{out}/{bk}/l{l:g}/{t}")
            gt += tot; gd += dn
            print(f"{lab:10}" + row + f"{f'{dn}/{tot}':>12}"
                  f"  {100*dn/tot if tot else 0:5.1f}%")
        print()
    print(f"overall {gd}/{gt}  ({100*gd/gt if gt else 0:.1f}%)")
    if a.detail and miss:
        print(f"\n{len(miss)} empty cells (first 40)")
        for m in miss[:40]:
            print("  " + m)


if __name__ == "__main__":
    main()
