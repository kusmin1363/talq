"""Add TAQ allocations for an arbitrary budget x an arbitrary (lo,hi) pair.

In TAQ, K is not a free hyperparameter but the means of hitting a budget:
    avg_bit = lo + K*(hi - lo)
So one budget can be made by several (lo,hi,K) combinations, and "was TAQ given the
best config at each budget" becomes the criterion of baseline fairness. This file
produces those combinations.

Bracketing feasibility against TALQ (by full_tc/summary.csv, bracketed on both sides
out of 30 cells):
    2.25 -> 16   (hits the lower limit of the lambda sweep. unsuitable for comparison)
    2.50 -> 24
    3.00 -> 30   3.25 -> 30   3.50 -> 30      <- these three are the effective comparison range

kind keeps the pair used as `-p{lo}{hi}`. This is because with two configs for the
same (backbone, task, budget) the resume key collides and one of them is silently
skipped. The report in taq_vs_talq groups by the `taq-is*` / `taq-kl*` prefix and
takes the best per method.

The TAQ-KL score has its injected noise delta = r/(2^b - 1) tied to the target bit,
so it is taken only from rows whose lo matches (the grid CSV stores the lo=2 version
and the lo=3 version separately).

  python -m talq.baselines.taq_budget --budgets 3.0,3.5
"""
import argparse
import csv
import json
import os
from fractions import Fraction as Fr

from talq.baselines import taq
from talq.paths import RESULTS_ROOT

TASKS6 = ["pr", "asr", "er", "ks", "ic", "asv"]
PAIRS = [(2, 3), (2, 4), (3, 4)]


def configs_for(budget, L):
    """All (lo, hi, n) that make that budget non-degenerately (n != 0, L).

    **Seen as within half a grid step, not as an exact match.** If the target is a
    repeating decimal such as 3 and 1/3, it can never become equal as a decimal
    string -- Fr("3.3333") = 33333/10000 while (36+4)/12 = 10/3, so == is false
    forever. That is why on 2026-09-11 spec emitted 0 rows and passed silently (the
    TAQ evaluation ended as "nothing to do").

    The grid spacing is 1/L, so within half a step (0.5/L) the n that makes that
    budget is unique. The avg recorded is not a corrected value but **the mean of the
    actual allocation**.
    """
    out = []
    tol = 0.5 / L
    for lo, hi in PAIRS:
        for n in range(1, L):
            if abs(float(Fr(n * hi + (L - n) * lo, L)) - float(budget)) < tol:
                out.append((lo, hi, n))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", default="3.0,3.5")
    ap.add_argument("--backbones", default="w2v2,wavlm,hubert,wavlmL,hubertL")
    ap.add_argument("--tasks", default=",".join(TASKS6))
    ap.add_argument("--grid", default=f"{RESULTS_ROOT}/taq/taq_grid_taskcalib.csv")
    ap.add_argument("--outdir", default=f"{RESULTS_ROOT}/taq")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.grid)))
    budgets = [float(x) for x in a.budgets.split(",")]
    total = 0
    for bk in a.backbones.split(","):
        out = f"{a.outdir}/taq_spec_{bk}.csv"
        os.makedirs(a.outdir, exist_ok=True)
        # On the first emit into a new outdir the file does not exist. Start from
        # empty there instead of dying -- that is the case when the calib axis is
        # changed and emitted into a separate directory.
        cur = list(csv.DictReader(open(out))) if os.path.exists(out) else []
        fields = list(cur[0].keys()) if cur else [
            "kind", "task", "budget", "seq", "avg", "lo", "hi", "n_layers",
            "calib", "changed"]
        have = {(r["kind"], r["task"], str(float(r["budget"]))) for r in cur}
        new = []
        for task in a.tasks.split(","):
            for meth in ("taq-is", "taq-kl"):
                for bud in budgets:
                    L = None
                    for lo, hi, n in configs_for(bud, 24) + configs_for(bud, 12):
                        # the score must match lo (KL's delta is tied to lo)
                        src = [r for r in rows
                               if r["backbone"] == bk and r["task"] == task
                               and int(r["lo"]) == lo
                               and (r["method"] == meth
                                    or (meth == "taq-kl"
                                        and r["method"] == "taq-kl-cos"))]
                        if not src:
                            continue
                        L = int(src[0]["n_layers"])
                        if (lo, hi, n) not in configs_for(bud, L):
                            continue          # at this layer count that combination cannot hit the budget
                        kind = f"{meth}-p{lo}{hi}"
                        if (kind, task, str(bud)) in have:
                            continue
                        bm = taq.allocate_topk(json.loads(src[0]["scores"]),
                                               n / L, hi, lo)
                        seq = "".join(str(bm[k]) for k in range(L))
                        avg = sum(int(c) for c in seq) / L
                        # the target may be a repeating decimal. Within half a grid
                        # step it is hit, and the table uses avg (what is actually achieved).
                        assert abs(avg - bud) <= 0.5 / L + 1e-9, \
                            (bk, task, kind, bud, avg)
                        have.add((kind, task, str(bud)))
                        new.append({"kind": kind, "task": task, "budget": bud,
                                    "seq": seq, "avg": round(avg, 4),
                                    "lo": lo, "hi": hi, "n_layers": L,
                                    "calib": src[0]["calib"]})
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fields)
            w.writeheader(); w.writerows(cur + new)
        total += len(new)
        print(f"  {bk:9s} +{len(new):3d} rows (total {len(cur)+len(new)})")
    print(f"\nadded {total} rows")


if __name__ == "__main__":
    main()
