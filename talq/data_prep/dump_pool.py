"""Materialise the allocation audio pool as ordered file lists.

The pool that decides bit allocation is not a stored tensor. It is re-derived from
the corpora on every run -- `talq.eval.salience.train_pool` indexes the training
split, `random.Random(seed)` shuffles it, and `int(sec / MAX_S[task])` items are
taken from the front. TALQ's allocation and the TAQ-KL baseline both consume it
that way (see `talq.baselines.taq_grid.build_pool_calib`), which is what lets the
two be described as seeing the same audio.

Deriving it rather than storing it is deliberate: a stored tensor would freeze the
crop length, and the tasks need 8, 6 and 1 second contexts respectively.

The consequence is that the pool is invisible in the repository -- reproducing it
requires running the code against the corpora. This module writes it out as plain
path lists so it can be inspected and compared directly:

    python -m talq.data_prep.dump_pool                  # 3,600 s, seed 0
    python -m talq.data_prep.dump_pool --sec 1024

One file per task, plus one per ER session fold, each line one utterance in the
order the pool is consumed. Paths are relative to the directory the corpus
unpacks into, so they carry nothing machine-specific.
"""
import argparse
import os
import random

from talq.eval import salience as S
from talq.paths import DATA_ROOT, RESULTS_ROOT

# IEMOCAP is reached through a manifest and may sit outside DATA_ROOT, so a
# relative path against DATA_ROOT would not shorten it. Anchor on the corpus root.
ANCHORS = ("IEMOCAP_full_release/",)

TASKS = ("pr", "asr", "ks", "ic", "asv")
ER_FOLDS = (1, 2, 3, 4, 5)


def portable(path, root):
    """A path addressed from the directory its corpus unpacks into."""
    p = os.path.realpath(path)
    for a in ANCHORS:
        i = p.find(a)
        if i >= 0:
            return p[i + len(a):]
    return os.path.relpath(p, root) if p.startswith(root) else p


def dump(task, er_fold, sec, seed, data, out_dir):
    """Write one pool list. Returns (name, n_selected, pool_size, max_s)."""
    max_s = S.MAX_S[task]
    pool = S.train_pool(task, data, er_fold=er_fold)
    random.Random(seed).shuffle(pool)
    sel = pool[:max(1, int(sec / max_s))]

    root = os.path.realpath(data)
    name = task if er_fold is None else f"{task}_fold{er_fold}"
    with open(f"{out_dir}/{name}.txt", "w") as fh:
        for c in sel:
            rel = portable(c[0], root)
            # A machine-specific prefix here would be published with the list.
            assert not rel.startswith("/"), f"absolute path would leak: {rel}"
            fh.write(rel + "\n")
    return name, len(sel), len(pool), max_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sec", type=float, default=3600.0,
                    help="pool budget in seconds (default: the 3,600 s the paper "
                         "allocates against)")
    ap.add_argument("--seed", type=int, default=0,
                    help="must match taq_grid's --pool-seed to get the same list")
    ap.add_argument("--data", default=f"{DATA_ROOT}")
    ap.add_argument("--out", default=None,
                    help="default: results/pool{sec}")
    a = ap.parse_args()

    out_dir = a.out or f"{RESULTS_ROOT}/pool{a.sec:g}"
    os.makedirs(out_dir, exist_ok=True)

    jobs = [(t, None) for t in TASKS] + [("er", n) for n in ER_FOLDS]
    for task, fold in jobs:
        name, n, total, max_s = dump(task, fold, a.sec, a.seed, a.data, out_dir)
        print(f"{name:10s} MAX_S={max_s:<4g} pool={total:>6d} -> {n:>4d}"
              f" x {max_s:g}s = {n * max_s:.0f}s", flush=True)
    print(f"-> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
