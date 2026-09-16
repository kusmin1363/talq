"""Run TAQ-IS / TAQ-KL over all 5 backbones x 7 tasks in our task-calib setting.

Why taq.py is not called repeatedly: a fresh process per cell re-pays the model load
and the CUDA init every time. Here the model is loaded once per backbone and the 7
tasks are iterated over it.

calib follows our convention as is: calib/task_{task}.pt (48x2s = 96s, ks alone 96x1s).
ER has separate fold-split files (task_er_fold*.pt), but the pooled task_er.pt is used
here.

**On the bit allocation.** TAQ's allocation policy is essentially two-level in the
original paper -- "top K% by score -> hi, the rest -> lo". A three-level allocation
does not exist in the original paper. So to place it in our {2,3,4} regime a (lo,hi)
pair has to be chosen, and here all three pairs are emitted:
  (2,4) K=25% -> 2.50 bit on average
  (2,3) K=25% -> 2.25 bit on average
  (3,4) K=25% -> 3.25 bit on average
The TAQ-IS score is independent of (lo,hi), so it is computed once. For TAQ-KL the
injected noise magnitude delta_ell = r_ell/(2^b - 1) is tied to the target bit b, so
b=2 and b=3 are emitted separately.

  python -m talq.baselines.taq_grid                      # everything
  python -m talq.baselines.taq_grid --backbones w2v2 --tasks er,ks
"""
import argparse
import csv
import json
import os

import torch

from talq.baselines import taq
from talq.paths import DATA_ROOT, PROBES, PROBES_SUPERB, REPO_ROOT, RESULTS_ROOT

TASKS = ["pr", "asr", "er", "ks", "ic", "sid", "asv"]
PAIRS = [(2, 4), (2, 3), (3, 4)]
FIELDS = ["backbone", "task", "method", "lo", "hi", "topk", "n_layers",
          "hi_layers", "avg_bits_linear", "W_GB", "calib", "scores"]


def build_pool_calib(task, er_fold, sec, seed, dev):
    """Build the calib tensor under the same convention as TALQ(talq.alloc.run_task).

        pool = S.train_pool(task, data, er_fold)   # list of file paths
        random.Random(0).shuffle(pool)
        pool = pool[:int(sec / MAX_S[task])]
        wav  = load_wav(path, max_s=MAX_S[task])

    Not a stored .pt but **read from the source files every time**. The length is
    MAX_S, so it is the same context TALQ sees in training (with a 2s calib one
    cannot claim to have seen an 8s context). The crop position is random inside
    load_wav, so this is one draw from that pool.
    """
    import random as _r
    import numpy as _np
    from talq.eval import salience as _S
    from talq.eval.probe_train import load_wav as _lw
    max_s = _S.MAX_S[task]
    pool = _S.train_pool(task, f"{DATA_ROOT}", er_fold=er_fold)
    _r.Random(seed).shuffle(pool)
    n = max(1, int(sec / max_s))
    pool = pool[:n]
    _r.seed(seed)                      # the crop in load_wav uses the global RNG
    L = int(max_s * 16000)
    out = _np.zeros((len(pool), L), dtype="float32")
    for i, c in enumerate(pool):
        w = _lw(c[0], max_s=max_s)
        out[i, :len(w)] = w[:L]
    tag = f"train_pool({task}{f'/fold{er_fold}' if er_fold else ''}, {sec:g}s, "
    print(f"  pool calib {task}: {len(pool)} utterances x {max_s:g}s = {len(pool)*max_s:.0f}s",
          flush=True)
    return torch.from_numpy(out).to(dev), tag + f"n={len(pool)})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", default="w2v2,wavlm,hubert,wavlmL,hubertL")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--topk", type=float, default=taq.TOPK)
    ap.add_argument("--reservoir", type=int, default=taq.R_RESERVOIR)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probes", default=PROBES)
    ap.add_argument("--out", default=f"{RESULTS_ROOT}/taq/taq_grid_taskcalib.csv")
    ap.add_argument("--calib-tpl", default="calib/task_{task}.pt",
                    help="calib path template. Touch it only when changing the "
                         "calib length axis (e.g. calib/task1024_{task}.pt). The "
                         "default is the 96s version, and that is what the main "
                         "numbers in the paper are.")
    ap.add_argument("--calib-pool", type=float, default=0, metavar="SEC",
                    help="if greater than 0, **the .pt is not read** and calib is "
                         "built the same way TALQ does: the file list from "
                         "talq.eval.salience.train_pool() is shuffled with "
                         "random.Random(0), int(SEC/MAX_S[task]) items are taken "
                         "from the front, and they are read with "
                         "load_wav(max_s=MAX_S). That is, TAQ-KL's importance sees "
                         "the **same source utterance pool** as TALQ's allocation "
                         "learning. The crop position is random inside load_wav so "
                         "it is a single draw, and this is 'same audio pool', not "
                         "'identical samples'.")
    ap.add_argument("--pool-seed", type=int, default=0)
    ap.add_argument("--head", default=None, metavar="task=path",
                    help="replace the scoring task head. The TAQ-KL score is a "
                         "divergence of task head outputs, so the head changes the "
                         "ranking. e.g.: "
                         "'ks=probes/hubert-large-ll60k_ks_superb_baldev.pt'")
    ap.add_argument("--er-fold", type=int, default=None, choices=[1, 2, 3, 4, 5],
                    help="SUPERB fold of ER. If given, calib puts er_fold{n} in the "
                         "{task} slot (calib/task1024_er_fold{n}.pt) and the task "
                         "head used is probes_superb/{tag}_er_fold{n}.pt. ER of "
                         "TALQ and Sens-DP is 5-fold, so TAQ must also be scored "
                         "per fold to match. The fold is distinguished by the calib "
                         "column of the output CSV (schema unchanged).")
    ap.add_argument("--pairs", default="2,4;2,3;3,4",
                    help="(lo,hi) pairs. TAQ's allocation is two-level, so a pair "
                         "has to be chosen. Table 1 uses only '3,4' -- to match the "
                         "bit set of TALQ")
    a = ap.parse_args()

    from talq.eval import arm_eval
    from talq.quant.precompute import BACKBONES

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    pairs = [tuple(int(x) for x in q.split(",")) for q in a.pairs.split(";") if q]
    rows = []
    for bk in a.backbones.split(","):
        model = None
        tag = BACKBONES[bk].split("/")[-1]
        for task in a.tasks.split(","):
            # ER 5-fold: both calib and the task head have separate per-fold files.
            # Using the pooled ones (task1024_er.pt / the pooled _er.pt under PROBES)
            # does not line up with the fold.
            key = f"er_fold{a.er_fold}" if (task == "er" and a.er_fold) else task
            cpath = f"{REPO_ROOT}/" + a.calib_tpl.format(task=key)
            ppath = (f"{PROBES_SUPERB}/{tag}_{key}.pt"
                     if key != task else f"{a.probes}/{tag}_{task}.pt")
            for item in (a.head or "").split(","):
                if item.strip() and item.split("=", 1)[0].strip() == task:
                    ppath = f"{REPO_ROOT}/" + item.split("=", 1)[1]
                    print(f"  head swap {task} <- {os.path.basename(ppath)}", flush=True)
            if a.calib_pool > 0:
                calib, cpath = build_pool_calib(task, a.er_fold, a.calib_pool,
                                                a.pool_seed, dev)
            else:
                if not os.path.exists(cpath):
                    print(f"  [skip] {bk}/{task}: no calib {cpath}", flush=True)
                    continue
                calib = torch.load(cpath, map_location=dev)
            if model is None:                 # loaded only once per backbone
                model = taq.load_backbone(bk, dev, calib)
            L = len(model.encoder.layers)
            torch.manual_seed(a.seed)

            scored = {}
            res, var = taq.collect_is_stats(model, calib, r=a.reservoir,
                                            batch=a.batch, seed=a.seed)
            scored["taq-is"] = taq.score_is(res, var, device=dev)["score"]

            if os.path.exists(ppath):
                probe = arm_eval.build_probe(task, torch.load(ppath, map_location=dev), dev)
                emb = (task == "asv")     # asv has no posterior, so embedding cosine (D2)
                name = "taq-kl-cos" if emb else "taq-kl"
                for tb in (2, 3):
                    scored[f"{name}@b{tb}"] = taq.score_kl(
                        model, probe, calib, target_bit=tb, batch=a.batch,
                        seed=a.seed, embed_mode=emb)["score"]
                del probe
            else:
                print(f"  [skip kl] {bk}/{task}: no task head", flush=True)

            for method, sc in scored.items():
                for lo, hi in pairs:
                    # the KL score is tied to the target bit, so use only the one matching lo
                    if "@b" in method and int(method.split("@b")[1]) != lo:
                        continue
                    bm = taq.allocate_topk(sc, a.topk, hi, lo)
                    f = taq.weight_footprint(model, bm)
                    rows.append({
                        "backbone": bk, "task": task,
                        "method": method.split("@b")[0], "lo": lo, "hi": hi,
                        "topk": a.topk, "n_layers": L,
                        "hi_layers": " ".join(str(k) for k in sorted(bm) if bm[k] == hi),
                        "avg_bits_linear": round(f["avg_bits_linear"], 4),
                        "W_GB": round(f["W_GB"], 5),
                        "calib": os.path.basename(cpath),
                        "scores": json.dumps([round(x, 6) for x in sc]),
                    })
            print(f"  {bk}/{task}: " + "  ".join(
                f"{m}=[{' '.join(str(k) for k in sorted(taq.allocate_topk(sc, a.topk, 4, 2)) if taq.allocate_topk(sc, a.topk, 4, 2)[k] == 4)}]"
                for m, sc in scored.items()), flush=True)
        del model
        torch.cuda.empty_cache()

    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
