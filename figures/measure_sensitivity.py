"""Figure 1 re-evaluation -- per-layer isolated 2-bit sensitivity with the per-task
1,024-second calibration candidates. This is what produced the CSV behind the
paper's Figure 1.

Why talq/eval/sensitivity.py cannot be used: that script **applies one candidate
directory in common to all six tasks** and writes six task metrics together on
one row (that is the structure from the era of the single generic emilia
calibration set). The new Figure 1 has to read a different 1,024-second candidate
per task and evaluate **that task only**.

Rules (fixed 2026-09-12):
  1. FP32 backbone + frozen task head.
  2. Replace layer k only, with that task's quants_tc1024_{task}/{bk}_L{k}_b2.pt.
  3. The remaining layers stay FP32 (32 in talq.search.compose = restore the original).
  4. Evaluate that task only, on the full official test split.
  5. sensitivity = quantization error - FP32 error.
  6. Rank independently per (backbone, task) -> Top-4.
  7. For ER, evaluate with the per-fold candidates (quants_tc1024_er_fold{n}) and
     the held-out session, then **average the per-layer increase over the 5 folds**
     and only then rank (ranking per fold and averaging the ranks mixes the ranks
     up).
  8. The old emilia results (results/sens_class.csv) are not touched. This writes
     a new CSV.

  python figures/measure_sensitivity.py --backbones w2v2 --tasks pr   # partial run
  python figures/measure_sensitivity.py --smoke 2                     # only 2 layers
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import argparse
import csv
import os
import statistics as st
import time

import torch

import talq.search as A
import talq.alloc as D3
from talq.paths import QUANT_ROOT, RESULTS_ROOT

QDIR = "quants_1024/GPTQ/quants_tc1024_{key}"
FOLDS = [1, 2, 3, 4, 5]
MET = D3.METRIC


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", default="w2v2,hubert,wavlm")
    ap.add_argument("--tasks", default="pr,asr,ks,ic,er,asv")
    ap.add_argument("--bits", type=int, default=2)
    ap.add_argument("--smoke", type=int, default=0, help="if >0, only the first N layers")
    ap.add_argument("--csv", default=f"{RESULTS_ROOT}/sens_1024_iso.csv")
    a = ap.parse_args()
    dev = "cuda"
    D3.BITS = [2, 3, 4]

    os.makedirs(os.path.dirname(a.csv), exist_ok=True)
    fresh = not os.path.exists(a.csv)
    done = set()
    if not fresh:
        for r in csv.DictReader(open(a.csv)):
            done.add((r["backbone"], r["task"], int(r["layer"]), int(r["fold"] or 0)))
    fh = open(a.csv, "a", newline="")
    W = csv.DictWriter(fh, ["backbone", "task", "layer", "bits", "fold",
                            "metric", "fp32", "value", "delta", "quant_dir", "sec"])
    if fresh:
        W.writeheader(); fh.flush()

    for bk in a.backbones.split(","):
        tasks = [t for t in a.tasks.split(",") if t]
        log(f"===== {bk} loading evaluation assets (full official test split)")
        m, probes, er_probes, d, orig, nl = D3.setup_full(bk, dev)
        layers = range(min(a.smoke, nl) if a.smoke else nl)
        for t in tasks:
            for fold in (FOLDS if t == "er" else [0]):
                key = f"er_fold{fold}" if t == "er" else t
                qdir = QDIR.format(key=key)
                # FP32 baseline -- measured through the same code path so that delta does not ride on a difference of yardstick
                m.load_state_dict(orig)
                t0 = time.time()
                base = D3.eval_full(t, m, probes, er_probes, d, dev,
                                    fold if t == "er" else None)
                log(f"  {bk}/{t}{f'/f{fold}' if fold else ''} FP32 "
                    f"{MET[t]}={base:.4f} ({time.time()-t0:.0f}s)")
                LQ = D3.load_layer_quants(bk, nl, qdir)
                for k in layers:
                    if (bk, t, k, fold) in done:
                        continue
                    vec = [32] * nl
                    vec[k] = a.bits
                    m.load_state_dict(orig)
                    A.compose(m, orig, vec, LQ, dev)
                    t1 = time.time()
                    v = D3.eval_full(t, m, probes, er_probes, d, dev,
                                     fold if t == "er" else None)
                    W.writerow({"backbone": bk, "task": t, "layer": k,
                                "bits": a.bits, "fold": fold, "metric": MET[t],
                                "fp32": round(base, 6), "value": round(v, 6),
                                "delta": round(v - base, 6), "quant_dir": qdir,
                                "sec": round(time.time() - t1)})
                    fh.flush()
                    log(f"    L{k:<2} {MET[t]}={v:.4f}  Δ={v-base:+.4f}  "
                        f"({time.time()-t1:.0f}s)")
                del LQ
                torch.cuda.empty_cache()
        del m, probes, er_probes, orig
        torch.cuda.empty_cache()
    fh.close()
    log(f"saved {a.csv}")


if __name__ == "__main__":
    main()
