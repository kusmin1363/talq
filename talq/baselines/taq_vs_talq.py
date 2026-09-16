"""Evaluate TAQ allocations on **the same axis** as TALQ and compare by bracketing.

Why the earlier TAQ evaluation script cannot be used -- the axis differs in
two places:

  1) How it is realized. TALQ **composes** per-layer pre-quantized candidates
     (quants_tc_{task}/{bk}_L{k}_b{b}.pt) with `talq.search.compose`. The earlier
     TAQ evaluation did an actual requantization with sequential GPTQ. The two
     paths give different values (super-additivity). To compare, the path has to
     be the same.
  2) The evaluation protocol. TALQ's `talq.alloc.eval_full` is the **full** official
     SUPERB test split (all of PR/ASR test-clean, all 3081 of ks / 3793 of ic, the
     full official ASV trial list) and its ER is **fold5 alone** (leakage avoidance,
     probes_superb). The earlier TAQ evaluation used talq.eval.arm_eval's caps
     (n_ctc=64, CAP 1000, sv_stride 4) and the probes/ ER.

So here setup_full/eval_full of talq.alloc and talq.search.compose are called as
they are. uniform 2/3/4 is already measured on the same axis in the uni2/uni3/uni4
columns of TALQ's summary.csv, so it is not recomputed.

Bracketing comparison: TALQ cannot constrain avg_bit exactly and draws the curve
with a lambda sweep. So for TAQ's fixed budgets (2.25 / 3.25),
  lower point = the largest among avg_bit <= target
  upper point = the smallest among avg_bit >= target
are taken to see where TAQ lies between them. Better than the upper point makes a
TAQ win certain, worse than the lower point makes a loss certain, and in between it
has to be judged by efficiency per bit.

  python -m talq.baselines.taq_vs_talq --backbones w2v2 --budgets 3.25
  python -m talq.baselines.taq_vs_talq --report-only   # table from what is already measured
"""
import argparse
import csv
import os
import time

import torch

from talq.paths import REPO_ROOT, RESULTS_ROOT

TASKS6 = ["pr", "asr", "er", "ks", "ic", "asv"]
FIELDS = ["backbone", "task", "kind", "budget", "avg_bit", "seq", "metric",
          "value", "quant_dir", "eval", "sec"]


def load_taq_specs(outdir, backbones, tasks, budgets, kind_sub=None,
                   only_changed=False):
    """{(bk, task): [(kind, budget, seq)]} — from the talq.baselines.taq_budget output.

    kind_sub    : only rows whose kind contains **all** of these strings. Given
                  several comma-separated, it is an AND
                  (e.g. "-s3600" / "taq-kl,-s3600" -> KL only & the 3600s version only).
    only_changed: only rows whose changed column is "1". If the 3600s version has the
                  same allocation as the 96s version, it is identical to the value
                  already measured, so re-evaluating is a waste.
    """
    out = {}
    for bk in backbones:
        fp = f"{outdir}/taq_spec_{bk}.csv"
        if not os.path.exists(fp):
            print(f"  [skip] {fp} missing"); continue
        for r in csv.DictReader(open(fp)):
            if r["task"] not in tasks:
                continue
            b = round(float(r["budget"]), 4)
            if budgets and b not in budgets:
                continue
            if kind_sub and not all(x in r["kind"]
                                    for x in kind_sub.split(",") if x):
                continue
            if only_changed and str(r.get("changed", "")) != "1":
                continue
            out.setdefault((bk, r["task"]), []).append((r["kind"], b, r["seq"]))
    return out


def bracket(summary_rows, bk, task, target):
    """The two TALQ points bracketing the target avg_bit. (lower, upper), each dict or None."""
    v = [r for r in summary_rows if r["backbone"] == bk and r["task"] == task]
    below = [r for r in v if float(r["avg_bit"]) <= target]
    above = [r for r in v if float(r["avg_bit"]) >= target]
    lo = max(below, key=lambda r: float(r["avg_bit"])) if below else None
    hi = min(above, key=lambda r: float(r["avg_bit"])) if above else None
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", default="w2v2,wavlm,hubert,wavlmL,hubertL")
    ap.add_argument("--tasks", default=",".join(TASKS6))
    ap.add_argument("--budgets", default="2.25,3.25")
    ap.add_argument("--specdir", default=f"{RESULTS_ROOT}/taq")
    ap.add_argument("--summary",
                    default=f"{RESULTS_ROOT}/full_tc/summary.csv")
    ap.add_argument("--quant-dir", default="quants_tc_{task}")
    ap.add_argument("--csv", default=f"{RESULTS_ROOT}/taq/taq_full_tc.csv",
                    help="write target. For reading, the --read pattern is used.")
    ap.add_argument("--read", default=f"{RESULTS_ROOT}/taq/taq_full_tc*.csv",
                    help="glob to read in for the report. Even if split per backbone, it is merged into one table.")
    ap.add_argument("--er-fold", type=int, default=None, choices=[1, 2, 3, 4, 5],
                    help="SUPERB fold of ER. If given, (i) the evaluation is aligned "
                         "to test=Session n and (ii) quant-dir resolves to "
                         "..._er_fold{n}. Omitted, it is the old behavior (fold5 "
                         "alone), and that does not match TALQ's 5-fold.")
    ap.add_argument("--head", default=None, metavar="task=path",
                    help="replace the evaluation head. Must be the same head used when the allocation was produced.")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--kind-sub", default=None,
                    help="evaluate only the specs whose kind contains this string (e.g. -s3600)")
    ap.add_argument("--only-changed", action="store_true",
                    help="evaluate only the specs with changed=1")
    a = ap.parse_args()

    backbones = a.backbones.split(",")
    tasks = a.tasks.split(",")
    budgets = {round(float(x), 4) for x in a.budgets.split(",")}
    summary = list(csv.DictReader(open(a.summary)))
    specs = load_taq_specs(a.specdir, backbones, tasks, budgets,
                           a.kind_sub, a.only_changed)

    if not a.report_only:
        from talq import alloc as D3
        from talq import search as A

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        done = set()
        fresh = not os.path.exists(a.csv)
        if not fresh:
            for r in csv.DictReader(open(a.csv)):
                done.add((r["backbone"], r["task"], r["kind"], r["budget"]))
        fh = open(a.csv, "a", newline="")
        w = csv.DictWriter(fh, FIELDS)
        if fresh:
            w.writeheader(); fh.flush()

        for bk in backbones:
            todo = [t for t in tasks if (bk, t) in specs
                    and any((bk, t, k, f"{b}") not in done for k, b, _ in specs[(bk, t)])]
            if not todo:
                print(f"[{bk}] nothing to do"); continue
            print(f"[{bk}] preparing evaluation assets (official full split)...", flush=True)
            m, probes, er_probes, d, orig, nl = D3.setup_full(bk, dev)
            for item in (a.head or "").split(","):
                if not item.strip():
                    continue
                t_, path = item.split("=", 1)
                fp_ = f"{REPO_ROOT}/{path}"
                if not os.path.exists(fp_):
                    raise FileNotFoundError(f"head missing: {fp_}")
                from talq.eval import arm_eval as _AE
                probes[t_.strip()] = _AE.build_probe(
                    t_.strip(), torch.load(fp_, map_location=dev,
                                           weights_only=False), dev)
                print(f"  head swap {t_.strip()} <- {os.path.basename(fp_)}", flush=True)
            for t in tasks:
                if (bk, t) not in specs:
                    continue
                # ER fold: the candidate directory is quants_tc1024_er_fold{n}.
                qkey = f"er_fold{a.er_fold}" if (t == "er" and a.er_fold) else t
                qdir = a.quant_dir.format(task=qkey)
                LQ = None
                for kind, bud, seq in specs[(bk, t)]:
                    if (bk, t, kind, f"{bud}") in done:
                        print(f"  [skip] {bk}/{t}/{kind} @{bud}"); continue
                    if LQ is None:
                        LQ = D3.load_layer_quants(bk, nl, qdir)
                    t0 = time.time()
                    m.load_state_dict(orig)
                    A.compose(m, orig, [int(c) for c in seq], LQ, dev)
                    v = D3.eval_full(t, m, probes, er_probes, d, dev,
                                     a.er_fold if t == "er" else None)
                    rec = {"backbone": bk, "task": t, "kind": kind,
                           "budget": bud, "avg_bit": round(
                               sum(int(c) for c in seq) / len(seq), 4),
                           "seq": seq, "metric": D3.METRIC[t],
                           "value": round(v, 6), "quant_dir": qdir,
                           "eval": "full", "sec": round(time.time() - t0)}
                    w.writerow(rec); fh.flush()
                    print(f"  {bk}/{t:4s} {kind:7s} @{bud} {seq}  "
                          f"{rec['metric']}={v:.4f}  ({rec['sec']}s)", flush=True)
                del LQ
                torch.cuda.empty_cache()
            del m, probes, er_probes, orig
            torch.cuda.empty_cache()
        fh.close()

    # ---------------- report ----------------
    import glob as _glob
    files = sorted(_glob.glob(a.read))
    if not files:
        print(f"no measurement files: {a.read}"); return
    taq = {}
    for fp in files:
        for r in csv.DictReader(open(fp)):
            taq[(r["backbone"], r["task"], r["kind"],
                 round(float(r["budget"]), 4))] = r
    if not taq:
        print(f"no measurement rows ({len(files)} files)"); return
    print(f"read: {', '.join(os.path.basename(f) for f in files)}  ({len(taq)} rows)")
    for bud in sorted(budgets):
        print(f"\n{'='*100}\nbudget {bud} bit  (full official evaluation, quants_tc composition, lower is better everywhere)\n{'='*100}")
        print(f"{'bk':9s}{'task':5s}{'TAQ-IS':>9s}{'TAQ-KL':>9s} | "
              f"{'TALQ-lo':>9s}{'bit':>6s}{'TALQ-hi':>9s}{'bit':>6s} | verdict")
        for bk in backbones:
            for t in tasks:
                # The same budget can be reached by several configs (at 24 layers,
                # besides (2,3)/(3,4), (2,4) with a different K is also possible). To
                # give TAQ a fair chance the best among the configs has to be taken
                # per method -- hence grouping by prefix and using min.
                def best_of(prefix):
                    c = [r for (b2, t2, k2, u2), r in taq.items()
                         if b2 == bk and t2 == t and u2 == bud
                         and k2.startswith(prefix)]
                    return min(c, key=lambda r: float(r["value"])) if c else None

                ti, tk = best_of("taq-is"), best_of("taq-kl")
                if not ti and not tk:
                    continue
                lo, hi = bracket(summary, bk, t, bud)
                best = min([float(x["value"]) for x in (ti, tk) if x], default=None)
                verdict = "-"
                if best is not None and hi is not None:
                    hv, hb = float(hi["value"]), float(hi["avg_bit"])
                    lv = float(lo["value"]) if lo else None
                    if best < hv:
                        verdict = f"TAQ wins (beats even the lower-bit TALQ upper point)"
                    elif lv is not None and best > lv:
                        verdict = f"TAQ loses (loses even to the TALQ lower point with fewer bits)"
                    else:
                        verdict = "between"
                def val(x):
                    return f"{float(x['value']):.4f}" if x else "-"

                def bit(x):
                    return f"{float(x['avg_bit']):.2f}" if x else "-"

                cfg = ""
                for x in (ti, tk):
                    if x and x["kind"].endswith("-p24"):
                        cfg = " *p24"
                print(f"{bk:9s}{t:5s}{val(ti):>9s}{val(tk):>9s} | "
                      f"{val(lo):>9s}{bit(lo):>6s}{val(hi):>9s}{bit(hi):>6s} | "
                      f"{verdict}{cfg}")


if __name__ == "__main__":
    main()
