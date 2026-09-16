"""Table 1 -- exactly the confirmed rule (2026-09-12). Produces the numbers to put into Template.tex.

Rule summary (user-confirmed):
  Budget   B1=3.6667(44/88 bits), B2=3.3333(40/80). Written as 3.67 / 3.33.
  TALQ     the **observed point** with the largest achieved bit among those with
           avg_bit <= B. Performance is not used for the selection, and there is
           no interpolation either. The row name is TALQ (<=B), the parenthesis
           is the actual mean bit width.
  Baseline TAQ-KL{3,4} is placed **exactly on the common cap B**.
           (It is not matched to TALQ's individual achieved bit.)
  ER       lambda is not picked per fold. A common lambda whose five-fold mean
           achieved bit width is under the cap is picked, and both performance
           and bit width use the fold mean. The baseline and uniform also use
           the fold mean.
  Aggregation  task value = arithmetic mean of the 5-backbone error. The TALQ
           parenthesis = mean achieved bits.
           Rel. GMean = exp( (1/6) * sum_t log( E_method,t / E_fp32,t ) ).
  Bold     the minimum among TAQ-KL / TALQ within each backend x cap. uniform/FP32 excluded.

Only hubertL/KS has a different task head (_superb_baldev). FP32/uniform/TALQ/baseline are
**all** read from that head's outputs -- if the heads are mixed within one cell, that cell's
comparison is invalid.

  python figures/table1.py            # table
  python figures/table1.py --latex    # tex rows verbatim
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import argparse
import csv
import glob
import json
import math
import os
import re
import statistics as st

from talq.paths import RESULTS_ROOT

BKS = ["w2v2", "wavlm", "hubert", "wavlmL", "hubertL"]
NL = {"w2v2": 12, "wavlm": 12, "hubert": 12, "wavlmL": 24, "hubertL": 24}
T6 = ["pr", "asr", "ks", "ic", "er", "asv"]
LBL = {"pr": "PR", "asr": "ASR", "ks": "KS", "ic": "IC", "er": "ER", "asv": "ASV"}
MET = {"pr": "PER", "asr": "WER", "ks": "KS_ERR", "ic": "IC_ERR",
       "er": "ER_ERR", "asv": "EER"}
FOLDS = [1, 2, 3, 4, 5]
CAPS = [3.6667, 3.3333]
NEWHEAD = {("hubertL", "ks")}


def snap(B, nl):
    """Snap the nominal B onto the layer grid. 3.3333 is a truncated decimal of
    10/3, so a plain comparison drops the point that sits exactly on the budget."""
    return round(B * nl) / nl


# ------------------------------------------------------------------ raw data

def fp32(bk, t):
    if (bk, t) in NEWHEAD:
        for r in json.load(open(f"{RESULTS_ROOT}/ks_head_hubertL.json"))["rows"]:
            if r["config"] == "fp32":
                return r["value"]
        return None
    if t == "er":
        try:
            return json.load(open(f"{RESULTS_ROOT}/fp32_ours.json"))[bk]["er"]
        except Exception:
            return None
    for x in csv.DictReader(open(f"{RESULTS_ROOT}/grid6.csv")):
        if x["backbone"] == bk and x["config"] == "fp32":
            try:
                return float(x[MET[t]])
            except Exception:
                return None
    return None


def uniform(q, bk, t, b):
    if (bk, t) in NEWHEAD:
        for r in json.load(open(f"{RESULTS_ROOT}/ks_head_hubertL.json"))["rows"]:
            if r["config"] == f"{q}-uniform{b}":
                return r["value"]
        return None
    if t == "er":                                  # fold mean
        v = []
        for f in FOLDS:
            try:
                v.append(json.load(open(
                    f"{RESULTS_ROOT}/sw_{q}_erf{f}/{bk}/base/{bk}_b3.json"
                ))["uniform"][str(b)]["er"])
            except Exception:
                pass
        return st.mean(v) if len(v) == len(FOLDS) else None
    try:
        return json.load(open(f"{RESULTS_ROOT}/sw_{q}/{bk}/base/{bk}_b3.json")
                         )["uniform"][str(b)][t]
    except Exception:
        return None


def talq_points(q, bk, t):
    """[(achieved bit, error)] -- observed points only. For ER, both are averaged over the fold-common λ."""
    if (bk, t) in NEWHEAD:
        pat = f"{RESULTS_ROOT}/sw_{q}_kshead/{bk}/l*/{bk}_b3.json"
    elif t == "er":
        per = {}
        for f in FOLDS:
            for fp in glob.glob(f"{RESULTS_ROOT}/sw_{q}_erf{f}/{bk}/l*/{bk}_b3.json"):
                lam = float(re.search(r"/l([0-9.]+)/", fp).group(1))
                d = json.load(open(fp))
                if "er" in d.get("avg", {}):
                    per.setdefault(lam, []).append((d["avg"]["er"], d["eval"]["er"]))
        return [(st.mean(b for b, _ in v), st.mean(e for _, e in v))
                for v in per.values() if len(v) == len(FOLDS)]
    else:
        pat = f"{RESULTS_ROOT}/sw_{q}/{bk}/l*/{bk}_b3.json"
    out = []
    for fp in glob.glob(pat):
        d = json.load(open(fp))
        if t in d.get("avg", {}):
            out.append((d["avg"][t], d["eval"][t]))
    return out


def talq(q, bk, t, B):
    """cap rule: the largest achieved bit among those <= B."""
    ok = [p for p in talq_points(q, bk, t) if p[0] <= snap(B, NL[bk]) + 1e-9]
    return max(ok, key=lambda p: p[0]) if ok else None


def sensdp(q, bk, t, B, bits="234"):
    d_ = "sensdp1024" if bits == "234" else "sensdp1024_b34"
    if t == "er":
        v = []
        for f in FOLDS:
            fp = (f"{RESULTS_ROOT}/{d_}/results/{q}_{bk}_er_B{B:g}"
                  f"_p3600_s0_b16_B{bits}_f{f}.json")
            try:
                d = json.load(open(fp))
                if d.get("status") == "ok":
                    v.append(d["value"])
            except Exception:
                pass
        return st.mean(v) if len(v) == len(FOLDS) else None
    fp = (f"{RESULTS_ROOT}/{d_}/results/{q}_{bk}_{t}_B{B:g}"
          f"_p3600_s0_b16_B{bits}.json")
    try:
        d = json.load(open(fp))
    except Exception:
        return None
    return d["value"] if d.get("status") == "ok" else None


def taq(q, bk, t, B):
    if (bk, t) in NEWHEAD:
        pats = [f"{RESULTS_ROOT}/taq3600_b34_kshead/eval_{q}_b{B:g}.csv"]
    elif t == "er":
        v = []
        for f in FOLDS:
            fp = f"{RESULTS_ROOT}/taq3600_b34_erf{f}/eval_{q}_b{B:g}.csv"
            if not os.path.exists(fp):
                continue
            for r in csv.DictReader(open(fp)):
                if (r["backbone"] == bk and r["task"] == "er"
                        and r["kind"].startswith("taq-kl")):
                    v.append(float(r["value"]))
        return st.mean(v) if len(v) == len(FOLDS) else None
    else:
        pats = [f"{RESULTS_ROOT}/taq3600_b34/eval_{q}_b{B:g}.csv"]
    for fp in pats:
        if not os.path.exists(fp):
            continue
        for r in csv.DictReader(open(fp)):
            if (r["backbone"] == bk and r["task"] == t
                    and r["kind"].startswith("taq-kl")):
                return float(r["value"])
    return None


# ------------------------------------------------------------------ aggregation

def row_mean(fn):
    """Arithmetic mean over the 5 backbones per task. None if even one is missing."""
    out = {}
    for t in T6:
        v = [x for bk in BKS if (x := fn(bk, t)) is not None]
        out[t] = st.mean(v) if len(v) == len(BKS) else None
    return out


def gmean(row, base):
    v = [math.log(row[t] / base[t]) for t in T6
         if row.get(t) and base.get(t)]
    return math.exp(st.mean(v)) if len(v) == len(T6) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true")
    a = ap.parse_args()

    base = row_mean(fp32)
    rows = [("--", "FP32", base, None)]
    for q in ("gptq", "awq"):
        rows.append((q.upper(), "Uniform-4", row_mean(lambda b, t: uniform(q, b, t, 4)), None))
        for B in CAPS:
            tag = f"{B:.2f}"
            rows.append((q.upper(), f"TAQ-KL-{tag}",
                         row_mean(lambda b, t: taq(q, b, t, B)), None))
            err = row_mean(lambda b, t: (x[1] if (x := talq(q, b, t, B)) else None))
            bit = row_mean(lambda b, t: (x[0] if (x := talq(q, b, t, B)) else None))
            rows.append((q.upper(), f"TALQ (<={tag})", err, bit))
        rows.append((q.upper(), "Uniform-3", row_mean(lambda b, t: uniform(q, b, t, 3)), None))

    # Bold: the minimum among TAQ/Sens/TALQ within backend x cap
    groups = {}
    for i, (q, name, r, _) in enumerate(rows):
        for B in CAPS:
            if f"{B:.2f}" in name:
                groups.setdefault((q, f"{B:.2f}"), []).append(i)
    bold = {}
    for key, idx in groups.items():
        for t in T6 + ["gm"]:
            vals = [(rows[i][2][t] if t != "gm" else gmean(rows[i][2], base), i)
                    for i in idx if (rows[i][2].get(t) if t != "gm" else True)]
            vals = [(v, i) for v, i in vals if v is not None]
            if not vals:
                continue
            lo = min(v for v, _ in vals)
            for v, i in vals:
                if round(v, 4) == round(lo, 4):
                    bold[(i, t)] = True

    print(f"{'Backend':8}{'Allocation':17}"
          + "".join(f"{LBL[t]:>16}" for t in T6) + f"{'Rel.GMean':>10}")
    for i, (q, name, r, bit) in enumerate(rows):
        cells = []
        for t in T6:
            if r.get(t) is None:
                cells.append(f"{'--':>16}"); continue
            s = f"{r[t]:.4f}" + (f" ({bit[t]:.2f})" if bit and bit.get(t) else "")
            cells.append(f"{'*' if bold.get((i, t)) else ''}{s:>{16 - (1 if bold.get((i,t)) else 0)}}")
        g = gmean(r, base)
        gs = f"{g:.3f}" if g else "--"
        print(f"{q:8}{name:17}" + "".join(cells)
              + f"{('*' if bold.get((i,'gm')) else '') + gs:>10}")
    print("\n* = the minimum within that backend x cap group (TAQ-KL / TALQ)")

    if a.latex:
        print("\n% ---- tex rows ----")
        for i, (q, name, r, bit) in enumerate(rows):
            def f(t):
                if r.get(t) is None:
                    return "--"
                s = f"{r[t]:.4f}" + (f" ({bit[t]:.2f})" if bit and bit.get(t) else "")
                return f"\\textbf{{{s}}}" if bold.get((i, t)) else s
            g = gmean(r, base)
            gs = (f"\\textbf{{{g:.3f}}}" if bold.get((i, "gm")) and g
                  else (f"{g:.3f}" if g else "--"))
            nm = name.replace("<=", "$\\leq$")
            print(f"  {q} & {nm}\n  & " + " & ".join(f(t) for t in T6)
                  + f" & {gs} \\\\")


if __name__ == "__main__":
    main()
