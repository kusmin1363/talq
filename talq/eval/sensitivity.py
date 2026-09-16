"""Per-layer x per-bit sensitivity map -- official SUPERB full-split evaluation, raw metric values.

Replaces the earlier sensitivity sweep. Three changes:

1. **Official full-split evaluation.** The old version measured coarsely, with
   --cap 400 / --n-ctc 32 / --sv-stride 16 (the stated reason was that this is a
   design input). So there was no telling whether a value like "KS +4,746% at L3"
   was real or 400-sample noise. Here the same evaluation protocol as talq.eval.grid_eval is
   used -- pr/asr all 2,608, ks 3,081, ic 3,793, asv the official 37,720 trial pairs.

2. **Raw metric values are recorded.** Relative degradation % is blown up to
   hundreds of percent by noise when the fp32 baseline is small (ks 0.038). Store
   the raw values and let the reader do the normalization.
   The fp32 baselines are already there, in the config=fp32 rows of
   results/grid_official_{bk}.csv.

3. **6 tasks: PR / ASR / KS / IC / ER / ASV.** Only SID is excluded (the official
   iden_split probe is stuck at 0.26~0.38 on the three base backbones -- short of
   the official level of 0.186, so it is not usable yet). These four were confirmed
   to agree with the official SUPERB evaluation (PR/ASR have no crop to begin with,
   every KS clip is exactly 1 second, and IC tops out at 5.29 s so it never hits the
   6 s cap -> the crop code never runs).
   Why the other three are missing:
     * ASV -- our eval_sv does an 8 s head crop, but SUPERB does not crop at
       evaluation time (s3prl sv_voxceleb1/expert.py: max_timestep is only in
       train_config). 34% of the evaluation files were being cut, throwing away 23%
       of the audio. To be re-measured without cropping.
     * ER / SID -- probes are being retrained (ER 5-fold, SID iden_split 1,251
       speakers). Both were also non-deterministic, because a random crop was
       applied at evaluation time (ER 24%, SID 30%).

The quantizer is fixed to GPTQ. quants/{bk}_L{k}_b{b}.pt is **isolated quantization**
(fitted on the activations with every other layer in fp32), so its conditions match
this design -- which loads a single layer -- exactly. Recomputing on the fly gives
the same values for the same calib, so the cache is used.
RTN/AWQ comparisons are run separately by changing --quants (to check whether the
sensitivity depends on the quantizer).

Layers can be split for parallelism. wavlmL is 72 configs, 22 hours on its own:
  python -m talq.eval.sensitivity --backbone wavlmL --layers 0-11
  python -m talq.eval.sensitivity --backbone wavlmL --layers 12-23
"""
import argparse, csv, os, time
import torch

import torch.nn.functional as F

from talq.paths import PROBES, PROBES_SUPERB, QUANT_GPTQ, RESULTS_ROOT

from talq.eval import arm_eval as AE
from talq.eval.grid_eval import official_eval, setup, BIG
from talq.eval.grid_eval import METRIC as _METRIC_5

from talq.eval.probe_train import (hidden_states, load_wav, eval_ctc, eer,
                                   iemocap_index, IEMOCAP_MANIFEST)

TASKS = ["pr", "asr", "ks", "ic", "er", "asv"]   # reason for excluding SID is in the docstring
METRIC = {**_METRIC_5, "er": "ER_ERR"}   # talq.eval.grid_eval has no er

IC_SLOTS = (6, 14, 4)
AUX = ["KL_pr", "KL_asr", "KL_ks", "KL_ic", "KL_er", "COS_asv", "COS_hidden"]
# Fixed sample size for the auxiliary metrics. KL produces a value per frame, so its
# variance is very small and the full split is not needed. Cut from the front,
# deterministically (not at random).
AUX_N = 200
AUX_MAX_S = {"pr": 8.0, "asr": 8.0, "ks": 1.0, "ic": 6.0, "er": 8.0, "asv": 8.0}
PROBES_V2 = PROBES             # pr/asr/ks/ic/asv
PROBES_SB = PROBES_SUPERB      # er fold1~5 (SUPERB 5-fold)
FOLDS = [1, 2, 3, 4, 5]


def load_probes6(hf_id, dev):
    """pr/asr/ks/ic/asv come from PROBES_V2; er reads the five folds in PROBES_SB
    separately. talq.eval.arm_eval.load_probes only looks for a single {tag}_{task}.pt, so er
    is built here directly."""
    base = AE.load_probes(hf_id, PROBES_V2, dev,
                          tasks=["pr", "asr", "ks", "ic", "asv"])
    tag = hf_id.split("/")[-1]
    er = {}
    for n in FOLDS:
        fp = f"{PROBES_SB}/{tag}_er_fold{n}.pt"
        if os.path.exists(fp):
            er[n] = AE.build_probe("er", torch.load(fp, map_location=dev,
                                                    weights_only=False), dev)
    return base, er


@torch.no_grad()
def clf_err_nocrop(m, probe, items, dev, slots=None):
    """Same as talq.eval.arm_eval._clf_err, except it **does not crop**.

    _clf_err uses load_wav(f, max_s=...), and that is a random.randint random crop,
    so (a) measuring the same model twice gives different values and (b) it differs
    from SUPERB too. s3prl does not crop at evaluation time -- max_timestep goes
    into train_config only (checked in voxceleb1/sv_voxceleb1 expert.py). 24% of
    IEMOCAP is over 6 s, so this was a particular problem for ER."""
    wrong = 0
    for f, y in items:
        hs = hidden_states(m, torch.tensor(load_wav(f))[None].to(dev)).float()
        lg = probe(hs)
        if slots:
            o, ok = 0, True
            for i, sz in enumerate(slots):
                ok &= int(lg[0, o:o + sz].argmax()) == y[i]; o += sz
            wrong += not ok
        else:
            wrong += int(lg.argmax(-1)[0]) != y
    return wrong / max(len(items), 1)


@torch.no_grad()
def eval_sv_nocrop(m, probe, root, trials, dev):
    """The version of eval_sv with the 8 s head crop taken out. 34% of the
    evaluation files were being cut, throwing away 23% of the audio (measured)."""
    files = sorted({p for _, a, b in trials for p in (a, b)})
    embs = {}
    for f in files:
        x = torch.tensor(load_wav(os.path.join(root, "wav", f)))[None].to(dev)
        embs[f] = F.normalize(probe.embed(hidden_states(m, x).float()), dim=-1)[0]
    sc = [float(torch.dot(embs[a], embs[b])) for _, a, b in trials]
    return eer(sc, [l for l, _, _ in trials])


def eval6(m, probes, er_probes, d, dev):
    """Full-split evaluation of the 6 tasks. ER is measured per fold on its own test
    session and averaged (the SUPERB way)."""
    r = {}
    for t in ("pr", "asr"):
        if t in probes:
            items, itos, sep = d[t]
            r[t] = eval_ctc(m, probes[t], items, dev, itos, sep, n=BIG)
    if "ks" in probes:
        r["ks"] = clf_err_nocrop(m, probes["ks"], d["ks"], dev)
    if "ic" in probes:
        r["ic"] = clf_err_nocrop(m, probes["ic"], d["ic"], dev, slots=IC_SLOTS)
    if er_probes:
        v = [clf_err_nocrop(m, er_probes[n], d["er"][n], dev) for n in sorted(er_probes)]
        r["er"] = sum(v) / len(v)
    if "asv" in probes:
        tri, root = d["asv"]
        r["asv"] = eval_sv_nocrop(m, probes["asv"], root, tri, dev)
    return r


def load_wav_head(path, max_s):
    """**Deterministic** load that cuts from the front. talq.eval.probe_train.load_wav(max_s=...)
    does a random crop with random.randint, so it must never be used for the
    auxiliary metrics -- the reference (p) and the comparison (q) would look at
    different segments, and KL would end up measuring an audio difference.
    eval_sv does its own head crop for the same reason."""
    w = load_wav(path)
    n = int(max_s * 16000)
    return w[:n] if len(w) > n else w


def aux_sample(d):
    """(file list) per task. Uses the front of the evaluation set, deterministically."""
    return {"pr": [f for f, _, _ in d["pr"][0][:AUX_N]],
            "asr": [f for f, _, _ in d["asr"][0][:AUX_N]],
            "ks": [f for f, _ in d["ks"][:AUX_N]],
            "ic": [f for f, _ in d["ic"][:AUX_N]],
            # ER has a different test session per fold, so take the first 40 of each fold (200 total).
            "er": {n: [f for f, _ in d["er"][n][:AUX_N // len(FOLDS)]] for n in FOLDS},
            "asv": sorted({p for _, a2, b2 in d["asv"][0][:AUX_N]
                           for p in (a2, b2)})[:AUX_N]}


@torch.no_grad()
def aux_forward(m, probes, er_probes, sample, asv_root, dev):
    """Pulls out every output needed to compute the auxiliary metrics.

    Called once on fp32 it gives the reference (p); called on the quantized model it
    gives the comparison target (q). The logits are pre-softmax values, so KL is
    computed safely with log_softmax."""
    out = {}
    for t in ("pr", "asr", "ks", "ic"):
        if t not in probes:
            continue
        lg = []
        for f in sample[t]:
            x = torch.tensor(load_wav_head(f, AUX_MAX_S[t]))[None].to(dev)
            hs = hidden_states(m, x).float()
            lg.append(probes[t](hs)[0].detach())      # pr/asr: (T,C)  ks/ic: (C,)
            if t == "pr":                             # hidden cos reuses the same forward
                out.setdefault("_hid", []).append(hs[-1, 0].detach())
        out[t] = lg
    if er_probes:                                     # each fold with its own probe
        lg = []
        for n in sorted(er_probes):
            for f in sample["er"][n]:
                x = torch.tensor(load_wav_head(f, AUX_MAX_S["er"]))[None].to(dev)
                lg.append(er_probes[n](hidden_states(m, x).float())[0].detach())
        out["er"] = lg
    if "asv" in probes:
        emb = []
        for f in sample["asv"]:
            x = torch.tensor(load_wav_head(os.path.join(asv_root, "wav", f),
                                           AUX_MAX_S["asv"]))[None].to(dev)
            emb.append(probes["asv"].embed(hidden_states(m, x).float())[0].detach())
        out["asv"] = emb
    return out


def _kl(p_lg, q_lg):
    """KL(p||q). p is the fp32 reference. If there is a frame axis, averaged over frames."""
    p = F.log_softmax(p_lg.float(), -1)
    q = F.log_softmax(q_lg.float(), -1)
    return float((p.exp() * (p - q)).sum(-1).mean())


def aux_compare(ref, cur):
    """Compares the fp32 reference ref against the quantized cur to produce the 6 auxiliary metrics."""
    r = {}
    for t in ("pr", "asr", "ks", "er"):
        if t in ref and t in cur:
            r[f"KL_{t}"] = round(
                sum(_kl(a, b) for a, b in zip(ref[t], cur[t])) / len(ref[t]), 6)
    if "ic" in ref and "ic" in cur:                   # measure the 3 slots separately and sum
        tot, n = 0.0, 0
        for a, b in zip(ref["ic"], cur["ic"]):
            o = 0
            for sz in IC_SLOTS:
                tot += _kl(a[o:o + sz], b[o:o + sz]); o += sz
            n += 1
        r["KL_ic"] = round(tot / n, 6)
    if "asv" in ref and "asv" in cur:                 # an embedding is not a distribution -- cosine
        r["COS_asv"] = round(1 - sum(
            float(F.cosine_similarity(a[None], b[None])) for a, b in
            zip(ref["asv"], cur["asv"])) / len(ref["asv"]), 6)
    if "_hid" in ref and "_hid" in cur:               # task-independent representation distortion
        v = [float(F.cosine_similarity(a, b, dim=-1).mean())
             for a, b in zip(ref["_hid"], cur["_hid"])]
        r["COS_hidden"] = round(1 - sum(v) / len(v), 6)
    return r




def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--bits", default="4,3,2")
    ap.add_argument("--layers", default="", help="'0-11' form. all layers if empty")
    ap.add_argument("--quants", default=f"{QUANT_GPTQ}")
    ap.add_argument("--csv", default="")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    dev = "cuda"
    bk = a.backbone
    bits = [int(x) for x in a.bits.split(",")]

    m, _unused, orig = setup(bk, dev)
    import talq.quant.task_fisher as _T
    probes, er_probes = load_probes6(_T.BACKBONES[bk], dev)
    nl = m.config.num_hidden_layers
    if a.layers:
        lo, hi = (int(x) for x in a.layers.split("-"))
        layers = list(range(lo, min(hi, nl - 1) + 1))
    else:
        layers = list(range(nl))
    tag = f"_L{layers[0]}-{layers[-1]}" if a.layers else ""
    csv_path = a.csv or f"{RESULTS_ROOT}/sens_official_{bk}{tag}.csv"

    d = official_eval(cap=40 if a.smoke else None)
    cap = 40 if a.smoke else None
    d["er"] = {n: iemocap_index(IEMOCAP_MANIFEST, {f"Session{n}"})[:cap] if cap
               else iemocap_index(IEMOCAP_MANIFEST, {f"Session{n}"}) for n in FOLDS}
    smp = aux_sample(d)
    log("  caching fp32 reference outputs (for the auxiliary metrics)...")
    t_ref = time.time()
    ref = aux_forward(m, probes, er_probes, smp, d["asv"][1], dev)
    log(f"  cache done {time.time()-t_ref:.0f}s  sample "
        + " ".join(f"{k}:{len(v)}" for k, v in smp.items()))
    log(f"[{bk}] {layers[0]}~{layers[-1]} of {nl} layers x bits {bits}  "
        f"probe {sorted(probes)} + er fold {sorted(er_probes)}")
    log(f"  eval set pr {len(d['pr'][0])} asr {len(d['asr'][0])} ks {len(d['ks'])} "
        f"ic {len(d['ic'])} er {sum(len(v) for v in d['er'].values())} "
        f"asv {len(d['asv'][0])} pairs")

    done = set()
    fresh = not os.path.exists(csv_path)
    if not fresh:
        for r in csv.DictReader(open(csv_path)):
            done.add((int(r["layer"]), int(r["bits"])))
    fh = open(csv_path, "a", newline="")
    fields = ["backbone", "layer", "bits"] + [METRIC[t] for t in TASKS] + AUX + ["sec"]
    w = csv.DictWriter(fh, fieldnames=fields)
    if fresh:
        w.writeheader(); fh.flush()

    for k in layers:
        for b in bits:
            if (k, b) in done:
                log(f"  [skip] L{k} b{b}"); continue
            fp = f"{a.quants}/{bk}_L{k}_b{b}.pt"
            if not os.path.exists(fp):
                log(f"  [missing] {fp}"); continue
            t0 = time.time()
            m.load_state_dict(orig)                       # after reverting to fp32
            q = torch.load(fp, map_location="cpu", weights_only=False)
            assert q["layer"] == k and q["bits"] == b, \
                f"{fp}: layer/bits mismatch {q['layer']}/{q['bits']} != {k}/{b}"
            m.load_state_dict({kk: v.to(dev) for kk, v in q["weights"].items()},
                              strict=False)              # replace layer k only
            res = eval6(m, probes, er_probes, d, dev)
            aux = aux_compare(ref, aux_forward(m, probes, er_probes, smp,
                                               d["asv"][1], dev))
            row = {"backbone": bk, "layer": k, "bits": b,
                   "sec": round(time.time() - t0)}
            row.update({METRIC[t]: round(res[t], 4) for t in TASKS if t in res})
            row.update(aux)
            w.writerow(row); fh.flush()
            log(f"  L{k:>2} b{b}  " + "  ".join(f"{METRIC[t]}={res[t]:.4f}"
                                                for t in TASKS if t in res)
                + "  | " + " ".join(f"{k2}={v:.4g}" for k2, v in aux.items())
                + f"  ({row['sec']}s)")
    fh.close()
    print(f"SENS_OFFICIAL_DONE {bk}{tag}", flush=True)


if __name__ == "__main__":
    main()
