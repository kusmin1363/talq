"""Phase-2 sweep: score every precomputed quant variant with the frozen probes.

For each (backbone, quant-variant) we load the fp32 backbone, swap in the
variant's quantized Linear weights (strict=False), then measure all 4 judges:

  PER  <backbone>_pr  probe   LibriSpeech dev-clean (n=64, same as training eval)
  WER  <backbone>_asr probe   LibriSpeech dev-clean (n=64)
  EER  <backbone>_sv  probe   VoxCeleb1 veri_test2  (--sv-stride subsample)
  CKA  training-free          last-hidden fp32 vs quantized, over the PR eval set

One CSV row per condition; 4 metric columns. Conditions:
  base    layer=NONE bits=32  (fp32, no swap)  -> reference row, CKA==1
  uniform layer=ALL  bits b   (all layers quantized)
  isolated layer=k   bits b   (one layer quantized, rest fp32)

Resumable: rows already in the CSV are skipped, so a timed-out job just reruns.
The probes are frozen -- swapping backbone weights never touches them.
"""
import argparse
import csv
import glob
import os

import numpy as np
import torch

from talq.eval.probe_train import (
    BLSTMCTC, CHARS, LinearCTC, SVProbe, cka, eval_ctc, eval_sv,
    hidden_states, librispeech_index, load_lexicon, load_wav,
)
from talq.paths import DATA_ROOT, PROBES, QUANT_GPTQ, RESULTS_ROOT

BACKBONES = {"w2v2": "facebook/wav2vec2-base", "wavlm": "microsoft/wavlm-base",
             "w2v2960h": "facebook/wav2vec2-base-960h",
             "hubert": "facebook/hubert-base-ls960",
             "wavlmL": "microsoft/wavlm-large",
             "hubertL": "facebook/hubert-large-ll60k"}
TASKS = ["pr", "asr", "sv"]
FIELDS = ["backbone", "layer", "bits", "PER", "WER", "EER", "CKA"]


def build_probe(task, ck, dev):
    """Reconstruct a frozen probe from its checkpoint dict."""
    nl, dim = ck["n_layers"], ck["dim"]
    if task == "pr":
        p = LinearCTC(nl, dim, len(ck["itos"]) + 1)
    elif task == "asr":
        p = BLSTMCTC(nl, dim, len(ck["itos"]) + 1)
    else:
        p = SVProbe(nl, dim, ck["n_spk"])
    p.load_state_dict(ck["state"])
    return p.eval().to(dev).requires_grad_(False)


def last_hidden_cat(backbone, items, dev, n):
    """Concatenated last-layer frames over n eval clips -> (frames, dim) for CKA."""
    reps = []
    for f, _, _ in items[:n]:
        x = torch.tensor(load_wav(f))[None].to(dev)
        reps.append(hidden_states(backbone, x)[-1, 0])  # (T, D)
    return torch.cat(reps, 0)


def conditions(out_dir):
    """Yield (backbone_key, layer, bits, weights_or_None). base first per backbone."""
    for bk in BACKBONES:
        yield bk, "NONE", 32, None
        for path in sorted(glob.glob(f"{out_dir}/{bk}_*.pt")):
            d = torch.load(path, map_location="cpu")
            yield bk, str(d["layer"]), d["bits"], d["weights"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quants", default=QUANT_GPTQ)
    ap.add_argument("--probes", default=PROBES)
    ap.add_argument("--data", default=DATA_ROOT)
    ap.add_argument("--csv", default=RESULTS_ROOT / "sweep_results.csv")
    ap.add_argument("--backbone",
                    choices=["w2v2", "wavlm", "w2v2960h", "hubert", "both"],
                    default="both")
    ap.add_argument("--n-ctc", type=int, default=64, help="PR/WER eval clips")
    ap.add_argument("--sv-stride", type=int, default=4,
                    help="veri_test2 trial subsample (1 = full 37.6k, ~8min/cond)")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.n_ctc, a.sv_stride = 4, 512
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # rows already scored -> skip (resume). key = (backbone, layer, bits)
    done = set()
    if os.path.exists(a.csv):
        for r in csv.DictReader(open(a.csv)):
            done.add((r["backbone"], r["layer"], int(r["bits"])))
    fh = open(a.csv, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if not done:
        w.writeheader(); fh.flush()

    from transformers import AutoModel
    lex, phones = load_lexicon()
    pr_items = librispeech_index(f"{a.data}/librispeech/LibriSpeech/dev-clean", "pr", lex)
    asr_items = librispeech_index(f"{a.data}/librispeech/LibriSpeech/dev-clean", "asr", lex)
    trials = [(int(l), p1, p2) for l, p1, p2 in
              (ln.split() for ln in open(f"{a.data}/voxceleb1/veri_test2.txt"))][::a.sv_stride]
    sv_dir = f"{a.data}/voxceleb1/test"

    cache = {}  # backbone_key -> (model, orig_state, probes, ref_last_hidden)
    for bk, layer, bits, weights in conditions(a.quants):
        if a.backbone != "both" and bk != a.backbone:
            continue
        if (bk, layer, bits) in done:
            print(f"skip {bk} L{layer} b{bits}", flush=True)
            continue
        if bk not in cache:
            m = AutoModel.from_pretrained(BACKBONES[bk], dtype=torch.float32).eval().to(dev)
            m.requires_grad_(False)
            probes = {t: build_probe(t, torch.load(
                f"{a.probes}/{BACKBONES[bk].split('/')[-1]}_{t}.pt", map_location=dev), dev)
                for t in TASKS}
            ref = last_hidden_cat(m, pr_items, dev, a.n_ctc)
            cache[bk] = (m, {k: v.clone() for k, v in m.state_dict().items()}, probes, ref)
        m, orig, probes, ref = cache[bk]

        m.load_state_dict(orig)                     # reset to fp32
        if weights is not None:
            m.load_state_dict({k: v.to(dev) for k, v in weights.items()}, strict=False)

        per = eval_ctc(m, probes["pr"], pr_items, dev, phones, " ", n=a.n_ctc)
        wer_ = eval_ctc(m, probes["asr"], asr_items, dev, CHARS, "", n=a.n_ctc)
        eer_ = eval_sv(m, probes["sv"], sv_dir, trials, dev)
        cur = last_hidden_cat(m, pr_items, dev, a.n_ctc)
        ck = cka(ref, cur)

        row = {"backbone": bk, "layer": layer, "bits": bits,
               "PER": round(per, 4), "WER": round(wer_, 4),
               "EER": round(eer_, 4), "CKA": round(ck, 4)}
        w.writerow(row); fh.flush()
        print(f"{bk} L{layer} b{bits} | PER {per:.4f} WER {wer_:.4f} "
              f"EER {eer_:.4f} CKA {ck:.4f}", flush=True)
    fh.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
