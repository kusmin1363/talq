"""Calibration audio loading and the WER helper.

The rest of the repository imports this module for four things -- load_calib (the
fixed calibration tensor), load_emilia (raw Emilia EN clips), crop (pad/crop to the
calibration length) and wer.

Run as a script it is also the standalone report it started out as: 4-bit GPTQ on
wav2vec2 with real Emilia (EN) speech, reporting the four things asked for on a
handful of Emilia clips:
  1. fidelity  -- encoder cosine / SQNR, fp32 vs int4
  2. quality   -- WER (fp32 vs int4 vs ground truth) + fp32<->int4 agreement
  3. size      -- packed int4 bytes vs fp32 (exact: packing is deterministic)
  4. speed     -- measured fp32 vs fake-quant latency (+ honest note on real int4)

Emilia is streamed straight from .tar with stdlib tarfile (no webdataset dep);
mp3 decoded via soundfile, linear-resampled 32k->16k.
"""
import io
import json
import os
import re
import tarfile
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

from talq.paths import EMILIA_EN, RESULTS_ROOT
from talq.quant.gptq import quantize_layer, encoder_repr, cosine, sqnr_db

SR = 16000
CLIP_S = 2.0
L = int(CLIP_S * SR)


def load_emilia(tar_paths, n, seed=0, with_speaker=False):
    """Up to n (full-length waveform, text[, speaker]) tuples from Emilia EN tars.

    Full length is kept: cropping is a calib-only concern (done in main), whereas
    WER needs the whole utterance -- a 2s crop vs the full transcript inflates WER
    with words that were simply cut out of the audio. with_speaker also returns the
    json "speaker" id (for speaker-separability metrics).
    """
    out = []
    texts, spks = {}, {}   # basename -> transcript / speaker, across all shards
    for tp in tar_paths:
        with tarfile.open(tp) as tar:
            for m in tar:
                base, ext = os.path.splitext(m.name)
                if ext == ".json":
                    j = json.load(tar.extractfile(m))
                    texts[base] = j["text"]
                    spks[base] = j.get("speaker", "")
                elif ext == ".mp3":
                    w, fsr = sf.read(io.BytesIO(tar.extractfile(m).read()), dtype="float32")
                    if w.ndim > 1:
                        w = w.mean(1)
                    if fsr != SR:  # linear resample -- fine for a fidelity/WER probe
                        w = np.interp(np.linspace(0, len(w) - 1, int(len(w) * SR / fsr)),
                                      np.arange(len(w)), w).astype("float32")
                    w = (w - w.mean()) / np.sqrt(w.var() + 1e-7)  # wav2vec2 norm
                    out.append([base, w])
                if len(out) >= n:
                    break
        if len(out) >= n:
            break
    if with_speaker:
        return [(torch.tensor(w), texts.get(b, ""), spks.get(b, "")) for b, w in out[:n]]
    return [(torch.tensor(w), texts.get(b, "")) for b, w in out[:n]]


def crop(w):
    """Pad/crop a waveform to the fixed calibration length L."""
    return torch.nn.functional.pad(w, (0, max(0, L - len(w))))[:L]


def load_calib(calib_pt, emilia_dir, dev, n=48):
    """The fixed n-clip, 2s calibration tensor (deterministic: first n Emilia-EN
    mp3s). Prefers a saved .pt (~6MB, portable) over re-reading the Emilia tars
    (~GB) -- identical values either way, so results are unchanged."""
    import glob
    if calib_pt and os.path.exists(calib_pt):
        return torch.load(calib_pt, map_location=dev)
    shards = sorted(glob.glob(os.path.join(emilia_dir, "*.tar")))[:2]
    return torch.stack([crop(w) for w, _ in load_emilia(shards, n)]).to(dev)


def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z' ]+", " ", t.upper())).strip()


def wer(ref, hyp):
    r, h = norm(ref).split(), norm(hyp).split()
    if not r:
        return 0.0 if not h else 1.0
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
    return d[-1] / len(r)


def packed_size(model, group=128, bits=4):
    """Exact bytes: fp32 weights vs int4 codes + fp16 per-group scale/zero."""
    fp32 = pk = nweights = 0
    for _, m in model.wav2vec2.encoder.named_modules():
        if isinstance(m, nn.Linear):
            o, i = m.weight.shape
            ng = -(-i // group)             # ceil
            fp32 += o * i * 4
            pk += o * i * bits / 8          # 4-bit codes
            pk += o * ng * 2 * 2            # fp16 scale + fp16 zero per (row, group)
            nweights += o * i
    return fp32, pk, nweights


@torch.no_grad()
def transcribe(model, proc, x):
    ids = model(x).logits.argmax(-1)
    return proc.batch_decode(ids)


def timeit(model, x, reps=5):
    with torch.no_grad():
        model(x)  # warmup
        t = time.time()
        for _ in range(reps):
            model(x)
    return (time.time() - t) / reps


def main():
    import argparse
    from transformers import Wav2Vec2ForCTC, AutoProcessor
    torch.manual_seed(0)

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/wav2vec2-base-960h")
    ap.add_argument("--emilia", default=f"{EMILIA_EN}")
    ap.add_argument("--nshards", type=int, default=2)
    ap.add_argument("--ncalib", type=int, default=48)
    ap.add_argument("--neval", type=int, default=8)
    ap.add_argument("--out", default=f"{RESULTS_ROOT}/emilia_report.json")
    a = ap.parse_args()

    import glob
    shards = sorted(glob.glob(os.path.join(a.emilia, "*.tar")))[:a.nshards]
    pairs = load_emilia(shards, a.ncalib + a.neval)
    assert len(pairs) >= a.ncalib + a.neval, f"got only {len(pairs)} clips"
    calib = torch.stack([crop(w) for w, _ in pairs[:a.ncalib]])  # 2s crops: H full-rank
    eval_pairs = pairs[a.ncalib:a.ncalib + a.neval]              # full length: honest WER

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained(a.model)
    model = Wav2Vec2ForCTC.from_pretrained(a.model, dtype=torch.float32).eval().to(dev)
    calib = calib.to(dev)
    eval_x = [w[None].to(dev) for w, _ in eval_pairs]   # each clip its own (1, len) batch
    eval_txt = [t for _, t in eval_pairs]

    # --- fp32 reference (before quantization), per full-length clip ---
    ref_repr = [encoder_repr(model, x) for x in eval_x]
    ref_txt = [transcribe(model, proc, x)[0] for x in eval_x]
    t_fp32 = timeit(model, eval_x[0])   # latency on one representative full clip
    fp32_bytes, pk_bytes, nweights = packed_size(model)

    # --- quantize encoder in place (sequential GPTQ) ---
    for i, layer in enumerate(model.wav2vec2.encoder.layers):
        quantize_layer(model, layer, calib, bits=4, group=128)
        c = np.mean([cosine(r, encoder_repr(model, x)) for r, x in zip(ref_repr, eval_x)])
        print(f"[layer {i:2d}] cosine={c:.4f}", flush=True)

    q_repr = [encoder_repr(model, x) for x in eval_x]
    q_txt = [transcribe(model, proc, x)[0] for x in eval_x]
    t_fakeq = timeit(model, eval_x[0])   # fake-quant runs at fp32 speed by construction

    # --- metrics (averaged over eval clips) ---
    wer_fp32 = np.mean([wer(r, h) for r, h in zip(eval_txt, ref_txt)])
    wer_int4 = np.mean([wer(r, h) for r, h in zip(eval_txt, q_txt)])
    wer_self = np.mean([wer(r, h) for r, h in zip(ref_txt, q_txt)])  # fp32<->int4 drift
    cos = np.mean([cosine(r, q) for r, q in zip(ref_repr, q_repr)])
    snr = np.mean([sqnr_db(r, q) for r, q in zip(ref_repr, q_repr)])

    rep = {
        "clips": {"calib": a.ncalib, "eval": a.neval, "shards": [os.path.basename(s) for s in shards]},
        "fidelity": {"encoder_cosine": float(cos), "encoder_sqnr_db": float(snr)},
        "wer": {"fp32_vs_gt": float(wer_fp32), "int4_vs_gt": float(wer_int4),
                "int4_vs_fp32": float(wer_self)},
        "size": {"encoder_weights": nweights,
                 "fp32_MB": fp32_bytes / 1e6, "int4_packed_MB": pk_bytes / 1e6,
                 "compression_x": fp32_bytes / pk_bytes,
                 "bits_per_weight": pk_bytes * 8 / nweights},
        "speed": {"device": dev, "fp32_s": t_fp32, "fakequant_s": t_fakeq,
                  "note": "fake-quant == fp32 speed by design; real int4 speedup "
                          "(memory-bound, ~compression_x ceiling) needs a packed "
                          "int4 kernel, which no speech-encoder runtime provides "
                          "-- that gap is the research target"},
    }
    print("\n" + json.dumps(rep, indent=2), flush=True)
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=2)
    print(f"\nsaved -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
