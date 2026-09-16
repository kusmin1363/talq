"""SUPERB-style frozen-backbone probes: the fixed judges for the quantization sweep.

Pipeline (agreed): frozen fp32 SSL backbone -> train light head ONCE -> freeze head
-> sweeps quantize the backbone only and re-measure with the SAME head.

Judge panel:
  pr  : LibriSpeech tc-100, linear + CTC over phonemes (lexicon G2P)  -> PER
  asr : LibriSpeech tc-100, 2-layer BLSTM + CTC over chars            -> WER
  sv  : VoxCeleb1 dev, weighted-sum + mean-pool + proj + speaker CE   -> EER (veri_test2)
  cka : training-free linear CKA between fp32/quantized reps (used by the sweep)

The learnable softmax layer weights double as a diagnostic (which layers the task
uses) and make EVERY layer observable to every judge.
"""
import argparse
import glob
import json
import os
import random

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

from talq.eval.calib_io import wer
from talq.paths import DATA_ROOT, IEMOCAP_MANIFEST as _IEMOCAP, LIBRISPEECH_LEXICON

SR = 16000
CHARS = " '" + "".join(chr(c) for c in range(ord("A"), ord("Z") + 1))  # blank=0, 1..28
LEXICON = LIBRISPEECH_LEXICON
# emotion (SER, 3rd task): IEMOCAP 4-class standard, exc merged into hap.
IEMOCAP_MANIFEST = _IEMOCAP
EMO4 = {"neu": 0, "hap": 1, "exc": 1, "ang": 2, "sad": 3}  # {oth,fru,sur,...} dropped
EMO_TRAIN = {"Session1", "Session2", "Session3", "Session4"}  # spk-independent split
EMO_TEST = {"Session5"}


def load_lexicon(path=LEXICON):
    """word -> phone list (ARPAbet, stress digits stripped). Dictionary G2P:
    exact for LibriSpeech vocab; OOV utterances are simply dropped from PR."""
    lex = {}
    for line in open(path):
        parts = line.split()
        lex.setdefault(parts[0].upper(), [p.rstrip("0123456789") for p in parts[1:]])
    phones = sorted({p for ps in lex.values() for p in ps})
    return lex, phones


class LinearCTC(nn.Module):
    """PR head (SUPERB): weighted layer sum -> linear -> CTC."""

    def __init__(self, n_layers, dim, n_out):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n_layers))
        self.head = nn.Linear(dim, n_out)

    def fuse(self, hs):  # hs: (L, B, T, D)
        return (hs * torch.softmax(self.w, 0)[:, None, None, None]).sum(0)

    def forward(self, hs):
        return self.head(self.fuse(hs))


class BLSTMCTC(LinearCTC):
    """ASR head (SUPERB): weighted layer sum -> 2-layer BLSTM -> CTC."""

    def __init__(self, n_layers, dim, n_out, hidden=512):
        super().__init__(n_layers, dim, n_out)
        self.lstm = nn.LSTM(dim, hidden, num_layers=2, bidirectional=True,
                            batch_first=True)
        self.head = nn.Linear(2 * hidden, n_out)

    def forward(self, hs):
        return self.head(self.lstm(self.fuse(hs))[0])


class SVProbe(LinearCTC):
    """SID head: weighted sum -> mean-pool -> proj -> speaker CE; EER on proj emb."""

    def __init__(self, n_layers, dim, n_spk, emb=256):
        super().__init__(n_layers, dim, n_spk)
        self.proj = nn.Linear(dim, emb)
        self.head = nn.Linear(emb, n_spk)

    def embed(self, hs):
        return self.proj(self.fuse(hs).mean(1))

    def forward(self, hs):
        return self.head(self.embed(hs))


class SERProbe(LinearCTC):
    """SER head: weighted layer sum -> mean-pool -> linear -> emotion CE; error metric.
    Same shape as PR's linear head but pooled over time (utterance-level label)."""

    def forward(self, hs):
        return self.head(self.fuse(hs).mean(1))


@torch.no_grad()
def hidden_states(backbone, x):
    """(L, B, T, D) stacked hidden states; backbone frozen so no grad here."""
    return torch.stack(backbone(x, output_hidden_states=True).hidden_states)


def cka(x, y):
    """Linear CKA between (n, d) feature matrices. 1 = same representation up to
    rotation/scale; the training-free axis of the sweep (GPTQ's own objective,
    but occupancy-corrected unlike raw cosine)."""
    x = x - x.mean(0)
    y = y - y.mean(0)
    return ((x.T @ y).norm() ** 2 / ((x.T @ x).norm() * (y.T @ y).norm())).item()


def load_wav(path, max_s=None):
    w, fsr = sf.read(path, dtype="float32")
    if w.ndim > 1:
        w = w.mean(1)
    assert fsr == SR, f"{path}: {fsr} != {SR}"  # LibriSpeech/VoxCeleb are both 16k
    if max_s and len(w) > int(max_s * SR):
        o = random.randint(0, len(w) - int(max_s * SR))
        w = w[o:o + int(max_s * SR)]
    return (w - w.mean()) / np.sqrt(w.var() + 1e-7)


def librispeech_index(split_dir, task, lex=None, max_s=16.0, phones=None):
    """[(flac, ref_str, token_ids)]. asr: ref=text, tokens=chars.
    pr: ref=space-joined phones, tokens=phone ids; OOV utterances dropped."""
    items, dropped = [], 0
    if task == "pr":
        # If phones is passed, its symbol list/order is used. The official SUPERB
        # vocab keeps stress -> 71 symbols; the default (ours) strips it -> 39.
        if phones is None:
            _, phones = load_lexicon()
        p2i = {p: i + 1 for i, p in enumerate(phones)}
    for tr in glob.glob(os.path.join(split_dir, "*/*/*.trans.txt")):
        d = os.path.dirname(tr)
        for line in open(tr):
            uid, text = line.strip().split(" ", 1)
            f = os.path.join(d, uid + ".flac")
            if os.path.getsize(f) > max_s * SR * 2:  # flac ~<=2 bytes/sample bound
                continue
            if task == "asr":
                c2i = {c: i + 1 for i, c in enumerate(CHARS)}
                items.append((f, text, [c2i[c] for c in text.upper() if c in c2i]))
            else:
                ps = []
                for wd in text.upper().split():
                    if wd not in lex:
                        ps = None
                        break
                    ps += lex[wd]
                if ps is None:
                    dropped += 1
                    continue
                items.append((f, " ".join(ps), [p2i[p] for p in ps]))
    if dropped:
        print(f"pr: dropped {dropped} OOV utterances", flush=True)
    return items


def voxceleb_index(dev_dir):
    """[(wav, spk_idx)] + spk list from VoxCeleb1 dev (wav/idXXXXX/clip/*.wav)."""
    wavs = glob.glob(os.path.join(dev_dir, "wav", "id*", "*", "*.wav"))
    spks = sorted({p.split("/wav/")[1].split("/")[0] for p in wavs})
    s2i = {s: i for i, s in enumerate(spks)}
    return [(p, s2i[p.split("/wav/")[1].split("/")[0]]) for p in wavs], spks


def iemocap_index(manifest, splits):
    """[(wav, emo_idx)] over the given IEMOCAP sessions; 4-class (exc->hap)."""
    items = []
    for ln in open(manifest):
        d = json.loads(ln)
        if d["split"] in splits and d["emo_class"] in EMO4:
            items.append((d["audio"], EMO4[d["emo_class"]]))
    return items


def pad_batch(waves, dev):
    L = max(len(w) for w in waves)
    x = torch.zeros(len(waves), L)
    for i, w in enumerate(waves):
        x[i, :len(w)] = torch.tensor(w)
    return x.to(dev), torch.tensor([len(w) for w in waves])


def ctc_decode(ids, itos, sep):
    out, prev = [], 0
    for i in ids:
        if i != prev and i != 0:
            out.append(itos[i - 1])
        prev = i
    return sep.join(out)


def eer(scores, labels):
    o = np.argsort(scores)[::-1]
    lab = np.array(labels, dtype=float)[o]
    fnr = 1 - np.cumsum(lab) / lab.sum()
    fpr = np.cumsum(1 - lab) / (1 - lab).sum()
    i = np.argmin(np.abs(fnr - fpr))
    return float((fnr[i] + fpr[i]) / 2)


@torch.no_grad()
def eval_ctc(backbone, probe, items, dev, itos, sep, n=64):
    """WER for asr (char itos, sep=''), PER for pr (phone itos, sep=' ') --
    both are token-level Levenshtein via wer() on space-split refs."""
    errs = []
    for f, ref, _ in items[:n]:
        x = torch.tensor(load_wav(f))[None].to(dev)
        ids = probe(hidden_states(backbone, x)).argmax(-1)[0].tolist()
        errs.append(wer(ref, ctc_decode(ids, itos, sep)))
    return float(np.mean(errs))


@torch.no_grad()
def eval_sv(backbone, probe, test_dir, trials, dev, max_s=8.0):
    files = sorted({p for _, a, b in trials for p in (a, b)})
    embs = {}
    for f in files:
        w = load_wav(os.path.join(test_dir, "wav", f))
        w = w[:int(max_s * SR)]  # head crop: deterministic eval
        x = torch.tensor(w)[None].to(dev)
        embs[f] = F.normalize(probe.embed(hidden_states(backbone, x)), dim=-1)[0]
    scores = [float(torch.dot(embs[a], embs[b])) for _, a, b in trials]
    return eer(scores, [l for l, _, _ in trials])


@torch.no_grad()
def eval_ser(backbone, probe, items, dev, max_s=8.0):
    """Classification error (1 - weighted accuracy) over IEMOCAP test utterances.
    ponytail: WA not UA -- the sweep needs relative degradation, not balanced SOTA."""
    correct = 0
    for f, y in items:
        w = load_wav(f)[:int(max_s * SR)]
        x = torch.tensor(w)[None].to(dev)
        pred = probe(hidden_states(backbone, x)).argmax(-1)[0].item()
        correct += int(pred == y)
    return 1.0 - correct / len(items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, help="HF id or local path")
    ap.add_argument("--task", choices=["pr", "asr", "sv", "ser"], required=True)
    ap.add_argument("--data", default=DATA_ROOT)
    ap.add_argument("--out", required=True, help="probe checkpoint path (.pt)")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="continue from <out>.resume (login-node kill insurance)")
    a = ap.parse_args()
    if a.smoke:
        a.steps, a.eval_every = 60, 30
    random.seed(0); torch.manual_seed(0)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoModel
    backbone = AutoModel.from_pretrained(a.backbone, dtype=torch.float32).eval().to(dev)
    backbone.requires_grad_(False)
    n_layers = backbone.config.num_hidden_layers + 1
    dim = backbone.config.hidden_size

    itos = sep = trials = None
    if a.task in ("pr", "asr"):
        lex, phones = load_lexicon()
        itos, sep = (phones, " ") if a.task == "pr" else (CHARS, "")
        train = librispeech_index(f"{a.data}/librispeech/LibriSpeech/train-clean-100",
                                  a.task, lex)
        heldout = librispeech_index(f"{a.data}/librispeech/LibriSpeech/dev-clean",
                                    a.task, lex)
        probe = (LinearCTC(n_layers, dim, len(itos) + 1) if a.task == "pr"
                 else BLSTMCTC(n_layers, dim, len(itos) + 1)).to(dev)
    elif a.task == "sv":
        train, spks = voxceleb_index(f"{a.data}/voxceleb1/dev")
        trials = [(int(l), p1, p2) for l, p1, p2 in
                  (ln.split() for ln in open(f"{a.data}/voxceleb1/veri_test2.txt"))]
        probe = SVProbe(n_layers, dim, len(spks)).to(dev)
    else:  # ser
        train = iemocap_index(IEMOCAP_MANIFEST, EMO_TRAIN)
        heldout = iemocap_index(IEMOCAP_MANIFEST, EMO_TEST)
        probe = SERProbe(n_layers, dim, 4).to(dev)
    print(f"{a.task} | train items={len(train)} | layers={n_layers}", flush=True)

    opt = torch.optim.AdamW(probe.parameters(), lr=a.lr)
    best, start = float("inf"), 1
    rf = a.out + ".resume"
    if a.resume and os.path.exists(rf):
        r = torch.load(rf, map_location=dev)
        probe.load_state_dict(r["probe"]); opt.load_state_dict(r["opt"])
        start, best = r["step"] + 1, r["best"]
        print(f"resumed at step {start} (best={best:.4f})", flush=True)
    for step in range(start, a.steps + 1):
        batch = random.sample(train, a.bs)
        if a.task in ("pr", "asr"):
            x, lens = pad_batch([load_wav(f) for f, _, _ in batch], dev)
        else:  # sv / ser: (file, label); crop for batch memory
            ms = 3.0 if a.task == "sv" else 6.0
            x, lens = pad_batch([load_wav(f, max_s=ms) for f, _ in batch], dev)
        # bf16 backbone forward for TRAINING only (2-3x faster on A100). Evals,
        # best-ckpt selection, and the sweep all stay fp32; the head is robust
        # to the ~1e-3 feature delta.
        with torch.autocast("cuda", torch.bfloat16, enabled=dev == "cuda"):
            hs = hidden_states(backbone, x)
        hs = hs.float()
        if a.task in ("pr", "asr"):
            logits = probe(hs)
            in_lens = backbone._get_feat_extract_output_lengths(lens).to(torch.long)
            tgt = [torch.tensor(t) for _, _, t in batch]
            loss = F.ctc_loss(logits.log_softmax(-1).transpose(0, 1),
                              torch.cat(tgt).to(dev), in_lens,
                              torch.tensor([len(t) for t in tgt]),
                              blank=0, zero_infinity=True)
        else:
            loss = F.cross_entropy(probe(hs),
                                   torch.tensor([s for _, s in batch]).to(dev))
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(probe.parameters(), 5.0)  # CTC+LSTM spike insurance
        opt.step()

        if step % 200 == 0:  # heartbeat: cheap proof-of-life between evals
            print(f"[{step}] loss={loss.item():.3f}", flush=True)
        if step % 1000 == 0:  # atomic resume point (tmp+replace survives mid-write kill)
            torch.save({"probe": probe.state_dict(), "opt": opt.state_dict(),
                        "step": step, "best": best}, rf + ".tmp")
            os.replace(rf + ".tmp", rf)

        if step % a.eval_every == 0 or step == a.steps:
            probe.eval()
            # ponytail: 1/8 trial subsample during training (full EER ~8min/eval);
            # the sweep measures the frozen best probe on the FULL trial list.
            if a.task == "sv":
                m = eval_sv(backbone, probe, f"{a.data}/voxceleb1/test", trials[::8], dev)
            elif a.task == "ser":
                m = eval_ser(backbone, probe, heldout, dev)
            else:
                m = eval_ctc(backbone, probe, heldout, dev, itos, sep)
            probe.train()
            name = {"pr": "PER", "asr": "WER", "sv": "EER", "ser": "ERR"}[a.task]
            lw = torch.softmax(probe.w.detach(), 0).cpu().numpy().round(3).tolist()
            print(f"[{step}] loss={loss.item():.3f}  {name}={m:.4f}  layer_w={lw}",
                  flush=True)
            if m < best:
                best = m
                torch.save({"task": a.task, "backbone": a.backbone, "step": step,
                            "metric": m, "state": probe.state_dict(),
                            "n_layers": n_layers, "dim": dim, "itos": itos,
                            "n_spk": probe.head.out_features}, a.out)
    name = {"pr": "PER", "asr": "WER", "sv": "EER", "ser": "ERR"}[a.task]
    print(f"BEST {name} = {best:.4f} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
