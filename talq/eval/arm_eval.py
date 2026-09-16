"""Score a (possibly quantized) backbone on every SUPERB task we have a probe for.

Why a shared module: the AWQ baseline arm (arms 5-7) and talq.quant.task_fisher (arms 1-4)
each had their own three-metric eval. With seven tasks that duplication is where a
silent divergence would creep in -- one script scoring PR on dev-clean and the other
on test-clean would make the two arm families incomparable, and nothing would flag it.

Probes are the frozen fp32-trained ones in probes/. A task whose probe is missing
is skipped, so the sweep can start on the tasks that are ready (ER/KS/IC/SID/ASV) and
pick up PR/ASR when those probes land.

Reported set is the TEST split everywhere (task heads were selected on dev):
  pr/asr  test-clean      er   IEMOCAP Session5     ks/ic   official test csv/dir
  asv     veri_test2.txt  sid  dev-split holdout (SUPERB's SID test split is not
                               reproducible from what we have; the same fixed subset
                               is used for every variant so degradation stays comparable)
"""
import os

import torch

from talq.paths import DATA_ROOT, PROBES
from talq.eval.superb_data import ic_index, ks_index, voxceleb_sid_index
from talq.eval.superb_heads import ASVProbe, UttProbe
from talq.eval.probe_train import (
    CHARS, EMO_TEST, IEMOCAP_MANIFEST, LinearCTC, BLSTMCTC, SR, eval_ctc, eval_sv,
    hidden_states, iemocap_index, librispeech_index, load_lexicon, load_wav,
)

TASKS = ["pr", "asr", "er", "ks", "ic", "sid", "asv"]
METRIC = {"pr": "PER", "asr": "WER", "er": "ER_ERR", "ks": "KS_ERR",
          "ic": "IC_ERR", "sid": "SID_ERR", "asv": "EER"}
FIELDS = ["backbone", "variant"] + [METRIC[t] for t in TASKS]
CAP = {"er": 1241, "ks": 1000, "ic": 1000, "sid": 1000}   # fixed prefix, not random
MAX_S = {"er": 6.0, "ks": 1.0, "ic": 6.0, "sid": 8.0}


def build_probe(task, ck, dev):
    """Rebuild the head from its checkpoint. Output width comes from the saved
    tensor, not a constant -- a mismatch here would silently score garbage."""
    nl, dim, st = ck["n_layers"], ck["dim"], ck["state"]
    if task == "pr":
        p = LinearCTC(nl, dim, st["head.weight"].shape[0])
    elif task == "asr":
        p = BLSTMCTC(nl, dim, st["head.weight"].shape[0])
    elif task == "asv":
        p = ASVProbe(nl, dim, st["loss.W"].shape[1])
    else:
        p = UttProbe(nl, dim, st["head.weight"].shape[0])
    p.load_state_dict(st)
    return p.eval().to(dev).requires_grad_(False)


def load_probes(hf_id, probe_dir=PROBES, dev="cuda", tasks=TASKS):
    tag = hf_id.split("/")[-1]
    out = {}
    for t in tasks:
        fp = f"{probe_dir}/{tag}_{t}.pt"
        if os.path.exists(fp):
            out[t] = build_probe(t, torch.load(fp, map_location=dev,
                                               weights_only=False), dev)
    return out


def load_eval_data(data=DATA_ROOT, sv_stride=4):
    """Everything the seven evals need, built once per process."""
    lex, phones = load_lexicon()
    try:                                    # the PR task head was trained on the official 71 phonemes
        from talq.eval.superb_data import load_lexicon_official
        lex_o, phones_o = load_lexicon_official()
    except Exception:
        lex_o, phones_o = lex, phones
    LS = f"{data}/librispeech/LibriSpeech"
    d = {
        "pr": (librispeech_index(f"{LS}/test-clean", "pr", lex_o, phones=phones_o),
               phones_o, " "),
        "asr": (librispeech_index(f"{LS}/test-clean", "asr"), CHARS, ""),
        "er": iemocap_index(IEMOCAP_MANIFEST, EMO_TEST)[:CAP["er"]],
        "ks": ks_index("test")[0][:CAP["ks"]],
        "ic": ic_index("test")[0][:CAP["ic"]],
    }
    pool, _ = voxceleb_sid_index(f"{data}/voxceleb1/dev")
    import random
    random.Random(0).shuffle(pool)
    cut = int(len(pool) * 0.8)
    d["sid"] = pool[cut:][:CAP["sid"]]       # same seed 0 dev split as in training
    d["asv"] = ([(int(l), p1, p2) for l, p1, p2 in
                 (ln.split() for ln in open(f"{data}/voxceleb1/veri_test2.txt"))][::sv_stride],
                f"{data}/voxceleb1/test")
    return d


@torch.no_grad()
def _clf_err(m, probe, items, dev, max_s, slots=None):
    wrong = 0
    for f, y in items:
        hs = hidden_states(m, torch.tensor(load_wav(f, max_s=max_s))[None].to(dev)).float()
        lg = probe(hs)
        if slots:                            # ic: correct only if all 3 slots are right
            o, ok = 0, True
            for i, sz in enumerate(slots):
                ok &= int(lg[0, o:o + sz].argmax()) == y[i]; o += sz
            wrong += not ok
        else:
            wrong += int(lg.argmax(-1)[0]) != y
    return wrong / len(items)


@torch.no_grad()
def eval_all(m, probes, d, dev, n_ctc=64, ic_slots=(6, 14, 4)):
    """{task: metric}. A task whose head is missing is skipped."""
    r = {}
    for t in ("pr", "asr"):
        if t in probes:
            items, itos, sep = d[t]
            r[t] = eval_ctc(m, probes[t], items, dev, itos, sep, n=n_ctc)
    for t in ("er", "ks", "sid"):
        if t in probes:
            r[t] = _clf_err(m, probes[t], d[t], dev, MAX_S[t])
    if "ic" in probes:
        r["ic"] = _clf_err(m, probes["ic"], d["ic"], dev, MAX_S["ic"], slots=ic_slots)
    if "asv" in probes:
        trials, root = d["asv"]
        r["asv"] = eval_sv(m, probes["asv"], root, trials, dev)
    return r


def row(bk, variant, res):
    out = {"backbone": bk, "variant": variant}
    for t in TASKS:
        out[METRIC[t]] = round(res[t], 4) if t in res else ""
    return out
