"""Task calibration sets of 1,024 seconds -- **the selection protocol is exactly
the same as the 96-second sets and only the count changes**.

The existing 96-second sets (`calib/task_*.pt`) are not touched. These go out
under a separate name (`task1024_*.pt`) so that the two axes can be compared side
by side. The calibration size is the controlled variable, so nothing other than
the count may change:

  - same pool (the same indexer as the 96-second builder's pool / v2)
  - same shuffle (random.Random(0)) -> the 96-second set is the **leading subset**
    of this list
  - same preprocessing (load_wav normalizes the whole utterance, no random crop,
    truncate from the front and zero-pad if short)
  - KS alone uses 1-second clips (fixed in v2 -- padding to 2 s left 51% silence)
  - ER is five per-fold sets (fixed in v2 -- the test session of folds 1-4 leaked
    into calibration)

Because the shuffle seed is the same, the 96-second set is a proper subset of the
1,024-second set. That is intended -- we are looking at "more of the same data",
not "a different set of data".

WARNING: requantizing with this changes the wording of the paper. The current
Setup says "96 seconds of task-specific calibration audio", and the TAQ
comparison also presumes the same calibration. `results/calib_size_curve.csv`
says that from n=96->384 the **4-bit gain is effectively 0** (w2v2 PER 0.0687 ->
0.0713) and only 2-bit improves substantially. Check with one pilot cell before
scaling up.

  python -m talq.data_prep.make_task_calib
  python -m talq.data_prep.make_task_calib --sec 512 --tag task512
"""
import argparse, os, random

import torch

from talq.eval.probe_train import EMO_TRAIN, IEMOCAP_MANIFEST, iemocap_index, load_wav
from talq.paths import CALIB_ROOT, DATA_ROOT

CLIP = {"pr": 2.0, "asr": 2.0, "er": 2.0, "ic": 2.0, "asv": 2.0, "ks": 1.0}
# Used as --clip max_s, this follows the length TALQ actually sees
# (talq.eval.salience.MAX_S). It is needed to match TAQ-KL's importance to the
# same data as TALQ -- with 2-second crops one cannot say the same audio was seen
# as TALQ, which sees an 8-second context.
CLIP_MAXS = {"pr": 8.0, "asr": 8.0, "asv": 8.0, "er": 6.0, "ic": 6.0, "ks": 1.0}
TASKS = ["pr", "asr", "er", "ks", "ic", "asv"]


def pool(task, data=DATA_ROOT):
    """The same indexer as the 96-second builder's pool. sid is dropped (it is not
    one of the paper's 6 tasks)."""
    from talq.eval.superb_data import (ic_index, ks_index, load_lexicon_official,
                                       voxceleb_sid_index)
    from talq.eval.probe_train import librispeech_index
    LS = f"{data}/librispeech/LibriSpeech"
    if task == "pr":
        lex, ph = load_lexicon_official()
        return [r[0] for r in librispeech_index(f"{LS}/train-clean-100", "pr",
                                                lex, phones=ph)]
    if task == "asr":
        return [r[0] for r in librispeech_index(f"{LS}/train-clean-100", "asr")]
    if task == "er":
        return [f for f, _ in iemocap_index(IEMOCAP_MANIFEST, EMO_TRAIN)]
    if task == "ks":
        return [f for f, _ in ks_index("train")[0]]
    if task == "ic":
        return [f for f, _ in ic_index("train")[0]]
    return [f for f, _ in voxceleb_sid_index(f"{data}/voxceleb1/dev")[0]]


def build(files, clip_s, n):
    L = int(clip_s * 16000)
    assert len(files) >= n, f"pool too small: {len(files)} < {n}"
    ws = []
    for f in files[:n]:
        w = torch.tensor(load_wav(f))          # whole-utterance normalization, no random crop
        ws.append(torch.nn.functional.pad(w, (0, max(0, L - len(w))))[:L])
    return torch.stack(ws)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sec", type=float, default=1024, help="target total length (seconds)")
    ap.add_argument("--tag", default="task1024")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--clip", choices=["fixed", "max_s"], default="fixed",
                    help="fixed=2 s (KS 1 s, the existing protocol). "
                         "max_s=the length TALQ uses for training (pr/asr/asv 8 s, er/ic 6 s, "
                         "ks 1 s). For matching TAQ importance to the same audio as TALQ.")
    a = ap.parse_args()
    global CLIP
    if a.clip == "max_s":
        CLIP = dict(CLIP_MAXS)
        print(f"crop length = TALQ MAX_S protocol {CLIP}")
    os.makedirs(CALIB_ROOT, exist_ok=True)

    for t in [x for x in a.tasks.split(",") if x]:
        cs = CLIP[t]
        n = int(round(a.sec / cs))
        if t == "er":
            # For each fold, draw with that fold's own test session excluded (same as v2).
            for fold in (1, 2, 3, 4, 5):
                tr = {f"Session{i}" for i in range(1, 6)} - {f"Session{fold}"}
                p = [f for f, _ in iemocap_index(IEMOCAP_MANIFEST, tr)]
                random.Random(0).shuffle(p)
                x = build(p, cs, n)
                out = f"{CALIB_ROOT}/{a.tag}_er_fold{fold}.pt"
                torch.save(x, out)
                print(f"  er f{fold} {tuple(x.shape)}  "
                      f"{x.shape[0]*x.shape[1]/16000:.0f}s  "
                      f"zero-pad {float((x==0).float().mean()):.1%}  "
                      f"std {x.std():.3f} -> {out}", flush=True)
            continue
        p = pool(t)
        random.Random(0).shuffle(p)
        x = build(p, cs, n)
        out = f"{CALIB_ROOT}/{a.tag}_{t}.pt"
        torch.save(x, out)
        print(f"  {t:4} {tuple(x.shape)}  {x.shape[0]*x.shape[1]/16000:.0f}s  "
              f"zero-pad {float((x==0).float().mean()):.1%}  std {x.std():.3f} "
              f"(pool {len(p):,}) -> {out}", flush=True)


if __name__ == "__main__":
    main()
