"""AWQ task salience from the SUPERB probes (arm 6 input), for all seven tasks.

Replaces the earlier AWQ salience script for the same reason the full-split
sensitivity map replaced the per-layer bit sweep: the old script drives the loss
through the PER-LAYER task heads (probes/perlayer/), which exist only for
pr/sv/ser, and which read different layers than the probe that actually scores
the arms.

What AWQ does with this: awq_search_layer takes an optional `sal` vector that
REPLACES the stock salience (mean|x| over calib activations) when choosing the
per-channel scale. So this file decides, per Linear, which input channels the task
cares about.

    sal_j = E_batch | x_ij * dL_task/dx_ij |     (summed over frames, averaged over batches)

The per-module full_backward_hook is not an optimisation -- it is required. q/k/v
all receive the SAME hidden_states tensor, so a retain_grad() on that tensor would
give each of the three the SUM of all three gradients.

Output matches the --salience-dir contract the AWQ arm expects:
    salience/{bk}_{task}.pt  ->  {"salience": {"L{k}.{name}": tensor(in_features)}}

  python -m talq.eval.salience --backbone w2v2 --task all
"""
import argparse
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from talq.paths import CALIB_ROOT, DATA_ROOT, EMILIA_EN, PROBES, RESULTS_ROOT

from talq.eval import arm_eval
from talq.eval.calib_io import load_calib
from talq.quant.precompute import BACKBONES, patch_wavlm_eager
from talq.eval.probe_train import (
    EMO_TRAIN, IEMOCAP_MANIFEST, iemocap_index, librispeech_index, load_wav, pad_batch,
)

IC_SLOTS = (6, 14, 4)
# Must be identical to task-Fisher (CLF_MAX_S). With a different length the salience
# is computed on a distribution the probe has never seen.
MAX_S = {"pr": 8.0, "asr": 8.0, "sid": 8.0, "asv": 8.0, "er": 6.0, "ic": 6.0, "ks": 1.0}


class SalienceCollector:
    """sum|x * dL/dx| per Linear. Same formula as the earlier salience collector."""

    def __init__(self, layers):
        self.acc, self.n, self._x, self.handles = {}, 0, {}, []
        for k, layer in enumerate(layers):
            for name, mod in layer.named_modules():
                if isinstance(mod, nn.Linear):
                    key = f"L{k}.{name}"
                    self.handles.append(mod.register_forward_pre_hook(
                        lambda _m, a, kk=key: self._x.__setitem__(kk, a[0].detach())))
                    self.handles.append(mod.register_full_backward_hook(
                        lambda _m, gi, _go, kk=key: self._back(kk, gi)))

    def _back(self, key, grad_in):
        g, x = grad_in[0], self._x.pop(key, None)
        if g is None or x is None:
            return
        d = g.shape[-1]
        v = (x.reshape(-1, d) * g.reshape(-1, d)).abs().sum(0).detach()
        self.acc[key] = v if key not in self.acc else self.acc[key] + v

    def finish(self):
        for h in self.handles:
            h.remove()
        return {k: (v / max(self.n, 1)).cpu() for k, v in self.acc.items()}


def hidden_states_grad(model, x):
    """talq.eval.probe_train.hidden_states has @torch.no_grad() on it. Salience has to
    backpropagate through this forward, so with that helper there is no grad_fn and
    backward dies immediately. talq.quant.task_fisher calls it directly for the same
    reason."""
    return torch.stack(model(x, output_hidden_states=True).hidden_states)


def train_pool(task, data, er_fold=None):
    """Drawn from the training split -- building salience on the evaluation set is a leak.

    er_fold: only meaningful for ER. Given n, draws from the **4 sessions with
      Session n taken out**. ER is 5-fold in SUPERB, so scoring fold n requires that
      session to be absent from every training resource. calib was already separated
      per fold in make_task_calib_v2, but the alpha training pool was fixed to
      EMO_TRAIN (Session1~4). That puts the fold's own test session into the pool
      when scoring fold1~4 -- KL does not use the labels, but it does see the audio.
      None is the old behaviour (EMO_TRAIN fixed), which is safe only for fold5.
    """
    from talq.eval.superb_data import ic_index, ks_index, load_lexicon_official, voxceleb_sid_index
    LS = f"{data}/librispeech/LibriSpeech"
    if task == "pr":
        lex, ph = load_lexicon_official()
        return librispeech_index(f"{LS}/train-clean-100", "pr", lex, phones=ph)
    if task == "asr":
        return librispeech_index(f"{LS}/train-clean-100", "asr")
    if task == "er":
        tr = ({f"Session{i}" for i in range(1, 6)} - {f"Session{er_fold}"}
              if er_fold else EMO_TRAIN)
        return iemocap_index(IEMOCAP_MANIFEST, tr)
    if task == "ks":
        return ks_index("train")[0]
    if task == "ic":
        return ic_index("train")[0]
    return voxceleb_sid_index(f"{data}/voxceleb1/dev")[0]          # sid / asv


def task_loss(task, probe, hs, model, lens, chunk, dev):
    if task in ("pr", "asr"):
        tgt = [torch.tensor(t) for _, _, t in chunk]
        in_lens = model._get_feat_extract_output_lengths(lens).to(torch.long)
        return F.ctc_loss(probe(hs).log_softmax(-1).transpose(0, 1),
                          torch.cat(tgt).to(dev), in_lens,
                          torch.tensor([len(t) for t in tgt]), blank=0, zero_infinity=True)
    y = torch.tensor([c[1] for c in chunk]).to(dev)
    if task == "asv":
        flens = model._get_feat_extract_output_lengths(lens).to(torch.long)
        return probe(hs, y, flens=flens.tolist())
    if task == "ic":
        lg, loss, off = probe(hs), 0.0, 0
        for i, sz in enumerate(IC_SLOTS):
            loss = loss + F.cross_entropy(lg[:, off:off + sz], y[:, i]); off += sz
        return loss
    return F.cross_entropy(probe(hs), y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=list(BACKBONES))
    ap.add_argument("--task", default="all")
    ap.add_argument("--out", default=f"{RESULTS_ROOT}/salience")
    ap.add_argument("--probes", default=f"{PROBES}")
    ap.add_argument("--data", default=f"{DATA_ROOT}")
    ap.add_argument("--calib", default=f"{CALIB_ROOT}/emilia_calib.pt")
    ap.add_argument("--emilia", default=f"{EMILIA_EN}")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    bk, dev = a.backbone, "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)

    from transformers import AutoModel
    m = AutoModel.from_pretrained(BACKBONES[bk], dtype=torch.float32).eval().to(dev)
    m.requires_grad_(False)
    if bk.startswith("wavlm"):     # wavlmL included. q/k/v must be visible as modules for the hooks to attach
        patch_wavlm_eager(m, load_calib(a.calib, a.emilia, dev)[:2])
    n_lin = sum(1 for l in m.encoder.layers for _, mm in l.named_modules()
                if isinstance(mm, nn.Linear))

    probes = arm_eval.load_probes(BACKBONES[bk], a.probes, dev)
    tasks = sorted(probes) if a.task == "all" else [t for t in a.task.split(",")]
    for task in tasks:
        if task not in probes:
            print(f"[skip] {bk}/{task}: no probe", flush=True); continue
        probe = probes[task]
        probe.train() if task == "asr" else probe.eval()   # required by cudnn RNN backward
        pool = train_pool(task, a.data)
        random.Random(a.seed).shuffle(pool)
        col = SalienceCollector(m.encoder.layers)
        for i in range(0, min(a.steps * a.bs, len(pool)), a.bs):
            chunk = pool[i:i + a.bs]
            waves = [load_wav(c[0], max_s=MAX_S[task]) for c in chunk]
            x, lens = pad_batch(waves, dev)
            x = x.clone().requires_grad_(True)
            m.zero_grad(set_to_none=True)
            task_loss(task, probe, hidden_states_grad(m, x), m, lens, chunk, dev).backward()
            col.n += 1
        sal = col.finish()
        assert len(sal) == n_lin, \
            f"{bk}/{task}: salience {len(sal)}/{n_lin} Linears -- not every hook attached"
        # if q/k/v, which share an input, hold the same value, the hooks accumulated across each other
        keys = [k for k in sal if k.endswith(("q_proj", "k_proj"))][:2]
        if len(keys) == 2:
            assert not torch.allclose(sal[keys[0]], sal[keys[1]], atol=1e-6), \
                "q_proj/k_proj salience identical -- hooks accumulated across each other"
        fp = f"{a.out}/{bk}_{task}.pt"
        torch.save({"backbone": bk, "task": task, "steps": a.steps, "bs": a.bs,
                    "probe": "superb", "salience": sal}, fp)
        print(f"  {bk}/{task}: {len(sal)} Linears -> {fp}", flush=True)
    print(f"SALIENCE_DONE {bk}", flush=True)


if __name__ == "__main__":
    main()
