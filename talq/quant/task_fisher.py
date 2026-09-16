"""Path A: task-Fisher-weighted GPTQ Hessian.

GPTQ's native H = E[x xT] (reconstruction only -- how much a column of the
INPUT varies, nothing about whether that variance matters to the task). Here
we swap in H_task = E[w * x xT] where w = ||d task_loss / d layer_output||^2
per calib frame, captured via retain_grad() on each Linear's output during a
real backward pass through the frozen backbone + frozen probe head. This is
literally OBS's original task-Hessian, GPTQ just approximated it away with
the activation-only proxy -- so this stays inside GPTQ's own closed-form
column-quantization math (Cholesky/Hinv unchanged), only what goes into H
changes. No search, no allocation layer, no AWQ-style salience scale: this is
the third architectural position (TALQ = allocation level, AWQ =
salience/scale level, this = the quantizer's own Hessian).

base backbones only, uniform INT4 only, WER/EER/PER vs uniform-b4 baseline
(reconstruction-Hessian GPTQ, same bits/group).
"""
import argparse
import csv
import json
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from talq.paths import CALIB_ROOT, DATA_ROOT, EMILIA_EN, PROBES, RESULTS_ROOT
from talq.quant.precompute import BACKBONES, patch_wavlm_eager
from talq.quant.gptq import GPTQ, quantize_layer as quantize_layer_recon
from talq.eval.calib_io import load_calib
from talq.eval.sweep_probes import build_probe, TASKS
from talq.eval.probe_train import (
    load_lexicon, librispeech_index, voxceleb_index, load_wav, pad_batch,
    eval_ctc, eval_sv, CHARS,
)
from talq.eval.ctc_align import align_batch

BASE_BACKBONES = ["w2v2", "wavlm", "w2v2960h", "hubert"]
# --spec-json mode fills the mixed-precision half of the 2x2. For a spec derived from
# task tau (the ranking-based allocator), BOTH arms use that same spec, so the only thing differing
# between recon_a{tau} and fisher_{tau}_a{tau} is the Hessian -- the H axis stays clean.
# evalall() scores every variant on WER/EER/PER, so the cross-task cell (allocation
# from pr, read on EER) falls out of the same run with no extra work.
from talq.eval.arm_eval import FIELDS   # schema shared by the 7 tasks
IC_SLOTS = (6, 14, 4)                 # fluent_speech_commands: action/object/location
FISHER_TASKS = ["pr", "asr", "er", "ks", "ic", "sid", "asv"]  # content=PER (linear probe, no LSTM confound), speaker=AAM


def sv_aam_loss(probe, hs, y, m=0.2, s=30.0):
    """AAM-softmax on the frozen SVProbe's own embed()/head weights -- no probe
    retrain. CE optimizes a linear decision boundary; EER is scored by cosine
    similarity between unseen-speaker pairs, a different geometry. AAM adds an
    angular margin in that same cosine space the frozen head.weight rows
    already define, so the calib-time loss finally matches what eval scores."""
    e = F.normalize(probe.embed(hs), dim=-1)
    W = F.normalize(probe.head.weight, dim=-1)
    cos = (e @ W.t()).clamp(-1 + 1e-7, 1 - 1e-7)
    theta_y = torch.arccos(cos.gather(1, y[:, None]).squeeze(1))
    cos_margin = torch.cos(theta_y + m)
    logits = cos.scatter(1, y[:, None], cos_margin[:, None]) * s
    return F.cross_entropy(logits, y)


def add_batch_weighted(g: GPTQ, x, w):
    x = x.reshape(-1, g.cols).float()
    w = w.reshape(-1).float().clamp(min=0)
    g.add_batch(x * w.sqrt().unsqueeze(-1))


def reweight(w, gamma=1.0, alpha=1.0):
    """frame weight w -> (1-alpha) + alpha * (w/mean(w))^gamma.

    Two knobs merged into one function. Both are monotone transforms of w, so
    they leave the frame 'ranking' untouched and only change the dynamic range
    -- unlike blank/align (rejected), which were different quantities that
    changed the ranking of w itself.

      gamma < 1  (power scaling): compresses the spread. w=0 is still 0.
      alpha < 1  (hybrid Hessian): lays a (1-alpha) floor under every frame.

    Why hybrid does not need to accumulate a second H:
        (1-a)*sum_t x x^T + a*sum_t w_t x x^T = sum_t [(1-a) + a*w_t] x x^T
    Interpolating H_orig and H_task drawn from the same batch is exactly the
    same as an affine transform of the per-frame weight. So alpha=0 becomes
    "same batch, w=1", i.e. the control that separates the calib-data effect
    from the gradient effect.

    Dividing by the mean is mandatory. w = ||dL/dy||^2 has an arbitrary scale
    (depends on the loss scale and the batch size), so without normalization
    adding it to (1-alpha) makes the w term either dominate or vanish whatever
    alpha is, and alpha means nothing. At alpha=1 it looks like this
    normalization would not change the result, since GPTQ is invariant to the
    overall scale of H (percdamp is relative to mean(diag H)); but here we
    divide per batch, so the relative contribution across batches changes --
    which is why even the alpha=1 endpoint has to be re-measured inside the
    sweep so it does not mix with the existing fisher_* rows.
    """
    w = w.float().clamp(min=0)
    mu = w.mean()
    if mu <= 0:
        return torch.ones_like(w)
    w = w / mu
    if gamma != 1.0:
        w = w.pow(gamma)
        mu = w.mean()
        w = w / mu if mu > 0 else torch.ones_like(w)
    if alpha != 1.0:
        w = (1.0 - alpha) + alpha * w
    return w


def smooth(w, win):
    """w: (B,T) -> local moving-average of width `win` along T (replicate-pad).
    A smoothing-window diagnostic found the raw per-frame CTC gradient
    spike is mostly noise at layer 0 but partly jittery-but-real at mid/late
    layers -- smoothing raises asr/pr frame-weight correlation there
    (0.28->0.52 at win25) though the ΔW-cosine payoff is smaller."""
    if win <= 1 or w.dim() != 2:
        # non-(B,T) grad shape (e.g. WavLM's rel-pos-bias linears) -- skip
        # smoothing rather than guess which axis is time.
        return w
    pad = win // 2
    wp = F.pad(w.unsqueeze(1), (pad, pad), mode="replicate")
    return F.avg_pool1d(wp, kernel_size=win, stride=1).squeeze(1)[:, :w.shape[1]]


def adaptive_win(layer_idx, nl, base_win):
    """A smoothing-window diagnostic found smoothing raised asr/pr
    frame-weight correlation at mid/late layers (0.28->0.52 at win25) but did
    NOTHING at layer 0 (stayed ~0.0 regardless of window) -- the early-layer
    spike is pure noise, not a jittery-but-real signal, so smoothing it just
    blurs noise. Skip smoothing on the first third of layers, apply base_win
    to the rest."""
    return 1 if layer_idx < nl / 3 else base_win


def ess_alpha(w, target):
    """Axis 4 -- derive the shrinkage coefficient instead of tuning it.

    Normalizing w to mean(w)=1 makes w~ = (1-a) + a*w have mean 1 as well, so
        ESS(w~) = (sum w~)^2 / (n * sum w~^2) = 1 / mean(w~^2)
        mean(w~^2) = 1 + a^2 * Var(w)
    hence ESS = 1/(1 + a^2 Var(w)), and for a target ESS=T

        a = sqrt( (1/T - 1) / Var(w) )

    comes out in closed form. No numerical solve needed. If Var(w) is small
    (already even weights) a is clipped at >1 so shrinkage turns off, and the
    worse the spike the smaller a gets. Same shape as Ledoit-Wolf shrinkage --
    the more the estimator scatters, the harder it is pulled toward the
    task-free estimator."""
    w = w.float().clamp(min=0)
    mu = w.mean()
    if mu <= 0:
        return 0.0
    var = (w / mu).var(unbiased=False).item()
    if var <= 1e-12:
        return 1.0
    return float(min(1.0, max(0.0, ((1.0 / target - 1.0) / var) ** 0.5)))


def mc_fisher_loss(task, probe, hs, dev, in_lens=None, flens=None):
    """Axis 2 -- true Fisher instead of empirical Fisher.

    The w = ||dL(y, y*)/dy||^2 we use now is the *empirical* Fisher, which uses
    the true label y*. The better the model fits, the more g->0, so the mass
    piles up only on "wrong frames" -- the ESS of 1-4% and the spike we measured
    (the top 10% of frames hold 83-98% of the mass) are exactly this property.
    That is, the current weight may be the probe's error distribution rather
    than task importance.

    The true Fisher draws y^ ~ p(.|x) from the model's own predictive
    distribution instead of the label:
        F = E_{y^~p}[ grad log p(y^) grad log p(y^)^T ]
    This is the definition that agrees with Gauss-Newton, and it grows where the
    model is *uncertain*, not where it is wrong. And it needs no labels -- it can
    be measured on any audio (Emilia included).

    For CTC (pr/asr) sampling a label sequence is awkward, so we sample from the
    per-frame categorical and use a frame CE. This corresponds to the GGN of the
    per-frame softmax, and every speech frame contributes."""
    if task in ("asr", "pr"):
        logits = probe(hs)                                   # (B,T,C)
        B, T, C = logits.shape
        flat = logits.reshape(-1, C)
        with torch.no_grad():
            yhat = torch.multinomial(flat.softmax(-1), 1).squeeze(1)
        ce = F.cross_entropy(flat, yhat, reduction="none").reshape(B, T)
        if in_lens is not None:                              # drop padded frames
            m = (torch.arange(T, device=ce.device)[None, :] < in_lens[:, None].to(ce.device))
            return (ce * m).sum() / m.sum().clamp(min=1)
        return ce.mean()
    if task == "asv":
        # AMSoftmax's margin is a training trick, not part of p(y|x). The Fisher
        # must be measured on the distribution the model defines, so use the
        # margin-free logits.
        emb = probe.utt(probe._trunk(hs, flens))
        wf = F.normalize(emb, dim=1) @ F.normalize(probe.loss.W, dim=0)
        logits = probe.loss.s * wf
        with torch.no_grad():
            yhat = torch.multinomial(logits.softmax(-1), 1).squeeze(1)
        return F.cross_entropy(logits, yhat)
    if task == "ic":
        lg, loss, off = probe(hs), 0.0, 0
        for sz in IC_SLOTS:
            sl = lg[:, off:off + sz]
            with torch.no_grad():
                yhat = torch.multinomial(sl.softmax(-1), 1).squeeze(1)
            loss = loss + F.cross_entropy(sl, yhat); off += sz
        return loss
    if task in ("er", "ks", "sid"):
        lg = probe(hs)
        with torch.no_grad():
            yhat = torch.multinomial(lg.softmax(-1), 1).squeeze(1)
        return F.cross_entropy(lg, yhat)
    raise ValueError(f"mc_fisher_loss: unsupported task {task}")


def quantize_layer_taskfisher(model, layer, batches, probe, task, bits, group, dev,
                              smooth_win=1, percdamp=0.01, layer_idx=0, nl=1,
                              grad_weight=True, weight_mode="grad", rw=None,
                              fisher_type="emp", gproj=0, shrink_ess=0.0):
    """grad_weight=False -> w=1 for every frame, i.e. plain reconstruction GPTQ
    but on THESE batches. That's the control that separates "task-gradient
    weighting helps" from "in-domain calib data helps": same clips, same code
    path, only the weighting differs.

    weight_mode:
      "grad"  -- w = ||dL/dy_t||^2, the Fisher/Gauss-Newton term. Principled, but
                 for CTC it is spike-dominated (measured ESS 3-6% of frames, top
                 10% of frames hold 83-98% of the mass), which is why the content
                 axis loses to uniform at equal calib.
      "blank" -- w = 1 - p(blank)_t from the frozen CTC probe. NOT a curvature
                 term; a smooth proxy for "this frame carries phonetic content".
                 The point is to keep the task-relevance prior while removing the
                 spike, to test whether the spike alone is what sinks "grad".
                 Content tasks only (sv has no blank symbol). No backward needed.
      "align" -- w = ||dL_ce/dy_t||^2 where L_ce is frame-wise CE against a CTC
                 forced alignment. Still a real curvature/Fisher term (unlike
                 "blank"), but every speech frame carries a target instead of the
                 loss peaking on a few: measured alignment coverage is 48.8% of
                 frames vs 3-6% effective under CTC. Content tasks only.
    """
    linears = {n: m for n, m in layer.named_modules() if isinstance(m, nn.Linear)}
    gptq = {n: GPTQ(m) for n, m in linears.items()}
    cap_x, cap_y = {}, {}

    def hook(mod, args, out, n):
        cap_x[n] = args[0].detach()
        if out.requires_grad:  # false in the grad_weight=False (uniform) pass
            out.retain_grad()
        cap_y[n] = out

    handles = [m.register_forward_hook(lambda mod, a, o, n=n: hook(mod, a, o, n))
               for n, m in linears.items()]

    if grad_weight and weight_mode == "grad":
        probe.train()  # cudnn RNN backward (BLSTMCTC) requires training mode; probe
                       # params stay frozen (requires_grad_(False) at load), so this
                       # only flips cudnn's reserve-buffer / dropout flags, not grads
    else:
        probe.eval()   # blank mode reads posteriors, so keep dropout off
    # Axis 1 (gproj) needs the top subspace of G = sum_t g g^T, so it runs twice.
    #   pass 0: accumulate G only  /  pass 1: accumulate H with the projected weight
    # With gproj=0 there is a single pass, identical to the original behavior.
    Gacc, Uproj = {}, {}
    for _pass in ([0, 1] if gproj > 0 else [1]):
        for x, lens, tgt in batches:
            x = x.clone().to(dev).requires_grad_(True)
            frame_w = None
            if grad_weight and weight_mode == "blank":
                if task not in ("asr", "pr"):
                    raise ValueError("weight_mode='blank' needs a CTC probe (asr/pr)")
                with torch.no_grad():
                    hs = torch.stack(model(x, output_hidden_states=True).hidden_states)
                    p = probe(hs).softmax(-1)
                    frame_w = (1.0 - p[..., 0]).detach()   # blank=0, matches ctc_loss
            elif grad_weight and weight_mode == "align":
                if task not in ("asr", "pr"):
                    raise ValueError("weight_mode='align' needs a CTC probe (asr/pr)")
                in_lens = model._get_feat_extract_output_lengths(lens).to(torch.long)
                hs = torch.stack(model(x, output_hidden_states=True).hidden_states)
                logits = probe(hs)
                # align on detached logits (alignment is a target, not a gradient
                # path), then take the frame-wise CE gradient through the same logits
                labels, amask = align_batch(logits.detach().log_softmax(-1), tgt, in_lens)
                model.zero_grad(set_to_none=True)
                if amask.any():
                    F.cross_entropy(logits[amask], labels[amask]).backward()
            elif grad_weight and fisher_type == "mc":
                hs = torch.stack(model(x, output_hidden_states=True).hidden_states)
                il = (model._get_feat_extract_output_lengths(lens).to(torch.long)
                      if task in ("asr", "pr") else None)
                fl = (model._get_feat_extract_output_lengths(lens).to(torch.long).tolist()
                      if task == "asv" else None)
                loss = mc_fisher_loss(task, probe, hs, dev, in_lens=il, flens=fl)
                model.zero_grad(set_to_none=True)
                loss.backward()
            elif grad_weight:
                hs = torch.stack(model(x, output_hidden_states=True).hidden_states)
                if task in ("asr", "pr"):
                    logits = probe(hs)
                    in_lens = model._get_feat_extract_output_lengths(lens).to(torch.long)
                    loss = F.ctc_loss(
                        logits.log_softmax(-1).transpose(0, 1), torch.cat(tgt).to(dev),
                        in_lens, torch.tensor([len(t) for t in tgt]), blank=0, zero_infinity=True)
                elif task == "asv":
                    # ASVProbe returns the AMSoftmax loss directly. flens is needed so
                    # that StatsPooling computes its statistics excluding the padding
                    # (without it the padding ratio gets mixed in).
                    flens = model._get_feat_extract_output_lengths(lens).to(torch.long)
                    loss = probe(hs, torch.tensor(tgt).to(dev), flens=flens.tolist())
                elif task == "ic":
                    lg = probe(hs)                       # 24 = 6 + 14 + 4
                    y = torch.tensor(tgt).to(dev)
                    loss, off = 0.0, 0
                    for i, sz in enumerate(IC_SLOTS):
                        loss = loss + F.cross_entropy(lg[:, off:off + sz], y[:, i]); off += sz
                elif task in ("er", "ks", "sid"):
                    loss = F.cross_entropy(probe(hs), torch.tensor(tgt).to(dev))
                else:  # legacy sv probe (SVProbe) path
                    loss = sv_aam_loss(probe, hs, torch.tensor(tgt).to(dev))
                model.zero_grad(set_to_none=True)
                loss.backward()
            else:
                with torch.no_grad():
                    model(x)  # populate the hooks only; no probe, no loss, no backward
            for n in list(linears):
                if n not in cap_y:
                    continue
                if frame_w is not None:
                    # one per-frame weight shared by every Linear in the layer; fall
                    # back to uniform where the captured shape is not (B,T,*) (e.g.
                    # WavLM's rel-pos-bias linears)
                    w = (frame_w if cap_y[n].shape[:-1] == frame_w.shape
                         else torch.ones(cap_y[n].shape[:-1], device=cap_y[n].device))
                elif grad_weight:
                    if cap_y[n].grad is None:
                        continue
                    g = cap_y[n].grad.detach()
                    if _pass == 0:                       # axis 1: collect G only
                        gf = g.reshape(-1, g.shape[-1]).float()
                        Gacc[n] = (gf.T @ gf if n not in Gacc else Gacc[n] + gf.T @ gf)
                        continue
                    if gproj > 0 and n in Uproj:
                        # axis 1 -- ||g||^2 treats every output direction as equal.
                        # Projecting onto the top subspace of G keeps only the part
                        # that lies on the "output directions the task actually
                        # uses" and drops the noise directions that differ from
                        # frame to frame. It stays scalar-shaped, so GPTQ's row
                        # separability is unchanged (using G's off-diagonal would
                        # break it).
                        w = (g.reshape(-1, g.shape[-1]).float() @ Uproj[n]) \
                            .pow(2).sum(-1).reshape(g.shape[:-1])
                    else:
                        w = g.pow(2).sum(-1)
                    if weight_mode == "grad":
                        # smoothing exists to blunt the CTC spike; the aligned CE
                        # gradient is already spread, so blurring it would only
                        # discard the frame selectivity that is the point
                        w = smooth(w, adaptive_win(layer_idx, nl, smooth_win))
                else:
                    if _pass == 0:
                        continue
                    w = torch.ones(cap_y[n].shape[:-1], device=cap_y[n].device)
                if shrink_ess > 0:
                    # axis 4 -- derive alpha from a target ESS instead of tuning it
                    w = reweight(w, 1.0, ess_alpha(w, shrink_ess))
                if rw is not None:
                    # a no-op on the w=1 path (normalization/power/affine all preserve 1)
                    w = reweight(w, *rw)
                add_batch_weighted(gptq[n], cap_x[n], w)
            cap_x.clear(); cap_y.clear()
        if _pass == 0:
            # The top gproj eigenvectors. G is (out,out) and we hold only one layer
            # at a time.
            for n, G in Gacc.items():
                r = min(gproj, G.shape[0])
                try:
                    evals, evecs = torch.linalg.eigh(G.double())
                    Uproj[n] = evecs[:, -r:].to(G.dtype).contiguous()
                except Exception as e:
                    print(f"  [gproj] {n}: eigh failed({e}), falling back to isotropic", flush=True)
            continue

    for h in handles:
        h.remove()
    if grad_weight:
        probe.eval()
    for n in list(gptq):
        if gptq[n].H is None:
            print(f"  [skip] {n}: no grad captured, kept fp32", flush=True)
            del gptq[n]
            continue
        gptq[n].quantize(bits=bits, group=group, percdamp=percdamp)
    return gptq


def make_ctc_batches(items, dev, n=32, bs=8, seed=0, max_s=8.0):
    items = items[:]
    random.Random(seed).shuffle(items)
    items = items[:n]
    batches = []
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        waves = [load_wav(f, max_s=max_s) for f, _, _ in chunk]
        x, lens = pad_batch(waves, dev)
        tgt = [torch.tensor(t) for _, _, t in chunk]
        batches.append((x, lens, tgt))
    return batches


# Audio length cap for the utterance-level tasks. Matched to the MAX_S used at
# training time -- using a different length here would compute the Fisher on a
# distribution the task head has never seen.
CLF_MAX_S = {"sid": 8.0, "asv": 8.0, "er": 6.0, "ic": 6.0, "ks": 1.0}


def make_sv_batches(items, dev, n=32, bs=8, seed=0, max_s=3.0):
    items = items[:]
    random.Random(seed).shuffle(items)
    items = items[:n]
    batches = []
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        waves = [load_wav(f, max_s=max_s) for f, _ in chunk]
        x, lens = pad_batch(waves, dev)
        tgt = [s for _, s in chunk]
        batches.append((x, lens, tgt))
    return batches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=f"{RESULTS_ROOT}/task_fisher_b4.csv")
    ap.add_argument("--calib", default=f"{CALIB_ROOT}/emilia_calib.pt")
    ap.add_argument("--emilia", default=f"{EMILIA_EN}")
    ap.add_argument("--data", default=f"{DATA_ROOT}")
    ap.add_argument("--probes", default=f"{PROBES}")
    ap.add_argument("--bits-list", default="4", help="comma-sep uniform bit-widths to sweep")
    ap.add_argument("--backbones", default=",".join(BASE_BACKBONES),
                    help="comma-sep backbone keys; split by backbone to parallelise")
    ap.add_argument("--spec-json", default=None,
                    help="the ranking-based allocator's output. When given, --bits-list is ignored and "
                         "the run does mixed-precision instead of uniform.")
    ap.add_argument("--budgets", default="2.5,3.0,3.5",
                    help="--spec-json only: which avg-bit budgets to run")
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--n-calib", type=int, default=32)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--n-eval", type=int, default=64)
    ap.add_argument("--fisher-type", default="emp", choices=["emp", "mc"],
                    help="axis 2. emp=true labels (empirical Fisher, the existing one). "
                         "mc=sample from the model's predictive distribution (true Fisher, no labels needed).")
    ap.add_argument("--gproj", type=int, default=0,
                    help="axis 1. If >0, use the norm projected onto the top R eigen-"
                         "directions of G=sum g g^T. Becomes 2-pass, so twice the time.")
    ap.add_argument("--shrink-ess", type=float, default=0.0,
                    help="axis 4. If >0, the target ESS. alpha is not tuned but derived "
                         "as sqrt((1/T-1)/Var(w)).")
    ap.add_argument("--fisher-calib", default="task", choices=["task", "emilia"],
                    help="which audio to measure the Fisher on. emilia has no labels, so "
                         "it can only be used with --fisher-type mc.")
    ap.add_argument("--seed", type=int, default=0,
                    help="calib sampling + torch RNG. Change it only to estimate variance. "
                         "R{seed} is appended to the variant name so it does not mix with the seed 0 results.")
    ap.add_argument("--power-gamma", type=float, default=None,
                    help="w -> (w/mean w)^gamma. 0<gamma<1 compresses the spread. "
                         "Giving it takes the reweight path (gamma=1.0 too).")
    ap.add_argument("--hessian-alpha", type=float, default=None,
                    help="H = (1-a)H_orig + a*H_task. On the same batch this is the same as an "
                         "affine transform of the frame weight. a=0 is the recon (same batch, w=1) control.")
    ap.add_argument("--spec-tasks", default=None,
                    help="--spec-json only: whose task spec to run. Defaults to the same as "
                         "--fisher-tasks. arm2 (recon only) has to run the spec without fisher, so it is split off.")
    ap.add_argument("--fisher-tasks", default=None,
                    help='comma-separated. An empty string ("") skips fisher and runs only '
                         "recon (arm1). arm3 eats 88%% of the variants, so this is needed when splitting the run per arm.")
    ap.add_argument("--save-quant", default=None,
                    help="losslessly save the quantized weights into this folder ({bk}_{variant}.pt). "
                         "If task heads are added later, the columns can be filled in without requantizing.")
    ap.add_argument("--eval-cap", type=int, default=None,
                    help="cap on er/ks/ic/sid eval samples. For smoke tests. Default is talq.eval.arm_eval.CAP")
    ap.add_argument("--eval-split", default="test-clean",
                    help="LibriSpeech split to report on. The task head was selected on dev-clean, "
                         "so the default is test-clean. Use dev-clean to reproduce past results.")
    ap.add_argument("--sv-stride", type=int, default=4)
    ap.add_argument("--smooth-win", type=int, default=1,
                    help="temporal moving-average window (frames) applied to the "
                         "per-frame task-gradient weight before it enters H; "
                         "1 = off (raw per-frame, original behavior). Layer-adaptive: "
                         "forced to 1 on the first third of layers regardless (see "
                         "adaptive_win) since smoothing does nothing there.")
    ap.add_argument("--percdamp", type=float, default=0.01,
                    help="GPTQ damping for task-weighted H only (recon baseline "
                         "keeps the class default 0.01). Task-H is likely worse-"
                         "conditioned (CTC gradient spike concentration), so a "
                         "higher value shrinks more toward isotropic/uniform.")
    a = ap.parse_args()
    bits_list = [int(b) for b in a.bits_list.split(",")]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    specs = None
    if a.spec_json:
        specs = json.load(open(a.spec_json))
        budgets = [float(x) for x in a.budgets.split(",")]

    # Giving either one takes the reweight path. Giving neither runs byte-identically
    # to the original code -- so as not to invalidate the fisher_* rows already collected.
    rw = None if (a.power_gamma is None and a.hessian_alpha is None) else (
        1.0 if a.power_gamma is None else a.power_gamma,
        1.0 if a.hessian_alpha is None else a.hessian_alpha)
    if rw is not None:
        g, al = rw
        prefix = ("" if a.power_gamma is None else f"pow{g:g}") + \
                 ("" if a.hessian_alpha is None else f"hyb{al:g}")
        print(f"reweight: gamma={g:g} alpha={al:g} -> variant prefix '{prefix}'", flush=True)
    else:
        prefix = "fisher"
    # If the tuning/seed is not at its default, stamp it into the variant name so it does
    # not mix with the existing rows. No underscores -- the parser splits the variant on
    # '_b{bits}'/'_m{budget}'.
    if a.fisher_type != "emp":
        prefix += a.fisher_type.upper()
    if a.gproj > 0:
        prefix += f"P{a.gproj}"
    if a.shrink_ess > 0:
        prefix += f"E{a.shrink_ess:g}".replace(".", "")
    if a.fisher_calib != "task":
        prefix += "CE"
    if a.smooth_win > 1:
        prefix += f"S{a.smooth_win}"
    if a.percdamp != 0.01:
        prefix += f"D{a.percdamp:g}".replace(".", "")
    if a.seed != 0:
        prefix += f"R{a.seed}"
    print(f"variant prefix = '{prefix}'  (smooth_win={a.smooth_win} "
          f"percdamp={a.percdamp:g} seed={a.seed})", flush=True)

    recon_calib = load_calib(a.calib, a.emilia, dev)

    from talq.eval import arm_eval
    if a.eval_cap:
        arm_eval.CAP.update({t: a.eval_cap for t in ("er", "ks", "ic", "sid")})
    eval_data = arm_eval.load_eval_data(a.data, sv_stride=a.sv_stride)

    # The train-side samples on which the task-Fisher weight w = ||dL_task/dy||^2 is
    # computed. Drawn from TRAIN, not from the eval set -- building the Hessian from eval
    # data is leakage, plain and simple.
    from talq.eval.superb_data import ic_index, ks_index, load_lexicon_official, voxceleb_sid_index
    from talq.eval.probe_train import EMO_TRAIN, IEMOCAP_MANIFEST, iemocap_index
    LS = f"{a.data}/librispeech/LibriSpeech"
    lex_o, phones_o = load_lexicon_official()
    pools = {
        "pr":  librispeech_index(f"{LS}/train-clean-100", "pr", lex_o, phones=phones_o),
        "asr": librispeech_index(f"{LS}/train-clean-100", "asr"),
        "er":  iemocap_index(IEMOCAP_MANIFEST, EMO_TRAIN),
        "ks":  ks_index("train")[0],
        "ic":  ic_index("train")[0],
    }
    vox_pool, _ = voxceleb_sid_index(f"{a.data}/voxceleb1/dev")
    pools["sid"] = pools["asv"] = vox_pool
    print("calib pools: " + "  ".join(f"{t}={len(v)}" for t, v in pools.items()), flush=True)

    fresh = not os.path.exists(a.csv) or os.path.getsize(a.csv) == 0
    done = set()
    if not fresh:
        for r in csv.DictReader(open(a.csv)):
            done.add((r["backbone"], r["variant"]))
    fh = open(a.csv, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if fresh:
        w.writeheader(); fh.flush()

    for bk in a.backbones.split(","):
        print(f"=== {bk} ===", flush=True)
        m = AutoModel.from_pretrained(BACKBONES[bk], dtype=torch.float32).eval().to(dev)
        m.requires_grad_(False)
        if bk.startswith("wavlm"):   # wavlmL too: stock WavLM hides q/k/v from hooks
            patch_wavlm_eager(m, recon_calib[:2])
        probes = arm_eval.load_probes(BACKBONES[bk], a.probes, dev)
        if not probes:
            raise SystemExit(f"{bk}: no task head at all in probes/")
        print(f"  probe: {sorted(probes)}", flush=True)
        orig = {k: v.clone() for k, v in m.state_dict().items()}
        nl = len(m.encoder.layers)

        def evalall(variant=None):
            if a.save_quant and variant:
                from talq.quant.store import save_quant
                save_quant(m, f"{a.save_quant}/{bk}_{variant}.pt",
                           {"backbone": bk, "variant": variant, "group": a.group})
            return arm_eval.eval_all(m, probes, eval_data, dev, n_ctc=a.n_eval)

        # only build batches for tasks that have a task head (without one the Fisher cannot be computed anyway)
        want_f = (FISHER_TASKS if a.fisher_tasks is None
                  else [t for t in a.fisher_tasks.split(",") if t])
        # the spec loop is independent of fisher. Even when running recon only with
        # --fisher-tasks "", which task's bit allocation to use still has to be decided.
        want_spec = (want_f if a.spec_tasks is None
                     else [t for t in a.spec_tasks.split(",") if t])
        task_batches = {}
        if a.fisher_calib == "emilia":
            if a.fisher_type != "mc":
                raise SystemExit("--fisher-calib emilia has no labels. "
                                 "It must be used together with --fisher-type mc.")
            # Emilia 48 clips x 2 s. None goes in the label slot -- the mc path draws its
            # target from the model, so it never looks at tgt.
            eb = []
            for i in range(0, len(recon_calib), a.bs):
                xb = recon_calib[i:i + a.bs]
                eb.append((xb, torch.full((len(xb),), xb.shape[-1],
                                          dtype=torch.long), None))
            for t in want_f:
                if t in probes:
                    task_batches[t] = eb
        for t in want_f if a.fisher_calib == "task" else []:
            if t not in probes:
                continue
            if t in ("pr", "asr"):
                task_batches[t] = make_ctc_batches(pools[t], dev, n=a.n_calib,
                                                   bs=a.bs, seed=a.seed)
            else:
                task_batches[t] = make_sv_batches(pools[t], dev, n=a.n_calib, bs=a.bs,
                                                  seed=a.seed, max_s=CLF_MAX_S[t])
        print(f"  fisher targets: {sorted(task_batches)}", flush=True)
        # (tag, per-layer bit vector, which fisher tasks to pair with it)
        jobs = []
        if specs is None:
            for bits in bits_list:                       # uniform: arms 1 + 3
                jobs.append((f"b{bits}", [bits] * nl,
                             [t for t in want_f if t in probes]))
        else:
            for bud in budgets:                          # mixed: arms 2 + 4
                for task in [t for t in want_spec if t in probes]:
                    key = f"{bk}|{task}|{bud}"
                    if key not in specs:
                        print(f"  [skip] no spec: {key}", flush=True); continue
                    vec = specs[key]["spec"]
                    if len(vec) != nl:
                        raise SystemExit(f"spec length {len(vec)} != number of layers {nl} ({key})")
                    # in arm2, which does not run fisher, only the recon variant remains
                    jobs.append((f"a{task}_m{bud:g}", vec,
                                 [task] if task in want_f else []))

        for tag, bitvec, fisher_tasks in jobs:
            # --- base-H arm: reconstruction-Hessian GPTQ on this bit vector ---
            recon_variant = f"recon_{tag}"
            # skipped in the reweight sweep. The Emilia-calib recon is already in
            # arm1 (gptq_uniform.csv), and every parallel job would write the same row twice.
            if (rw is None and prefix == "fisher"
                    and (bk, recon_variant) not in done):
                m.load_state_dict(orig)
                for k, layer in enumerate(m.encoder.layers):
                    quantize_layer_recon(m, layer, recon_calib, bits=bitvec[k], group=a.group)
                res = evalall(recon_variant)
                print(f"  [{bk} {recon_variant}] " + "  ".join(
                    f"{arm_eval.METRIC[t]}={v:.4f}" for t, v in res.items()), flush=True)
                w.writerow(arm_eval.row(bk, recon_variant, res))
                fh.flush()

            # --- task-Fisher variants: content (PER, linear probe) + speaker (AAM) ---
            for task in fisher_tasks:
                variant = f"{prefix}_{task}_{tag}"
                if (bk, variant) in done:
                    continue
                m.load_state_dict(orig)
                for k, layer in enumerate(m.encoder.layers):
                    quantize_layer_taskfisher(
                        m, layer, task_batches[task], probes[task], task, bitvec[k], a.group, dev,
                        smooth_win=a.smooth_win, percdamp=a.percdamp, layer_idx=k, nl=nl,
                        rw=rw, fisher_type=a.fisher_type, gproj=a.gproj,
                        shrink_ess=a.shrink_ess)
                    print(f"  [{bk} {variant}] layer {k}/{nl} done", flush=True)
                res = evalall(variant)
                print(f"  [{bk} {variant}] " + "  ".join(
                    f"{arm_eval.METRIC[t]}={v:.4f}" for t, v in res.items()), flush=True)
                w.writerow(arm_eval.row(bk, variant, res))
                fh.flush()
        del m
        torch.cuda.empty_cache()

    fh.close()
    print("TASK_FISHER_ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
