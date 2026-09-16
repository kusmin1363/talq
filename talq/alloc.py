"""Three-way (W2/W3/W4) differentiable bit allocation -- replaces the earlier 2-way allocator.

Why do it again: the earlier allocator (i) was binary 2/4-bit, so it could never pick 3-bit,
(ii) trained with per-layer probes (only pr/sv/ser existed), so it looked at different
layers than the probe that scores an arm, and (iii) used a soft budget penalty, so the
budget drifted after hard rounding. All three are blocked here.

  (i)   p_k = softmax(alpha_k) in R^3 over {2,3,4}.  W_eff = sum_b p_b * W_b.
        The forward is a straight-through argmax (the hard choice that will actually be
        deployed), the backward is soft.
        It mixes the per-layer pre-quantized quants/{bk}_L{k}_b{b}.pt directly, so there is
        no re-quantization.
  (ii)  The task loss is measured with the SUPERB probe of probes/. That is the very head
        that scores an arm.
  (iii) The budget is **not matched** (--project none is the default). The average bit width
        the training landed on is used as is, and **compared against the uniform model at
        ceil(avg)** -- a conservative comparison in which we spend fewer bits and compete
        against the side that spends more. The DP projection is left in but not recommended
        (see the 2026-08-28 entry below).

Training itself is a surrogate. So the final numbers are re-measured on the **full official
SUPERB test split** (setup_full/eval_full, the same evaluation protocol as the standalone allocation
evaluator and talq.eval.grid_eval).

2026-08-28 revision -- every earlier run is treated as a smoke run and re-measured. Three fixes.

  Data budget: 200x8=1,600 draws **with replacement**, so only about 1,500 unique utterances
      remained. The ASV (1,211-way AMSoftmax) loss did not go down at all over the 200 steps,
      so the observation "SID/ASV demand the most bits" could not be told apart from an
      artifact of undertraining. Changed to epoch sampling without replacement and the default
      raised to 2,000x16=32,000.
      The task calib (calib/task_*.pt) cannot be used -- it stores waveforms only so it has no
      labels, and with 48 clips (96 s) it is 33x smaller instead. It uses the same indexer, so
      only the selection protocol was matched.

  Initialization: with uniform (1/3,1/3,1/3), W_eff is a convex combination, so independent
      rounding errors cancel and **an undeployable interior point was 12% more accurate than
      uniform 3-bit** (w2v2 measured 0.340 vs 0.385). Training then starts from that favorable
      point and the loss rises as it moves toward one-hot.
      Starting from (0.05,0.05,0.90) shrinks the bias to 0.94x (0.169 vs W4 0.179), and the
      compression is pulled down by the budget penalty. An exact one-hot is forbidden -- see
      the init_alpha comment.
      Accordingly the entropy penalty was also deferred to the later half (--ent-start). From a
      confident init the path down to lower bits passes through high entropy, so turning it on
      early blocks that move.

  forward: soft mixing -> **gumbel** (default). Annealing tau from 1.0 to 0.1 makes the forward
      nearly one-hot so the blend bias disappears, and unlike ste it **does not linearize**, so
      second-order effects survive. As a side effect avg_bits approaches the realized bits, so
      the budget term finally means the actual budget. The difference between the three is in
      the mix_probs comment.

  Entropy: beta default 0.05 -> **0 (removed)**. Its reason for existing was not "forcing a
      decision" but patching the convex-combination bias, and since the 4-bit init and gumbel
      remove that bias directly, its role is gone. Neither the DP projection nor argmax
      requires one-hot.

  DP projection discarded: training sits at avg 3.25 but DP cut it to exactly 3.0. DP fills
      that gap by reading **the tail of p** (the non-argmax probabilities, the region that is
      estimated worst and that the forward has never once evaluated). Measured: two runs whose
      argmax was completely identical (s600/s3600 both 333434444322) split 3.3x after the DP
      projection, PER 0.2425 vs 0.0739. The cost of demoting L3 to 3-bit differed 900x,
      -log(0.0005)=7.6 vs -log(0.4468)=0.81, and the expensive side instead demoted the most
      sensitive L6 by two steps, 4->2.
      => The budget is not matched; the comparison is against the ceil(avg) uniform.

  Evaluation: the even/odd A/B split and the sample caps (n_ctc 32, CAP 500, sv_stride 8) are
      all removed and it goes with the **full official test split**. Subsamples vary so much
      that they bury the differences between allocations -- even at sv_stride 4 the bootstrap
      SD of ASV EER was 3.8%. Training does not look at eval, so there was no reason to split
      in the first place (the same judgment as talq.eval.grid_eval).
      The baselines are re-measured with the same evaluation protocol, so archive's
      uniform_ab.json is not used.

  python -m talq.alloc --backbones w2v2 --tasks pr --smoke
  python -m talq.alloc --backbones w2v2 --tasks pr,asr,er,ks,ic,asv   # 6-task check
"""
import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
from torch.func import functional_call

import talq.search as A
from talq import tb_log
import talq.eval.salience as S
from talq.eval.probe_train import load_wav, pad_batch
from talq.paths import CKPT_ROOT, DATA_ROOT, QUANT_ROOT, RESULTS_ROOT, TB_ROOT

BITS = [2, 3, 4]                   # --bitset overwrites this in main()


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# -------------------------------------------------------------- weight mixing
class Mixer:
    """Per-layer three-way mixing. Keeps the per-layer quantized weights of quants/ on the
    GPU and builds W_eff = sum_b p_b * W_b. The name set is identical across the three bit widths."""

    def __init__(self, bk, nl, dev, qdir="quants"):
        LQ = load_layer_quants(bk, nl, qdir)
        self.dev, self.nl = dev, nl
        # The names differ per layer (encoder.layers.{k}...). What must match is across bit widths.
        self.names = {k: sorted(LQ[(k, BITS[0])]) for k in range(nl)}
        self.W = {}
        for k in range(nl):
            for b in BITS:
                assert sorted(LQ[(k, b)]) == self.names[k], f"L{k} b{b} name mismatch"
                self.W[(k, b)] = {n: LQ[(k, b)][n].to(dev) for n in self.names[k]}
        del LQ

    def overrides(self, p_ste):
        """p_ste: (nl, 3). -> {param_name: mixed tensor}"""
        out = {}
        for k in range(self.nl):
            for n in self.names[k]:
                w = sum(p_ste[k, i] * self.W[(k, b)][n] for i, b in enumerate(BITS))
                out[n] = w
        return out


def load_layer_quants(bk, nl, qdir):
    """{qdir}/{bk}_L{k}_b{b}.pt -> {(k,b): {param_name: tensor}}.

    talq.search.load_layer_quants has quants/ hard-coded, so it is written again here.
    quants (GPTQ) and quants_awq (AWQ) follow the **same protocol**: same parameter names,
    same calib (emilia48x2s), per-layer isolated (the rest an fp32 prefix). That is why the
    quantizer axis can be switched with this single argument.
    """
    q = {}
    for k in range(nl):
        for b in BITS:
            fp = QUANT_ROOT / qdir / f"{bk}_L{k}_b{b}.pt"
            if not os.path.exists(fp):
                raise SystemExit(f"missing: {fp}")
            q[(k, b)] = torch.load(fp, map_location="cpu",
                                   weights_only=False)["weights"]
    return q


def mix_probs(alpha, mode, tau=1.0):
    """Return the mixing coefficients for the forward together with the softmax that records the preference.

    soft   : W_eff is a real interpolation. The gradient passes through the actual forward, so
             second- and higher-order effects survive. But it is measured at an undeployable
             interior point -- independent rounding errors cancel, so the blend is better than
             reality (w2v2 measured 0.94x).

    gumbel : y = softmax((alpha + Gumbel)/tau). Lowering tau makes the forward nearly one-hot so
             the blend bias disappears, but **it does not linearize** -- it still passes through
             the actual forward. It keeps soft's advantage and removes only its drawback.
             As a side effect avg_bits approaches the realized bits, so the budget term finally
             means the actual budget (under soft it was an expectation over a distribution that
             is never sampled).

    ste    : the forward is one-hot, the backward is soft. **Do not use.**
             Since dW_eff/dp_b = W_b, dL/dp_b = <g, W_b>, and splitting W_b = W_fp32 + D_b,
             <g, W_fp32> does not depend on b and cancels in the softmax gradient; what remains
             is <g, D_b> -- **the sign-preserving first-order Taylor term** itself.
             The sign of D is random so it cancels, which fails to measure damage (second order,
             always positive) and instead rewards 'layers aligned with the gradient'. It failed
             twice independently: here w2v2/pr uniform 0.090 -> 0.206, and the g1w1s arm of the
             DP damage-formula ablation with a win rate of 1/21 and a mean of -186.3%.
    """
    soft = F.softmax(alpha, dim=-1)
    if mode == "soft":
        return soft, soft
    if mode == "gumbel":
        return F.gumbel_softmax(alpha, tau=tau, hard=False, dim=-1), soft
    hard = F.one_hot(soft.argmax(-1), len(BITS)).to(soft.dtype)
    return hard + soft - soft.detach(), soft


def tau_at(step, steps, t0, t1):
    """Geometric annealing. Smooth early (to secure signal), nearly one-hot late (to match deployment)."""
    if steps <= 1:
        return t1
    return t0 * (t1 / t0) ** ((step - 1) / (steps - 1))


# ---------------------------------------------------- exact budget projection
def project_budget(cost, total):
    """From the per-layer cost cost[k][i] (smaller is preferred), find exactly, by DP, an allocation with sum(bit) == total.
    The same separable resource-allocation DP as the earlier ranking-based allocator."""
    nl = len(cost)
    INF = float("inf")
    dp = [[INF] * (total + 1) for _ in range(nl + 1)]
    back = [[None] * (total + 1) for _ in range(nl + 1)]
    dp[0][0] = 0.0
    for k in range(nl):
        rem = nl - k - 1
        for s in range(total + 1):
            if dp[k][s] == INF:
                continue
            for i, b in enumerate(BITS):
                ns = s + b
                if ns > total or ns + 4 * rem < total or ns + 2 * rem > total:
                    continue
                c = dp[k][s] + cost[k][i]
                if c < dp[k + 1][ns]:
                    dp[k + 1][ns], back[k + 1][ns] = c, (s, b)
    assert dp[nl][total] < INF, f"budget {total} unreachable"
    vec, s = [0] * nl, total
    for k in range(nl, 0, -1):
        s, b = back[k][s]
        vec[k - 1] = b
    return vec


# --------------------------------------------------- full official evaluation
METRIC = {"pr": "PER", "asr": "WER", "ks": "KS_ERR", "ic": "IC_ERR",
          "er": "ER_ERR", "asv": "EER"}


def setup_full(bk, dev, cap=None):
    """Prepare the evaluation assets on the **full** official SUPERB test split.

    Every knob that reduced the sample is removed (the same protocol as talq.eval.grid_eval):
        eval_ctc n=32  -> all of test-clean (PR 2,418 / ASR 2,608)
        CAP 500        -> all of ks 3,081 / ic 3,793
        sv_stride 8    -> all official ASV trials (voxceleb1_test_v2.txt)
        no A/B split   -> training does not look at eval, so there is no reason to split

    Subsamples vary so much in value that they bury the differences between allocations. Even at
    sv_stride 4 the bootstrap SD of ASV EER was 3.8%.
    ER alone is **fold5 only** rather than the SUPERB 5-fold average -- because of leakage.
    See the eval_full comment. So the ER numbers cannot be compared directly with the ER column
    of grid6.csv.
    """
    import talq.eval.sensitivity as SO
    import talq.quant.task_fisher as _T
    from talq.eval.grid_eval import official_eval, setup
    from talq.eval.probe_train import iemocap_index, IEMOCAP_MANIFEST

    m, _u, orig = setup(bk, dev)
    probes, er_probes = SO.load_probes6(_T.BACKBONES[bk], dev)
    d = official_eval(DATA_ROOT, cap=cap)
    d["er"] = {n: iemocap_index(IEMOCAP_MANIFEST, {f"Session{n}"}) for n in SO.FOLDS}
    if cap:
        d["er"] = {n: v[:cap] for n, v in d["er"].items()}
    return m, probes, er_probes, d, orig, len(m.encoder.layers)


def migrate_uniform(res):
    """Old schema uniform={task: v} -> new schema uniform={"3": {task: v}}.

    Old runs measured only the single bit width equal to the budget. With project=none we do not
    know where the allocation will land, so it has to be held per bit width.
    """
    u = res.get("uniform", {})
    if u and not all(k.isdigit() for k in u):
        res["uniform"] = {str(int(res.get("budget", 3))): u}


def baseline_for(res, avg_bit, task):
    """Compare against the uniform model at the **ceiling** of the allocation's average bit width.

    At avg 3.25 it faces uniform 4-bit -- a conservative comparison in which we use 0.75 fewer
    bits and compete against the side that uses more. Winning or tying makes "the same
    performance with fewer bits" a claim as it stands. Comparing at the rounded value (3) would
    make it a win while using more bits, which weakens the claim.
    """
    ub = min(max(math.ceil(avg_bit - 1e-9), min(BITS)), max(BITS))
    return ub, res.get("uniform", {}).get(str(ub), {}).get(task)


def train_probe_of(task, probes, er_probes, er_fold=None):
    """One probe for alpha training. Only er has five folds, so it has to be chosen.

    er_fold=n uses the probe of fold n. That probe's training sessions are the four with Session n
    removed, and S.train_pool("er", er_fold=n) gives the same four -- the calib (the fold n
    version), the quantization candidates (fold n calib), the probe, the training pool and the
    evaluation are all aligned on the same fold.

    er_fold=None is the old behavior (fold5 only). Only then is it safe -- the training pool is
    fixed to EMO_TRAIN (Session1~4), so scoring fold1~4 puts their own test session into the pool.
    To take the 5-fold average, er_fold must be passed.
    """
    if task == "er":
        return er_probes[er_fold if er_fold in er_probes else max(er_probes)]
    return probes[task]


def eval_full(task, m, probes, er_probes, d, dev, er_fold=None):
    """Identical to the standalone allocation evaluator. Makes the two lines of work use the same evaluation protocol."""
    import talq.eval.sensitivity as SO
    from talq.eval.grid_eval import BIG
    from talq.eval.probe_train import eval_ctc

    if task in ("pr", "asr"):
        items, itos, sep = d[task]
        return eval_ctc(m, probes[task], items, dev, itos, sep, n=BIG)
    if task == "ks":
        return SO.clf_err_nocrop(m, probes["ks"], d["ks"], dev)
    if task == "ic":
        return SO.clf_err_nocrop(m, probes["ic"], d["ic"], dev, slots=SO.IC_SLOTS)
    if task == "er":
        # er_fold=n scores with Session n. It is valid only when that fold's calib/probe/training
        # pool were all built with Session n removed (see train_probe_of).
        # None means fold5 only -- the training pool is fixed to Session1~4, so only fold5 is leak-free.
        f = er_fold if er_fold in er_probes else max(er_probes)
        return SO.clf_err_nocrop(m, er_probes[f], d["er"][f], dev)
    tri, root = d["asv"]
    return SO.eval_sv_nocrop(m, probes["asv"], root, tri, dev)


# ------------------------------------------------------- label-free objective
def probe_out(task, probe, hs, model, lens, dev):
    """Produce the per-task 'output that makes a distribution'. Only asv is an embedding.

    The KL objective takes fp32 itself, not the ground truth, as the reference, so no labels are
    needed. The resource consumed becomes unlabeled audio, the same kind as the GPTQ/AWQ calib.
    """
    if task in ("pr", "asr"):
        # If lens is on CPU the result is on CPU too. The mask has to be made on the same device as the logits.
        in_lens = model._get_feat_extract_output_lengths(lens).to(torch.long).to(dev)
        return probe(hs), in_lens                    # (B,T,V)
    if task == "asv":
        flens = model._get_feat_extract_output_lengths(lens).to(torch.long)
        return probe.embed(hs, flens.tolist()), None  # (B,H) embedding
    return probe(hs), None                            # (B,C) logits


def kl_loss(task, ref, cur):
    """KL(fp32 || quant). The same definition as talq.eval.sensitivity._kl, used differentiably.

      pr/asr : averaged over the frame axis. Padding frames are excluded by a mask -- beyond the
               valid length the two models see the same garbage, so KL is diluted to 0 and the
               loss scale wobbles per batch with the length distribution.
      ic     : measured separately per slot (6,14,4) and summed. Putting a softmax on the
               concatenated 24 dimensions creates competition between slots, which gives a
               different distribution than the actual head.
      asv    : AMSoftmax needs labels for training and the evaluation is cosine EER. There is no
               distribution, so the cosine distance of the embeddings is used (identical to
               COS_asv of cond_sens).
    """
    (p_lg, in_lens), (q_lg, _) = ref, cur
    if task == "asv":
        return (1.0 - F.cosine_similarity(p_lg.float(), q_lg.float(), dim=-1)).mean()
    p_lg, q_lg = p_lg.float(), q_lg.float()
    if task == "ic":
        tot, off = 0.0, 0
        for sz in S.IC_SLOTS:
            p = F.log_softmax(p_lg[..., off:off + sz], -1)
            q = F.log_softmax(q_lg[..., off:off + sz], -1)
            tot = tot + (p.exp() * (p - q)).sum(-1).mean()
            off += sz
        return tot
    p = F.log_softmax(p_lg, -1)
    q = F.log_softmax(q_lg, -1)
    kl = (p.exp() * (p - q)).sum(-1)                  # (B,) or (B,T)
    if in_lens is None:
        return kl.mean()
    T = kl.shape[1]
    mask = (torch.arange(T, device=kl.device)[None, :] < in_lens[:, None]).float()
    return (kl * mask).sum() / mask.sum().clamp_min(1.0)


# ------------------------------------------------------------- initialization
def init_alpha(nl, spec, dev):
    """Build alpha from a specified initial p. The default piles it on the 4-bit side and pulls it
    down with the budget penalty.

    Why starting from uniform (1/3,1/3,1/3) is bad: W_eff is a convex combination of the three
    quantization results, so independent rounding errors cancel. The measured relative error on
    w2v2 is 0.340 for the uniform blend vs 0.385 for W3 alone, i.e. **an undeployable interior
    point is 12% more accurate than uniform 3-bit.** Training starts from that favorable point and
    sees the loss rise as it moves toward one-hot.
    With (0.05,0.05,0.90) it is 0.169 vs 0.179 for W4 alone, so the bias shrinks to 0.94x.

    p=0 is forbidden. In the softmax parameterization p_b=0 means alpha_b=-inf, and
    d p_b/d alpha ~ p_b(1-p_b) = 0, so the gradient toward that bit width dies.
    That is, starting from an exact one-hot [0,0,1] removes the path down to lower bits.
    """
    if spec == "uniform":
        p = torch.full((len(BITS),), 1.0 / len(BITS))
    else:
        p = torch.tensor([float(x) for x in spec.split(",")])
        assert len(p) == len(BITS), f"--init must have {len(BITS)} values: {BITS}"
        assert (p > 0).all(), (
            "--init cannot contain 0. An exact one-hot kills the gradient of that bit "
            "width and removes the path to move along. Use a small positive value such as 0.01~0.05.")
        p = p / p.sum()
    a0 = p.log()
    a0 = a0 - a0.max()                       # normalize to alpha_max = 0 (numerical stability)
    return a0.repeat(nl, 1).to(dev).clone().requires_grad_(True)


def batch_stream(pool, bs, steps, seed):
    """Go around epochs with sampling without replacement and yield (step, chunk).

    It used to be `pool[rng.randrange(len(pool))]` at every step, i.e. sampling **with
    replacement**, so at 200x8 only about 1,500 unique utterances came out. Changed to the same
    protocol as the task calib (seed shuffle -> take in order) so that min(steps*bs, len(pool))
    items are seen **without omission**.
    """
    rng = random.Random(seed)
    bs = min(bs, len(pool))                  # the pool can be smaller than the batch (96 s parity)
    idx, cur = list(range(len(pool))), 0
    rng.shuffle(idx)
    for _ in range(steps):
        if cur + bs > len(idx):              # end of epoch -> reshuffle
            rng.shuffle(idx); cur = 0
        yield [pool[i] for i in idx[cur:cur + bs]]
        cur += bs


# ------------------------------------------------------------------- training
def train_alpha(m, probe, mixer, task, pool, a, dev, tb=None):
    # The ASR head is a 2-layer BLSTM. cudnn refuses RNN backward in eval mode, so it is kept in
    # train mode only during the training section. These probes have no dropout/BatchNorm, so the
    # forward behavior is identical even when the mode changes. The same prescription as talq.eval.salience.
    # The ASR head is a 2-layer BLSTM. cudnn refuses RNN backward in eval mode, so it is kept in
    # train mode only during the training section. These probes have no dropout/BatchNorm, so the
    # forward behavior is identical even when the mode changes. The same prescription as talq.eval.salience.
    # It is needed for the KL objective too -- the reference (fp32) is under no_grad, but the
    # comparison side goes through the probe and backpropagates to alpha, so the RNN backward happens all the same.
    probe.train() if task == "asr" else probe.eval()
    nl = mixer.nl
    # Two global RNGs are involved. Without seeding, running the same command twice gives
    # different results, and it also changes if only the task order changes (an earlier task consumes the RNG).
    #   torch  : the Gumbel noise of gumbel_softmax
    #   random : the random crop of talq.eval.probe_train.load_wav(max_s=) (batch_stream's Random(seed)
    #            only fixes the batch 'order'; the crop position uses the global one)
    # Re-seed here for every task so that a run is determined by (seed, task) alone.
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    alpha = init_alpha(nl, a.init, dev)
    opt = torch.optim.Adam([alpha], lr=a.lr)
    tb = tb or tb_log.TB.off()
    max_s = S.MAX_S[task]
    ent_from = int(a.ent_start * a.steps)
    seen = set()
    hist = []
    with torch.no_grad():
        p0 = F.softmax(alpha[0], -1)
        avg0 = float((p0 * torch.tensor([float(b) for b in BITS], device=dev)).sum())
    log(f"    init p={[round(v, 3) for v in p0.tolist()]} (avg {avg0:.3f}bit) "
        f"| pool {len(pool):,} | budget {a.steps}x{a.bs}={a.steps * a.bs:,} draw "
        f"| ent from step {ent_from}")
    for step, chunk in enumerate(batch_stream(pool, a.bs, a.steps, a.seed), 1):
        seen.update(c[0] for c in chunk)
        x, lens = pad_batch([load_wav(c[0], max_s=max_s) for c in chunk], dev)
        if a.objective == "kl":
            # The fp32 reference. One pass with the original weights and no override -- it must
            # see the same batch so that KL measures only the quantization difference and does not
            # mix in the audio difference.
            with torch.no_grad():
                ref = probe_out(task, probe,
                                torch.stack(m(x, output_hidden_states=True).hidden_states),
                                m, lens, dev)
        tau = tau_at(step, a.steps, a.tau_start, a.tau_end)
        p_ste, p_soft = mix_probs(alpha, a.forward, tau)
        ov = mixer.overrides(p_ste)
        out = functional_call(m, ov, (x,), {"output_hidden_states": True})
        hs = torch.stack(out.hidden_states)
        if a.objective == "kl":
            tl = kl_loss(task, ref, probe_out(task, probe, hs, m, lens, dev))
        else:
            tl = S.task_loss(task, probe, hs, m, lens, chunk, dev)
        bits = torch.tensor([float(b) for b in BITS], device=dev)
        avg = (p_ste * bits).sum(-1).mean()
        ent = -(p_soft.clamp_min(1e-9).log() * p_soft).sum(-1).mean()
        # The entropy penalty only in the later half. Going from a confident init down to lower
        # bits necessarily passes through a high-entropy region, and turning it on from the start blocks that move.
        beta = a.beta if step >= ent_from else 0.0
        # The rate term. linear is the default -- since the DP projection was dropped, this term
        # alone does the job of actually pulling the bits down. (avg-B)^2 is two-sided, so it also blocks going below B.
        rate = avg if a.rate == "linear" else (avg - a.budget) ** 2
        loss = tl + a.lam * rate + beta * ent
        opt.zero_grad(); loss.backward(); opt.step()
        # TensorBoard records more densely than the log (default every 20 steps). .item() causes a
        # synchronization, but it is negligible against one step of a large backbone (hundreds of ms).
        if step % a.tb_every == 0 or step == 1:
            tb.scalar("train/task_loss", tl.item(), step)
            tb.scalar("train/loss", loss.item(), step)
            tb.scalar("train/avg_bits", avg.item(), step)
            tb.scalar("train/entropy", ent.item(), step)
            tb.scalar("train/rate", rate.item() if torch.is_tensor(rate) else rate, step)
            tb.scalar("sched/tau", tau, step)
            tb.scalar("sched/beta", beta, step)
            with torch.no_grad():
                pm = p_soft.mean(0)
            for i, b in enumerate(BITS):
                tb.scalar(f"prob/p{b}", pm[i].item(), step)
        if step % a.log_every == 0 or step == 1:
            hard = [BITS[i] for i in p_soft.argmax(-1).tolist()]
            log(f"    step {step:4d}  task_loss {tl.item():.4f}  avg_bits {avg.item():.3f}  "
                f"tau {tau:.3f}  ent {ent.item():.3f}  alloc {''.join(map(str, hard))}")
            hist.append({"step": step, "task_loss": tl.item(), "avg_bits": avg.item(),
                         "tau": tau, "ent": ent.item(), "alloc": hard})
    log(f"    unique utterances {len(seen):,} / requested {a.steps * a.bs:,} (pool {len(pool):,})")
    tb.scalar("data/unique_utts", len(seen), a.steps)
    tb.flush()
    with torch.no_grad():
        p = F.softmax(alpha, dim=-1)
    if a.project == "dp":
        cost = (-p.clamp_min(1e-9).log()).tolist()
        vec = project_budget(cost, A.budget_sum(nl, a.budget))
    else:
        # The budget is not matched exactly. The bits the training landed on by itself are used as is.
        vec = [BITS[i] for i in p.argmax(-1).tolist()]
    probe.eval()          # scoring is always in eval mode
    return vec, p.tolist(), hist, len(seen)


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", default="w2v2,wavlm,hubert")
    ap.add_argument("--tasks", default="pr,asr,er,ks,ic,sid,asv")
    ap.add_argument("--budget", type=float, default=3.0)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--init", default="0.05,0.05,0.9",
                    help="initial p (in 2,3,4bit order). 'uniform' means uniform. The default "
                         "starts from the 4-bit side and is pulled down by the budget penalty. "
                         "0 cannot be used -- the gradient of that bit width dies (see init_alpha).")
    ap.add_argument("--pool-cap", type=int, default=40000,
                    help="cap on the training pool (number of utterances). Sampling is without "
                         "replacement, so if steps*bs is larger than this it goes around epochs.")
    ap.add_argument("--pool-sec", type=float, default=0,
                    help="limit the training pool in **audio seconds**. 0 means --pool-cap. "
                         "This is the axis of the 'data budget' that has to be defended -- the "
                         "task calib equivalent is 96 s. The number of utterances is taken as "
                         "pool_sec / max_s(task) (a conservative count that over-states actual "
                         "consumption, since it is a crop upper bound). It is separate from the "
                         "optimization budget (--steps/--bs): going around a small pool for many "
                         "epochs is not using more data.")
    ap.add_argument("--ent-start", type=float, default=0.5,
                    help="when to turn on the entropy penalty (as a fraction of the total steps). "
                         "From a confident init the path to lower bits passes through high entropy, "
                         "so turning it on early blocks that move. 0 means from the start (old behavior).")
    ap.add_argument("--seed", type=int, default=0, help="batch order seed")
    ap.add_argument("--objective", choices=["kl", "task"], default="kl",
                    help="kl=KL(fp32||quant), no labels needed. The resource consumed becomes "
                         "unlabeled audio, the same kind as the GPTQ/AWQ calib. "
                         "task=ground-truth-based CTC/CE (old behavior). "
                         "Note: the KL for ER is weak as a surrogate "
                         "(rank correlation +0.10~0.73). Measure ER separately with task and compare.")
    ap.add_argument("--lam", type=float, default=0.3, help="budget penalty. The final budget is "
                    "guaranteed by the DP projection, so this only serves to keep training in the right region. "
                    "With a 4-bit init (avg 3.85) the initial penalty is lam*(3.85-3.0)^2, so "
                    "lam 10 gives 7.2 and overwhelms the task term. 0.3 gives 0.22, which is balanced.")
    ap.add_argument("--beta", type=float, default=0.0,
                    help="entropy penalty. **default 0 (off)**. The original purpose was 'forcing a "
                         "decision', but neither the DP projection nor argmax requires one-hot; they "
                         "use only the ranking of -log p. Its real reason for existing was to patch "
                         "the convex-combination bias, and since the 4-bit init and the gumbel forward "
                         "remove that bias directly, its role is gone. Kept only for reproduction.")
    ap.add_argument("--eval-cap", type=int, default=0,
                    help="cut the evaluation set to the first N items. **0 (the default) is full-split "
                         "evaluation** and the paper numbers must be 0. It is a smoke-only knob -- "
                         "subsamples vary so much that they bury the differences between allocations.")
    ap.add_argument("--er-fold", type=int, default=None, choices=[1, 2, 3, 4, 5],
                    help="the SUPERB fold for ER. Given n, the calib/probe/alpha training pool/"
                         "evaluation are all aligned on **the four sessions minus Session n + "
                         "test=Session n**. --quant-dir must also point at the one built with that "
                         "fold's calib. If omitted it is the old behavior (fold5 only), and only then "
                         "does the fixed training pool (Session1~4) produce no leakage.")
    ap.add_argument("--head", default=None, metavar="task=path[,task=path]",
                    help="replace the scoring head. It applies to **both training and evaluation** -- "
                         "the KL objective passes through the probe, so the head changes the allocation. "
                         "e.g. 'ks=probes/hubert-large-ll60k_ks_superb_baldev.pt'. "
                         "Do not write into the same out-dir as old-head results (the evaluation protocol differs).")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--tb-dir", default=os.environ.get("DA_TB_DIR", f"{TB_ROOT}"),
                    help="TensorBoard event root. An empty string turns it off. "
                         "It is also given by the environment variable DA_TB_DIR (the runner uses that).")
    ap.add_argument("--no-tb", action="store_true", help="turn off TensorBoard logging")
    ap.add_argument("--tb-every", type=int, default=20,
                    help="interval (in steps) for recording TensorBoard scalars. Kept dense "
                         "independently of --log-every -- a curve with 20 points shows nothing.")
    ap.add_argument("--out", default=f"{RESULTS_ROOT}/main")
    ap.add_argument("--forward", choices=["soft", "gumbel", "ste"], default="gumbel",
                    help="gumbel=nearly one-hot forward + gradient that passes through the actual forward. "
                         "soft=mixed forward (has the blend bias). "
                         "ste=do not use, it collapses to the first-order Taylor term (mix_probs comment).")
    ap.add_argument("--tau-start", type=float, default=1.0,
                    help="gumbel temperature start. Higher is smoother, so the signal is picked up well.")
    ap.add_argument("--tau-end", type=float, default=0.1,
                    help="gumbel temperature end. The lower it is, the closer the forward is to the deployed configuration.")
    ap.add_argument("--project", choices=["none", "dp"], default="none",
                    help="none (the default)=use the trained argmax as is, without projection. The "
                         "average bit width is where training landed, and it is compared against the "
                         "ceil(avg) uniform. "
                         "dp=a projection that matches the budget exactly. **Not recommended** -- "
                         "training sits at avg 3.25 but deployment is 3.0, and DP fills that gap by "
                         "reading the tail of p (the part that is estimated worst). In measurement, "
                         "two runs with completely identical argmax (s600/s3600) split into "
                         "PER 0.2425 vs 0.0739 after the DP projection.")
    ap.add_argument("--quant-dir", default="quants",
                    help="directory of the per-layer pre-quantization. **A {task} template can be used** -- "
                         "e.g. quants_tc_{task} reads quants_tc_pr, quants_tc_er "
                         "... per task (task calib). Then the uniform baseline is measured with the same "
                         "quantization too, so the calib becomes a controlled variable. "
                         "quants=GPTQ/emilia, quants_awq=AWQ/emilia. "
                         "The two follow the same protocol (same parameter names/calib/isolated), so "
                         "**the quantizer axis is switched with this single argument**. All 5 backbones "
                         "are covered. But the uniform baseline also comes out of the same dir, so "
                         "do not mix GPTQ results and AWQ results in one table.")
    ap.add_argument("--bitset", default="2,3,4",
                    help="the bit widths to use. Used for the deployability ablation -- the accelerated "
                         "kernels of GPTQModel 7.2.0 (Triton/Marlin/ExllamaV2/Machete/BitBLAS) "
                         "**support 3-bit in none of them** (TritonV2 is [2,4,8]). "
                         "The only thing that does 3-bit is TorchLinear, and that is a dequant->matmul "
                         "fallback, not a kernel. To attach measured latency it has to be run with "
                         "--bitset 2,4. Match the number of --init values as well.")
    ap.add_argument("--rate", choices=["linear", "quad"], default="linear",
                    help="linear (the default)=L = KL + lam*avg_bits. It is a pure rate-distortion "
                         "Lagrangian, so lam becomes the exchange rate 'how many bits one unit of KL "
                         "is traded for' and a lam sweep is directly the RD curve. No arbitrary "
                         "target budget B is needed. "
                         "quad=lam*(avg-B)^2. It is two-sided, so it cannot go below B -- "
                         "the form from the days when the DP projection matched the budget.")
    ap.add_argument("--uniform-bits", default="2,3,4",
                    help="the uniform bit widths at which to measure the baseline. With project=none "
                         "we do not know where the allocation will land, so the ceil(avg) candidates "
                         "are measured in advance.")
    ap.add_argument("--baseline-only", action="store_true",
                    help="measure only the uniform baseline and stop. Used to measure it once when "
                         "several runs share the same baseline, as in a lam sweep.")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    global BITS
    BITS = [int(x) for x in a.bitset.split(",") if x]
    assert len(BITS) >= 2, "--bitset must have 2 or more values"
    if a.init != "uniform" and len(a.init.split(",")) != len(BITS):
        # If the number of bits changes, init must change with it. It is piled on the last (the highest bit).
        eps = 0.05
        w = [eps] * (len(BITS) - 1) + [1.0 - eps * (len(BITS) - 1)]
        a.init = ",".join(f"{x:g}" for x in w)
        log(f"--init auto-adjusted to match --bitset {BITS} -> {a.init}")
    if a.smoke:
        a.steps, a.bs, a.log_every, a.eval_cap = 6, 4, 2, 40
    dev = "cuda"
    # Isolation for when several processes share one GPU (talq.sweep passes DA_MEM_GB).
    # Without this cap, over-allocation by one process **kills the neighboring processes** --
    # on 2026-09-08 a large 4-way run broke 13 of 20 cells that way. With the cap, the side that
    # overflows dies of OOM by itself, and the runner raises the reservation and retries.
    _mem_gb = float(os.environ.get("DA_MEM_GB", 0) or 0)
    if _mem_gb > 0:
        _tot = torch.cuda.get_device_properties(0).total_memory / 1024**3
        torch.cuda.set_per_process_memory_fraction(min(1.0, _mem_gb / _tot))
        log(f"  GPU cap {_mem_gb:g}GB ({_mem_gb/_tot:.0%} of {_tot:.1f}GB total)")
    os.makedirs(a.out, exist_ok=True)
    tasks = [t for t in a.tasks.split(",") if t]
    if a.no_tb:
        a.tb_dir = ""

    for bk in a.backbones.split(","):
        fp = f"{a.out}/{bk}_b{a.budget:g}.json"
        res = json.load(open(fp)) if os.path.exists(fp) else {
            "backbone": bk, "budget": a.budget, "allocs": {}, "probs": {},
            "eval": {}, "uniform": {}, "hist": {}, "avg": {}}
        log(f"===== {bk} =====")
        m, probes, er_probes, d, orig, nl = setup_full(bk, dev, a.eval_cap or None)
        if a.head:
            import talq.eval.arm_eval as AE
            for item in a.head.split(","):
                if not item.strip():
                    continue
                t_, path = item.split("=", 1)
                fp_ = path if os.path.isabs(path) else f"{CKPT_ROOT}/{path}"
                if not os.path.exists(fp_):
                    raise FileNotFoundError(f"head missing: {fp_}")
                probes[t_.strip()] = AE.build_probe(
                    t_.strip(), torch.load(fp_, map_location=dev,
                                           weights_only=False), dev)
                log(f"  head replaced {t_.strip()} <- {os.path.basename(fp_)}")
        # If --quant-dir has {task} in it, a different quantization (= task calib) is used per task.
        # Then the Mixer also has to be built anew per task, so it is managed with a cache.
        mx_cache = {}

        def get_mixer(task):
            # The ER candidates exist separately per fold (the calib is the four sessions minus Session f).
            # If --er-fold is given, er_fold{f} has to go into the {task} slot.
            key = (f"er_fold{a.er_fold}" if task == "er" and a.er_fold else task)
            qd = a.quant_dir.format(task=key) if "{task}" in a.quant_dir else a.quant_dir
            if qd not in mx_cache:
                for k in list(mx_cache):        # hold only one (1GB of GPU each)
                    del mx_cache[k]
                torch.cuda.empty_cache()
                mx_cache[qd] = Mixer(bk, nl, dev, qd)
            return mx_cache[qd]
        # er has 5 folds, so it is in er_probes separately. Filtering only by `t in probes` silently drops it.
        have = set(probes) | ({"er"} if er_probes else set())
        todo = [t for t in tasks if t in have and t not in res["allocs"]]
        miss = [t for t in tasks if t not in have]
        if miss:
            log(f"  [note] skipped, no probe: {miss}")
        log(f"  layers {nl}  probe {sorted(probes)}  remaining task {todo}")

        # Measure the uniform baseline per bit width. With project=none we do not know where the
        # allocation will land, so the baseline corresponding to ceil(avg) has to exist in advance.
        migrate_uniform(res)
        for ub in [int(x) for x in a.uniform_bits.split(",")]:
            slot = res["uniform"].setdefault(str(ub), {})
            need = [t for t in tasks if t in have and t not in slot]
            if not need:
                continue
            for t in need:
                # With a task calib the uniform baseline must be measured with that task's
                # quantization too -- otherwise the calib axis does not become a controlled variable.
                A.compose(m, orig, [ub] * nl, mixer_LQ(get_mixer(t)), dev)
                t0 = time.time()
                slot[t] = eval_full(t, m, probes, er_probes, d, dev, a.er_fold)
                log(f"  uniform{ub} {t:4s} {METRIC[t]}={slot[t]:.4f} "
                    f"({time.time()-t0:.0f}s)")
            json.dump(res, open(fp, "w"), indent=1)

        if a.baseline_only:
            log("  baseline only requested -- training skipped");
            del m, mx_cache; torch.cuda.empty_cache(); continue
        for t in todo:
            log(f"  --- {t} ---")
            try:
                run_task(m, probes, er_probes, get_mixer(t), t, orig,
                         res, d, a, dev, nl, fp)
            except Exception as e:
                import traceback
                log(f"  [FAIL] {bk}/{t}: {type(e).__name__}: {e}")
                traceback.print_exc()
        del m, mx_cache
        torch.cuda.empty_cache()
        log(f"  saved {fp}")
    # For correcting the reservation. allocated is the tensors, reserved is the total the caching
    # allocator took, and what the runner has to look at (= the value nvidia-smi shows) is
    # reserved + the CUDA context.
    log(f"  GPU peak allocated {torch.cuda.max_memory_allocated()/1024**3:.1f}GB "
        f"reserved {torch.cuda.max_memory_reserved()/1024**3:.1f}GB")
    print("DIFF3_DONE", flush=True)


def run_task(m, probes, er_probes, mixer, t, orig, res, d, a, dev, nl, fp):
    m.load_state_dict(orig)
    # The same indexer as the task calib (make_task_calib.py) -> the same training split.
    # The protocol is matched too: the first pool_cap items after the seed shuffle. But the calib
    # stores waveforms only and so has no labels, so the calib .pt itself cannot be used here;
    # the index is used directly.
    pool = S.train_pool(t, DATA_ROOT, er_fold=a.er_fold)
    random.Random(0).shuffle(pool)
    if a.pool_sec > 0:
        n = max(1, int(a.pool_sec / S.MAX_S[t]))
        log(f"    data budget {a.pool_sec:g}s / crop {S.MAX_S[t]:g}s -> {n} utterances")
        pool = pool[:n]
    else:
        pool = pool[:a.pool_cap]
    tag = tb_log.run_tag(a.out, res["backbone"], t, a.er_fold)
    tb = tb_log.TB(a.tb_dir, tag) if a.tb_dir else tb_log.TB.off()
    tb.text("cmd", " ".join(sys.argv), 0)
    t0 = time.time()
    vec, probs, hist, n_seen = train_alpha(m, train_probe_of(t, probes, er_probes,
                                                           a.er_fold),
                                           mixer, t, pool, a, dev, tb)
    log(f"    train {time.time()-t0:.0f}s -> alloc {''.join(map(str, vec))} "
        f"(sum {sum(vec)}, avg {sum(vec)/nl:.3f}, target {a.budget:g}, project={a.project})")
    m.load_state_dict(orig)
    A.compose(m, orig, vec, mixer_LQ(mixer), dev)
    t1 = time.time()
    v = eval_full(t, m, probes, er_probes, d, dev, a.er_fold)
    avg = sum(vec) / nl
    ub, u = baseline_for(res, avg, t)
    if u is None:
        rel = f"no uniform{ub} baseline -- {ub} has to be added to --uniform-bits"
    elif u > 0:
        rel = f"vs uniform{ub} {u:.4f} ({(u - v) / u * 100:+.1f}%, {ub - avg:+.2f}bit)"
    else:
        rel = f"vs uniform{ub} {u:.4f} (n/a, baseline 0)"
    log(f"    eval {time.time()-t1:.0f}s  {METRIC[t]}={v:.4f} @avg {avg:.3f}bit  {rel}")
    # The final number is on the full SUPERB test split, so its axis differs from the training
    # curve. step is fixed to a.steps so that it is seen side by side with the end of the curve in the scalar tab.
    tb.scalar(f"eval/{METRIC[t]}", v, a.steps)
    tb.scalar("eval/avg_bits", avg, a.steps)
    if u is not None:
        tb.scalar(f"eval/uniform{ub}_{METRIC[t]}", u, a.steps)
        if u > 0:
            tb.scalar("eval/rel_gain_pct", (u - v) / u * 100, a.steps)
    tb.text("alloc", "".join(map(str, vec)), a.steps)
    tb.close()
    res["allocs"][t] = vec; res["probs"][t] = probs
    res["eval"][t] = v; res["hist"][t] = hist
    res.setdefault("cmp_bits", {})[t] = ub
    res.setdefault("avg", {})[t] = avg
    res.setdefault("n_seen", {})[t] = n_seen
    res["cfg"] = {"init": a.init, "steps": a.steps, "bs": a.bs, "seed": a.seed,
                  "pool_cap": a.pool_cap, "ent_start": a.ent_start,
                  "lam": a.lam, "beta": a.beta, "lr": a.lr, "rate": a.rate,
                  "forward": a.forward, "project": a.project,
                  "tau_start": a.tau_start, "tau_end": a.tau_end, "beta": a.beta,
                  "objective": a.objective,
                  "quant_dir": a.quant_dir, "bitset": a.bitset, "head": a.head,
                  "er_fold": a.er_fold}
    json.dump(res, open(fp, "w"), indent=1)


def mixer_LQ(mixer):
    """Restore the {(k,b): {name: tensor}} form that A.compose expects."""
    return mixer.W


if __name__ == "__main__":
    main()
