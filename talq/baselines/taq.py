"""TAQ (Task-Aware Quantization) reproduction -- speech encoder port.

The original paper: LeVi et al., "You Had One Job: Per-Task Quantization Using LLMs' Hidden
Representations", ICML 2026 Workshop on AdaptFM (arXiv:2511.06516v4).
See arXiv:2511.06516v4 for the full paper text.

**There is no public implementation.** The last sentence of the abstract is "A reference
implementation is available at ." -- it was published with the URL blank and there is no link in
the appendix either. So this file is a self-reimplementation written from the paper text alone
(§4.2/§4.3/§4.4, Appendix E.2). Every point where we intentionally diverge from the original paper
is written down in DEVIATIONS below.

Three scoring rules (the quantization operator is identical in all three; only the score and the
allocation policy differ):
  TAQ-IS  entropy of the eigenvalues of the d x d hidden state covariance + activation variance
          (sign inverted), label-free
  TAQ-KL  KL of the output distribution after per-layer quantization noise injection, label-free
  TAQ-O   the actual task metric degradation when a single layer alone is lowered to a low bit
          width, uses labels (diagnostic)

DEVIATIONS (what the speech encoder port changed relative to the original paper -- to be stated in
the paper):
  D1 Bit set. The original paper uses {4,8} for TAQ-IS/KL and {4,16} for TAQ-O. We default to
     {2,3,4} to match the existing assets (quants/{bk}_L{k}_b{b}.pt, b in 2,3,4).
     sub-4bit is a regime the original paper does not explore. --hi/--lo can also set the
     original values.
  D2 The output distribution of TAQ-KL. The original paper uses the autoregressive next-token
     distribution. A speech encoder is non-AR and has no such distribution, so we substitute the
     posterior of the frozen SUPERB task head that is part of the deployed system. The task head
     is already trained and no ground truth enters the score computation, so the label-free
     property is preserved.
     - classification (er/ks/ic/sid): KL over the class softmax
     - CTC (pr/asr): per-frame softmax KL averaged over the valid frames
     - asv: there is no posterior (embedding task). Replaced by embedding cosine distance, and
       since that is not a KL it is recorded separately in the results as method="taq-kl-cos".
  D3 delta_ell. The original paper says "the per-example range of the last token hidden state".
     Speech has no corresponding position, so we define it as the per-example range over all
     valid frames.
  D4 valid token -> the valid frames after conv subsampling (per the attention mask).
  D5 layer indexing. The score has to be taken at the *output* of block ell, so we use
     hidden_states[ell+1] (hidden_states[0] is before entering the encoder, so it has no
     corresponding quantization target block).

  python -m talq.baselines.taq --selfcheck       # pure math verification, no GPU needed
  python -m talq.baselines.taq --backbone w2v2 --score is
  python -m talq.baselines.taq --backbone w2v2 --score kl --task er
  python -m talq.baselines.taq --backbone w2v2 --score oracle --task er
"""
import argparse
import csv
import json
import math
import os
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from talq.paths import CALIB_ROOT, EMILIA_EN, PROBES, RESULTS_ROOT

# Defaults from the original paper (§5.1 "TAQ implementation details", Appendix C)
R_RESERVOIR = 256      # number of TAQ-IS reservoir token vectors / layer
ALPHA, BETA = 0.5, 0.5  # TAQ-IS combination weights
TOPK = 0.25            # top K% -> high precision
ORACLE_K = 8           # number of top-k layers TAQ-O keeps at high precision
ORACLE_EDGE = 2        # number of first/last blocks TAQ-O unconditionally keeps at high precision
N_SENS = 16            # number of held-out examples for TAQ-O sensitivity estimation
GROUP = 128            # group-wise affine quantization group size
EPS = 1e-12


# ---------------------------------------------------------- shared utilities

def zscore(vals):
    """z-normalization along the layer axis. z(.) in Eq.7 of the original paper is the layer axis, not a global scalar axis."""
    t = torch.as_tensor(vals, dtype=torch.float64)
    sd = t.std(unbiased=False)
    if sd < EPS:                      # identical across all layers -> no information, set to 0
        return torch.zeros_like(t)
    return (t - t.mean()) / sd


def frame_mask(model, wav_lens, n_frames):
    """(B, T_frames) bool. If wav_lens is None everything is valid (no padding)."""
    if wav_lens is None:
        return None
    flens = model._get_feat_extract_output_lengths(torch.as_tensor(wav_lens))
    ar = torch.arange(n_frames)
    return ar[None, :] < torch.as_tensor(flens)[:, None]


def _select(h, mask):
    """(B, T, D) -> (N, D), valid frames only."""
    return h.reshape(-1, h.shape[-1]) if mask is None else h[mask]


# ---------------------------------------------------------------- TAQ-IS

def matrix_entropy(R, exact=False):
    """Eq.4-5 of the original paper. R: (r, d) token representations -> entropy of the eigenvalues of the centered covariance.

    When r < d, C is rank-deficient so most eigenvalues are 0 and p log(p+eps) -> 0. The original
    paper also uses combinations such as r=256, d=3584, so this situation is normal.

    Implementation: instead of forming C = Z^T Z / r and calling eigvalsh, we use the singular
    values of Z. The non-zero eigenvalues of C are exactly svdvals(Z)^2 / r, and the remaining
    d - min(r,d) are 0 and contribute 0 to the p log(p + eps) term, so **the entropy value is
    mathematically identical**. (With exact=True it takes the original d x d eigvalsh path, and
    selfcheck verifies that the two paths agree.)

    Why this matters: one 768x768 double eigvalsh took 21.8 s on this shared machine. It is called
    per layer, so 4 minutes for a single base model, and large (1024d) is worse. The 256x768 SVD
    path produces the same value in tens of ms.

    One more performance trap: on CPU this decomposition is extremely sensitive to LAPACK thread
    contention. The same input took 9825ms multi-threaded and 29.6ms single-threaded, a **330x**
    difference (shared machine). So score_is runs on GPU by default (warm 14ms). If it has to run
    on CPU, putting torch.set_num_threads(1) first is overwhelmingly faster.

    Note: the spectral entropy in §3 of the original paper is based on the token-token Gram
    (m x m), whereas the Info_ell that TAQ-IS actually uses here is based on the §4.2 covariance
    (d x d). The paper calls both definitions 'matrix entropy' without distinction, but the one
    that enters the algorithm is the latter.
    """
    R = R.double()
    Z = R - R.mean(0, keepdim=True)
    if exact:
        lam = torch.linalg.eigvalsh((Z.T @ Z) / R.shape[0]).clamp_min(0)
    else:
        lam = torch.linalg.svdvals(Z).pow(2) / R.shape[0]
    s = lam.sum()
    if s < EPS:
        return 0.0
    p = lam / s
    return float(-(p * (p + EPS).log()).sum())


def score_is(hs_per_layer, var_per_layer, alpha=ALPHA, beta=BETA, device=None):
    """Eq.5-7 of the original paper. hs_per_layer[ell] = (r, d) reservoir, var_per_layer[ell] = scalar variance.

    Stab_ell = -Var_ell. Miss the sign and you protect exactly the opposite layers.
    The reservoir is gathered on CPU to save memory, but the decomposition runs on device
    (the reason is in the matrix_entropy docstring).
    """
    info = [matrix_entropy(R if device is None else R.to(device))
            for R in hs_per_layer]
    stab = [-v for v in var_per_layer]                    # Eq.6
    zi, zs = zscore(info), zscore(stab)
    comb = alpha * zi + beta * zs                         # Eq.7
    return {"score": comb.tolist(), "info": info, "stab": stab,
            "z_info": zi.tolist(), "z_stab": zs.tolist()}


@torch.no_grad()
def collect_is_stats(model, calib, wav_lens=None, r=R_RESERVOIR, batch=8, seed=0):
    """Runs calib through and collects per-layer (reservoir, activation variance). block ell -> hidden_states[ell+1]."""
    dev = next(model.parameters()).device
    L = len(model.encoder.layers)
    pools = [[] for _ in range(L)]
    ssum = [0.0] * L
    ssq = [0.0] * L
    cnt = [0] * L
    for i in range(0, len(calib), batch):
        x = calib[i:i + batch].to(dev)
        lens = None if wav_lens is None else wav_lens[i:i + batch]
        hs = model(x, output_hidden_states=True).hidden_states
        m = frame_mask(model, lens, hs[0].shape[1])
        if m is not None:
            m = m.to(dev)
        for k in range(L):
            v = _select(hs[k + 1].float(), m)             # D5: the output of block k
            pools[k].append(v.cpu())
            # the variance is over all valid frames x hidden dim, not over the reservoir (Eq.6)
            ssum[k] += float(v.sum())
            ssq[k] += float((v * v).sum())
            cnt[k] += v.numel()
    g = torch.Generator().manual_seed(seed)
    res, var = [], []
    for k in range(L):
        P = torch.cat(pools[k], 0)
        if r and P.shape[0] > r:                          # reservoir sampling
            P = P[torch.randperm(P.shape[0], generator=g)[:r]]
        res.append(P)
        mu = ssum[k] / cnt[k]
        var.append(ssq[k] / cnt[k] - mu * mu)             # E[h^2] - E[h]^2
    return res, var


# ---------------------------------------------------------------- TAQ-KL

def kl_logits(p_logits, q_logits, T=1.0, mask=None):
    """KL(p || q). Takes both (B, C) classification and (B, T, C) CTC.

    CTC averages the per-frame KL over the valid frames (D2).
    """
    logp = F.log_softmax(p_logits.double() / T, dim=-1)
    logq = F.log_softmax(q_logits.double() / T, dim=-1)
    kl = (logp.exp() * (logp - logq)).sum(-1)             # (B,) or (B, T)
    if kl.dim() == 2:
        if mask is None:
            kl = kl.mean(-1)
        else:
            m = mask.to(kl.device).double()
            kl = (kl * m).sum(-1) / m.sum(-1).clamp_min(1)
    return kl


@torch.no_grad()
def delta_scales(model, calib, wav_lens=None, target_bit=2, batch=8):
    """delta_ell = r_ell / (2^b - 1) from §4.4 of the original paper.

    r_ell is 'the per-example range of the last token hidden state' in the original paper, but a
    speech encoder has no such position, so we take the per-example range over all valid frames (D3).
    """
    dev = next(model.parameters()).device
    L = len(model.encoder.layers)
    rng = [[] for _ in range(L)]
    for i in range(0, len(calib), batch):
        x = calib[i:i + batch].to(dev)
        lens = None if wav_lens is None else wav_lens[i:i + batch]
        hs = model(x, output_hidden_states=True).hidden_states
        m = frame_mask(model, lens, hs[0].shape[1])
        for k in range(L):
            h = hs[k + 1].float()
            for b in range(h.shape[0]):
                v = h[b] if m is None else h[b][m[b].to(dev)]
                rng[k].append(float(v.max() - v.min()))
    return [sum(r) / len(r) / (2 ** target_bit - 1) for r in rng]


class _NoiseHook:
    """Adds U(-delta/2, delta/2) to the encoder block output (Eq.9-10 of the original paper).

    Why a hook: the injection has to happen during the forward pass for the noise to propagate to
    downstream layers. Adding it only to the hidden_states stack would perturb just that one term
    of the weighted-sum head, which is a different quantity from what the original paper measures.
    """

    def __init__(self, delta, gen):
        self.delta, self.gen = delta, gen

    def __call__(self, mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        n = torch.empty_like(h).uniform_(-self.delta / 2, self.delta / 2,
                                         generator=self.gen)
        h = h + n
        return (h,) + out[1:] if isinstance(out, tuple) else h


@torch.no_grad()
def score_kl(model, probe, calib, wav_lens=None, target_bit=2, T=1.0,
             batch=8, seed=0, embed_mode=False):
    """Eq.11 of the original paper. Injects noise layer by layer and measures the output distribution change as a KL.

    embed_mode=True is the path for asv, which has no posterior: it uses the cosine distance of
    probe.embed(). Since it is not a KL, the call site has to record a distinct method name (D2).
    """
    dev = next(model.parameters()).device
    L = len(model.encoder.layers)
    deltas = delta_scales(model, calib, wav_lens, target_bit, batch)
    gen = torch.Generator(device=dev).manual_seed(seed)

    def outputs():
        outs, masks = [], []
        for i in range(0, len(calib), batch):
            x = calib[i:i + batch].to(dev)
            lens = None if wav_lens is None else wav_lens[i:i + batch]
            hs = torch.stack(model(x, output_hidden_states=True).hidden_states)
            m = frame_mask(model, lens, hs.shape[2])
            outs.append(probe.embed(hs) if embed_mode else probe(hs))
            masks.append(m)
        return outs, masks

    base, masks = outputs()
    scores = []
    for k in range(L):
        h = model.encoder.layers[k].register_forward_hook(_NoiseHook(deltas[k], gen))
        try:
            pert, _ = outputs()
        finally:
            h.remove()
        vals = []
        for p, q, m in zip(base, pert, masks):
            if embed_mode:
                vals.append(1 - F.cosine_similarity(p.double(), q.double(), dim=-1))
            else:
                vals.append(kl_logits(p, q, T, m))
        scores.append(float(torch.cat(vals).mean()))
    return {"score": scores, "delta": deltas}


# ---------------------------------------------------------------- TAQ-O

def score_oracle_from_csv(csv_path, backbone, task, bits):
    """Aggregates Delta_ell of Eq.8 of the original paper from an already measured sensitivity map.

    results/sensitivity_superb.csv (produced by the per-layer sensitivity sweep) holds
    (backbone, layer, task, bits, metric, value), which is the task metric when a single layer
    alone is taken to a low bit width in isolation, i.e. exactly the definition of TAQ-O.
    There is no need to rerun joint GPTQ.

    All 7 metrics in talq.eval.arm_eval.METRIC are error-type (PER/WER/*_ERR/EER), so larger is worse.
    The degradation is therefore max(0, M(f^(ell)) - S_base), which is the same quantity as the
    original paper's max(0, S_base - M) with only the sign flipped.
    """
    per = defaultdict(dict)
    for r in csv.DictReader(open(csv_path)):
        if r["backbone"] == backbone and r["task"] == task:
            per[int(r["bits"])][int(r["layer"])] = float(r["value"])
    base = per.get(32)
    if not base or bits not in per:
        raise SystemExit(f"{csv_path}: no ({backbone},{task},bits={bits}) or fp32 baseline row")
    layers = sorted(k for k in per[bits] if k in base)
    return {"layers": layers,
            "score": [max(0.0, per[bits][k] - base[k]) for k in layers]}


# ------------------------------------------------------------ allocation policy

def allocate_topk(scores, K=TOPK, hi=4, lo=2):
    """The original paper's TAQ-IS/TAQ-KL allocation: top K% by score -> hi bit, the rest -> lo bit.

    The size of the top group is rounded. Fig.5-7 of the original paper is the evidence: 28 layers
    -> 7 at 8bit (28*.25=7), 42 layers -> 11 (42*.25=10.5 -> 11). Flooring would give 10, which
    does not match. Ties promote the smaller layer index first, which makes it deterministic.

    The rounding uses math.floor(x + 0.5). Python's built-in round() is banker's rounding, so
    round(10.5) == 10 and the 42-layer case diverges from the original paper (selfcheck catches it).
    """
    L = len(scores)
    n = math.floor(L * K + 0.5)
    order = sorted(range(L), key=lambda k: (-scores[k], k))
    top = set(order[:n])
    return {k: (hi if k in top else lo) for k in range(L)}


def allocate_oracle(scores, k=ORACLE_K, edge=ORACLE_EDGE, hi=4, lo=2, frac=None):
    """The original paper's TAQ-O allocation: unconditionally keep the first/last `edge` blocks at
    hi, and on top of that keep the top `k` by Delta at hi. Everything else is lo.

    Even if an edge layer is already in the top-k, no additional k layers are drawn -- the original
    paper only says "keep a small set of edge layers ... and additionally keep the top-k layers",
    so the union has size at most 2*edge + k, and less when they overlap.

    **Depth saturation (a problem that came out of the speech encoder port, D6).** The original
    paper's k=8, edge=2 are absolute values that presume a 28-42 layer LLM, giving high precision
    fractions of 12/28=43% and 12/42=29% respectively. Used as is on a 12-layer base encoder,
    2*2+8 = 12 = every layer becomes high precision and the allocation disappears entirely
    (measured: 10 of 12 layers high precision on w2v2/er). So when frac is given, the total number
    of high precision layers is set to round(frac*L) and only the slots edge leaves over are
    assigned to top-k. With frac=None the original paper's absolute values are used as they are.
    """
    L = len(scores)
    keep = set(range(min(edge, L))) | set(range(max(0, L - edge), L))
    if frac is None:
        # literally as in the original paper: top-k over all layers. Overlap with edge shrinks the union by that much.
        cand = sorted(range(L), key=lambda i: (-scores[i], i))
    else:
        # frac mode has to actually hit the target count, so it draws from candidates with edge removed.
        # (Drawing from all layers lets top-k pick a layer that is already an edge layer and the budget evaporates --
        #  selfcheck caught that at 12 layers/frac=0.5 the target of 6 came out as only 4.)
        k = max(0, math.floor(frac * L + 0.5) - len(keep))
        cand = sorted((i for i in range(L) if i not in keep),
                      key=lambda i: (-scores[i], i))
    keep |= set(cand[:k])
    return {i: (hi if i in keep else lo) for i in range(L)}


# ------------------------------------------------------------- memory accounting

def _encoder_linears(model):
    """{layer_idx: [(name, weight_shape)]} -- the only tensors quantization touches."""
    out = defaultdict(list)
    for k, layer in enumerate(model.encoder.layers):
        for n, m in layer.named_modules():
            if isinstance(m, nn.Linear):
                out[k].append((f"encoder.layers.{k}.{n}.weight", tuple(m.weight.shape)))
    return out


def linear_bytes(shape, bits, group=GROUP):
    """Accounting of Appendix E.2 of the original paper. Quantized weight + FP16 group scale +
    per-group zero-point at that bit width. A layer left at 16bit stays the FP16 original."""
    R, C = shape
    if bits >= 16:
        return R * C * 2
    # same rule as talq.quant.store._encode: if the columns are not divisible by group, a whole row is one block
    g = group if (C >= group and C % group == 0) else C
    ngroup = R * (C // g)
    return (R * C * bits / 8) + (ngroup * 2) + (ngroup * bits / 8)


def weight_footprint(model, bit_map, group=GROUP):
    """Realized weight footprint W (GB) + average bits per linear layer.

    Everything that is not quantized (conv feature extractor, layer norm, pos conv embed, etc.) is
    counted as FP16 -- the same treatment as the original paper counting embedding/lm_head/norm as
    FP16. The average bits are computed from linear weights only, as in the original paper.
    """
    lins = _encoder_linears(model)
    quant_names = {n for v in lins.values() for n, _ in v}
    qbytes = wbits = wnum = 0.0
    for k, entries in lins.items():
        b = bit_map[k]
        for _, shape in entries:
            qbytes += linear_bytes(shape, b, group)
            wbits += shape[0] * shape[1] * b
            wnum += shape[0] * shape[1]
    other = sum(p.numel() for n, p in model.named_parameters() if n not in quant_names)
    total = qbytes + other * 2
    return {"W_GB": total / 2 ** 30,
            "quant_linear_GB": qbytes / 2 ** 30,
            "other_fp16_GB": other * 2 / 2 ** 30,
            "avg_bits_linear": wbits / wnum,
            "n_linear_params": int(wnum)}


# ---------------------------------------------------------------- selfcheck

def _selfcheck():
    """Pure math verification that runs without a GPU. Same role as --selfcheck in the allocator."""
    ok = []

    # 1. matrix_entropy: an isotropic Gaussian (uniform eigenvalues) approaches the maximum
    #    entropy log(d), rank-1 approaches 0.
    g = torch.Generator().manual_seed(0)
    d = 16
    iso = torch.randn(4096, d, generator=g)
    e_iso = matrix_entropy(iso)
    v = torch.randn(d, generator=g)
    rank1 = torch.randn(4096, 1, generator=g) @ v[None, :]
    e_r1 = matrix_entropy(rank1)
    assert e_r1 < 0.05 < e_iso < math.log(d) + 1e-9, (e_r1, e_iso, math.log(d))
    assert abs(e_iso - math.log(d)) < 0.05, e_iso
    ok.append(f"matrix_entropy  rank1={e_r1:.4f}  iso={e_iso:.4f} (max log d={math.log(d):.4f})")

    # 1b. whether the SVD path and the d x d eigvalsh path give the same value (rank-deficient included).
    for shape in ((256, 768), (64, 768), (512, 32)):
        M = torch.randn(*shape, generator=g)
        a_, b_ = matrix_entropy(M), matrix_entropy(M, exact=True)
        assert abs(a_ - b_) < 1e-8, (shape, a_, b_)
    ok.append("matrix_entropy  SVD path == eigvalsh path (256x768/64x768/512x32, error<1e-8)")

    # 2. Stab sign: a layer with smaller variance has to get a higher score (Eq.6).
    R = [iso[:64] for _ in range(3)]
    s = score_is(R, [10.0, 1.0, 5.0], alpha=0.0, beta=1.0)["score"]
    assert s[1] > s[2] > s[0], s
    ok.append(f"stab sign       var(10,1,5) -> score {[round(x,3) for x in s]}  (smaller variance, higher score)")

    # 3. zscore: mean 0, standard deviation 1, constant input gives 0.
    z = zscore([1.0, 2.0, 3.0, 4.0])
    assert abs(float(z.mean())) < 1e-12 and abs(float(z.std(unbiased=False)) - 1) < 1e-12
    assert float(zscore([7.0, 7.0, 7.0]).abs().max()) == 0.0
    ok.append("zscore          mean=0 sd=1, constant input -> 0")

    # 4. allocate_topk rounding: reproduces 28 layers->7 and 42 layers->11 of Fig.5-7 of the original paper.
    for L, want in ((28, 7), (42, 11), (12, 3), (24, 6)):
        bm = allocate_topk(list(range(L)), K=0.25, hi=4, lo=2)
        got = sum(1 for b in bm.values() if b == 4)
        assert got == want, (L, got, want)
    ok.append("allocate_topk   28 layers->7, 42->11, 12->3, 24->6 (matches Fig.5-7 of the original paper)")

    # 5. whether allocate_topk really picks the top scores + tie determinism.
    bm = allocate_topk([0.0, 9.0, 9.0, 0.0], K=0.5, hi=4, lo=2)
    assert [bm[i] for i in range(4)] == [2, 4, 4, 2], bm
    bm = allocate_topk([1.0, 1.0, 1.0, 1.0], K=0.5, hi=4, lo=2)
    assert [bm[i] for i in range(4)] == [4, 4, 2, 2], bm
    ok.append("allocate_topk   top selection correct, ties prefer the lower index (deterministic)")

    # 6. allocate_oracle: edge kept + union with top-k.
    bm = allocate_oracle([0.0] * 10, k=2, edge=2, hi=16, lo=4)
    hi = sorted(i for i, b in bm.items() if b == 16)
    assert hi == [0, 1, 8, 9], hi          # the top-2 tie is index 0,1 -> overlaps with edge
    bm = allocate_oracle([0, 0, 0, 0, 5, 6, 0, 0, 0, 0], k=2, edge=2, hi=16, lo=4)
    assert sorted(i for i, b in bm.items() if b == 16) == [0, 1, 4, 5, 8, 9]
    ok.append("allocate_oracle edge 2+2 fixed + union with top-k")

    # 6b. depth saturation (D6): using the original paper's absolute values on 12 layers collapses
    #     to every layer high precision. Given frac, the specified fraction is respected.
    sc12 = [float(i) for i in range(12)]
    sat = sum(1 for b in allocate_oracle(sc12, k=8, edge=2, hi=16, lo=4).values() if b == 16)
    assert sat >= 10, sat        # 10/12 = 83%, the same as the measured w2v2/er
    for f, want in ((0.25, 3), (0.5, 6), (1 / 3, 4)):
        bm = allocate_oracle(sc12, edge=2, hi=16, lo=4, frac=f)
        got = sum(1 for b in bm.values() if b == 16)
        assert got == max(want, 4), (f, got, want)   # edge 2+2=4 is the lower bound
    bm = allocate_oracle([float(i) for i in range(24)], edge=2, hi=16, lo=4, frac=0.25)
    assert sum(1 for b in bm.values() if b == 16) == 6
    ok.append("allocate_oracle 12-layer saturation reproduced + --oracle-frac keeps the fraction (edge 2+2 is the lower bound)")

    # 7. linear_bytes: 4bit is 1/4 of fp16 + metadata, 16bit is exactly 2 bytes/param.
    assert linear_bytes((768, 768), 16) == 768 * 768 * 2
    b4 = linear_bytes((768, 768), 4)
    raw = 768 * 768 * 0.5
    meta = 768 * 6 * 2 + 768 * 6 * 0.5           # scale fp16 + zero 4bit, group 128
    assert abs(b4 - (raw + meta)) < 1e-6, b4
    assert linear_bytes((768, 64), 4) == 768 * 64 * 0.5 + 768 * 2 + 768 * 0.5  # C<group
    ok.append(f"linear_bytes    768x768 4bit={b4/2**20:.4f}MiB ({b4/(768*768*2):.3f}x of fp16)")

    # 8. kl_logits: 0 for identical distributions, grows as the distributions move apart, the mask removes padding.
    p = torch.randn(4, 7, generator=g)
    assert float(kl_logits(p, p).abs().max()) < 1e-12
    far = kl_logits(p, p * 5).mean()
    near = kl_logits(p, p * 1.1).mean()
    assert far > near > 0, (far, near)
    seq = torch.randn(2, 5, 7, generator=g)
    m = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    full = kl_logits(seq, seq * 2)
    masked = kl_logits(seq, seq * 2, mask=m)
    assert abs(float(masked[1] - full[1])) < 1e-12          # identical when everything is valid
    assert abs(float(masked[0] - full[0])) > 1e-9           # differs because padding is excluded
    ok.append(f"kl_logits       identical=0, far {far:.3f} > near {near:.3f}, mask applied")

    # 9. score_oracle_from_csv: an improvement (negative degradation) is clipped to 0.
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="") as f:
        w = csv.writer(f)
        w.writerow(["backbone", "layer", "task", "bits", "metric", "value"])
        for k, (v32, v2) in enumerate([(0.10, 0.30), (0.10, 0.09), (0.10, 0.15)]):
            w.writerow(["w2v2", k, "er", 32, "ER_ERR", v32])
            w.writerow(["w2v2", k, "er", 2, "ER_ERR", v2])
        path = f.name
    r = score_oracle_from_csv(path, "w2v2", "er", 2)
    os.unlink(path)
    assert r["layers"] == [0, 1, 2]
    assert abs(r["score"][0] - 0.20) < 1e-9 and r["score"][1] == 0.0
    assert abs(r["score"][2] - 0.05) < 1e-9
    ok.append("oracle_from_csv degradation aggregation correct, improved layers clipped to 0")

    print("\n".join("  ok  " + s for s in ok))
    print(f"\nselfcheck passed ({len(ok)} items)")


# ---------------------------------------------------------------- CLI

def load_backbone(bk, dev, calib):
    from transformers import AutoModel
    from talq.quant.precompute import BACKBONES, patch_wavlm_eager
    # The dtype= keyword has a different name depending on the transformers version (dtype/torch_dtype).
    # fp32 is the default anyway, so we do not pass it as an argument and pin it with .float() -- the
    # other scripts in this repo use dtype=, but on the currently installed
    # transformers that dies with a TypeError.
    m = AutoModel.from_pretrained(BACKBONES[bk]).float().eval().to(dev)
    m.requires_grad_(False)
    if bk.startswith("wavlm"):
        # quants/ was produced on the eager MHA path. Without startswith, wavlmL is
        # silently left unpatched (a real bug found in the earlier sensitivity sweep).
        patch_wavlm_eager(m, calib[:2])
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--backbone")
    ap.add_argument("--score", choices=["is", "kl", "oracle"])
    ap.add_argument("--task", help="required for kl/oracle")
    ap.add_argument("--hi", type=int, default=4, help="high precision bit width (the original paper: IS/KL 8, O 16)")
    ap.add_argument("--lo", type=int, default=2, help="low precision bit width (the original paper: 4)")
    ap.add_argument("--topk", type=float, default=TOPK)
    ap.add_argument("--oracle-k", type=int, default=ORACLE_K)
    ap.add_argument("--oracle-edge", type=int, default=ORACLE_EDGE)
    ap.add_argument("--oracle-frac", type=float, default=None,
                    help="fraction of TAQ-O high precision layers. If given, used instead of --oracle-k. "
                         "The original paper's absolute values (k=8,edge=2) assume a 28-42 layer LLM, so on a "
                         "12-layer encoder they saturate to every layer high precision (see allocate_oracle).")
    ap.add_argument("--reservoir", type=int, default=R_RESERVOIR)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--beta", type=float, default=BETA)
    ap.add_argument("--target-bit", type=int, default=None,
                    help="reference bit width for the TAQ-KL noise magnitude (default: --lo)")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--n-calib", type=int, default=None, help="number to cut from the front of calib")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calib", default=f"{CALIB_ROOT}/emilia_calib.pt")
    ap.add_argument("--emilia", default=f"{EMILIA_EN}")
    ap.add_argument("--probes", default=f"{PROBES}")
    ap.add_argument("--sens-csv", default=f"{RESULTS_ROOT}/sensitivity_superb.csv")
    ap.add_argument("--out", default=f"{RESULTS_ROOT}/taq")
    a = ap.parse_args()

    if a.selfcheck:
        return _selfcheck()
    if not a.backbone or not a.score:
        raise SystemExit("--backbone and --score are required (or --selfcheck)")
    if a.score in ("kl", "oracle") and not a.task:
        raise SystemExit(f"--score {a.score} requires --task")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)

    if a.score == "oracle":
        # reads only the already measured sensitivity map. The model is loaded for the footprint computation.
        r = score_oracle_from_csv(a.sens_csv, a.backbone, a.task, a.lo)
        scores = r["score"]
        bit_map = allocate_oracle(scores, a.oracle_k, a.oracle_edge, a.hi, a.lo,
                                  frac=a.oracle_frac)
        extra = {"layers": r["layers"]}
        method = "taq-o"
    else:
        from talq.eval.calib_io import load_calib
        calib = load_calib(a.calib, a.emilia, dev, n=48)
        if a.n_calib:
            calib = calib[:a.n_calib]
        model = load_backbone(a.backbone, dev, calib)
        if a.score == "is":
            res, var = collect_is_stats(model, calib, r=a.reservoir,
                                        batch=a.batch, seed=a.seed)
            r = score_is(res, var, a.alpha, a.beta, device=dev)
            scores, extra, method = r["score"], r, "taq-is"
        else:
            import talq.eval.arm_eval as arm_eval
            from talq.quant.precompute import BACKBONES
            tag = BACKBONES[a.backbone].split("/")[-1]
            fp = f"{a.probes}/{tag}_{a.task}.pt"
            if not os.path.exists(fp):
                raise SystemExit(f"no task head: {fp}")
            probe = arm_eval.build_probe(a.task, torch.load(fp, map_location=dev), dev)
            r = score_kl(model, probe, calib, target_bit=a.target_bit or a.lo,
                         T=a.temp, batch=a.batch, seed=a.seed,
                         embed_mode=(a.task == "asv"))
            scores, extra = r["score"], r
            method = "taq-kl-cos" if a.task == "asv" else "taq-kl"
        bit_map = allocate_topk(scores, a.topk, a.hi, a.lo)

    if a.score == "oracle":
        from talq.eval.calib_io import load_calib
        calib = load_calib(a.calib, a.emilia, dev, n=2)
        model = load_backbone(a.backbone, dev, calib)
    fp_stats = weight_footprint(model, bit_map)
    n_hi = sum(1 for v in bit_map.values() if v == a.hi)
    if n_hi >= 0.9 * len(bit_map):
        print(f"  [warning] {n_hi} of {len(bit_map)} layers are high precision -- the allocation is effectively saturated. "
              f"For TAQ-O, use --oracle-frac.")

    out = {"method": method, "backbone": a.backbone, "task": a.task,
           "hi": a.hi, "lo": a.lo, "topk": a.topk,
           "scores": scores, "bit_map": {str(k): v for k, v in sorted(bit_map.items())},
           "footprint": fp_stats, "detail": extra,
           "args": {k: v for k, v in vars(a).items() if k != "selfcheck"}}
    name = f"{a.backbone}_{method}_{a.task or 'none'}_b{a.lo}{a.hi}.json"
    path = os.path.join(a.out, name)
    json.dump(out, open(path, "w"), indent=1, ensure_ascii=False)
    hi_layers = [k for k, v in sorted(bit_map.items()) if v == a.hi]
    print(f"{method} {a.backbone}/{a.task}: {a.hi}bit layers = {hi_layers}")
    print(f"  avg bits/linear = {fp_stats['avg_bits_linear']:.3f}   "
          f"W = {fp_stats['W_GB']:.3f} GB")
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
