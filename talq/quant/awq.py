"""Direct AWQ for wav2vec2/wavlm/hubert -- no llm-awq entry pipeline.

llm-awq's own `awq.quantize` package can't even be imported here: qmodule.py
unconditionally imports the compiled `awq_inference_engine` CUDA extension
(the real W4A16 GEMM kernel), which isn't built in this env, and TinyChat sits
on top of that same kernel for LLM/VLM decoder serving (Llama/Qwen/VILA/...) --
neither applies to a non-causal speech encoder. Same situation as GPTQ: no
library path, so this hand-rolls the algorithm directly via forward-pre-hooks,
reusing only the one architecture-agnostic piece of llm-awq worth reusing
(`pseudo_quantize_tensor`, reproduced verbatim below since importing it drags
in the same broken package chain).

AWQ's core idea (Lin et al. 2023): weight-only round-to-nearest (no GPTQ-style
Hessian error feedback), but first equalize each INPUT channel's magnitude by
a per-channel scale s searched to minimize local output reconstruction error,
then absorb the scale into the weight before quantizing:
    W' = quantize(W * s) / s
which is exactly what you'd get by fusing 1/s into the preceding op (as the
original repo does, via LayerNorm/Linear surgery hardcoded per decoder
architecture) and s into this Linear's weight -- except computed and undone
entirely LOCALLY, needing no knowledge of the neighboring module. That's the
one algebraic fact that makes this portable across wav2vec2 (post-norm),
WavLM (post-norm + gated relpos bias), and their large pre-norm variants
without writing per-architecture LN-fusion cases: identical math, one op.

Run standalone: `python -m talq.quant.awq` -> quantizes layer 0's 6 Linears and
prints fidelity vs fp32, same smoke test talq.quant.gptq runs.
"""
import torch
import torch.nn as nn


def pseudo_quantize_tensor(w, n_bit=4, zero_point=True, q_group_size=-1):
    """Group-wise asymmetric round-to-nearest. Verbatim from llm-awq's
    awq/quantize/quantizer.py (pseudo_quantize_tensor), MIT-licensed, copied
    rather than imported because importing the package pulls in the unbuilt
    CUDA extension (see module docstring)."""
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    assert w.dim() == 2
    max_val = w.amax(dim=1, keepdim=True)
    min_val = w.amin(dim=1, keepdim=True)
    max_int = 2 ** n_bit - 1
    scales = (max_val - min_val).clamp(min=1e-5) / max_int
    zeros = (-torch.round(min_val / scales)).clamp_(0, max_int)
    w = (torch.clamp(torch.round(w / scales) + zeros, 0, max_int) - zeros) * scales
    return w.reshape(org_w_shape)


@torch.no_grad()
def search_clip(W_eff, x_eff, bits, group, n_grid=20, max_shrink=0.5, n_sample=512,
                w_tok=None, w_alpha=0.25):
    """Finds the optimal clip range for each (output channel x group). A port of
    llm-awq's auto_clip_layer.

    Until now our implementation had ported only half of AWQ (auto_scale), and the
    quantization range was exactly the min/max pseudo_quantize_tensor uses. The
    original puts a clip search on top of that: cutting off both tails of the
    weight makes the quantization grid finer, so most values become more accurate
    (at the cost of the few tail values becoming less accurate).

    Follows llm-awq's order exactly: fix the scale first, then search the clip. The
    objective is, as in the original, the reconstruction error of the **per-group
    partial sum** (not of the full Linear output).

    W_eff = W*s (the scale-applied weight), x_eff = x/s (the matching input).
    Their elementwise product equals x*W, so this is equivalent to measuring the
    error against the final output.
    """
    co, ci = W_eff.shape
    gs = group if ci % group == 0 else ci
    ng = ci // gs
    xf = x_eff.reshape(-1, ci)
    # raw w_t is extremely skewed -- of 98k tokens only 40~164 are effective, and a
    # uniform subsample (stride 192) leaves **0** of the top 200 tokens. That is,
    # used uncompressed the weighting is effectively not applied at all. With w^0.25
    # the effective tokens recover to 15k~60k.
    # (same prescription as the one learned on output channel g, and the same logic
    # as AWQ using mean|x|^ratio)
    wt = None if w_tok is None else w_tok.to(xf.device).float().clamp(min=0).pow(w_alpha)
    if wt is not None:
        assert wt.numel() == xf.shape[0], \
            f"frame-weight length {wt.numel()} != token count {xf.shape[0]} (chunk split mismatch)"
    if xf.shape[0] > n_sample:                       # token subsample (memory)
        st = max(1, xf.shape[0] // n_sample)
        xf = xf[::st][:n_sample]
        if wt is not None:
            wt = wt[::st][:n_sample]                 # subsample the weights the same way
    if wt is not None:
        wt = (wt / wt.sum().clamp(min=1e-12)).view(1, -1, 1)   # normalize to sum=1
    xf = xf.reshape(1, -1, ng, gs)                   # [1, T, ng, gs]
    W = W_eff.reshape(co, 1, ng, gs)

    ob = 64 if co % 64 == 0 else co                  # output-channel batch (OOM guard)
    best_all = []
    for i in range(0, co, ob):
        w = W[i:i + ob]
        org_max = w.abs().amax(dim=-1, keepdim=True)          # [ob,1,ng,1]
        best = org_max.clone()
        err_min = torch.full_like(org_max, float("inf"))
        org_out = (xf * w).sum(-1)                            # [ob,T,ng]
        for j in range(int(max_shrink * n_grid)):
            mv = org_max * (1 - j / n_grid)
            qw = pseudo_quantize_tensor(torch.clamp(w, -mv, mv),
                                        n_bit=bits, q_group_size=gs)
            d2 = (xf * qw).sum(-1).sub_(org_out).pow_(2)      # [ob, T, ng]
            err = (d2.mean(1) if wt is None else (d2 * wt).sum(1)).view_as(err_min)
            better = err < err_min
            err_min[better] = err[better]
            best[better] = mv[better]
            del qw, err
        best_all.append(best)
    return torch.cat(best_all, 0).view(co, ng, 1).expand(co, ng, gs).reshape(co, ci)


@torch.no_grad()
def awq_search_layer(m: nn.Linear, x: torch.Tensor, bits: int, group: int, n_grid: int = 20,
                     sal: torch.Tensor = None, return_scale: bool = False,
                     out_w: torch.Tensor = None):
    """One Linear: search the per-input-channel scale that minimizes output
    reconstruction MSE after W*s is quantized and unscaled. x: (N, in_features)
    captured calib activations. Returns the final quantized weight (fp32,
    same shape as m.weight) -- this IS the deployable fake-quant weight, no
    scale buffer needed at inference (unlike the LN-fused deployment).

    sal: optional (in_features,) salience vector REPLACING AWQ's stock
    `mean|x|`. AWQ defines salience purely from the forward pass, which carries
    no information about what a downstream task does with the output -- fine for
    an LLM (one task) but undefined for an SSL encoder (task unknown at quant
    time). The AWQ task-salience script supplies a task-gradient salience here; the
    reconstruction objective and grid below are untouched, so the two conditions
    differ in exactly one vector.

    out_w: optional (out_features,) weight on the reconstruction objective --
    THIS is where task information belongs. AWQ shares one scale vector `s`
    across all output rows and scores it by an UNWEIGHTED mean over output
    channels, so re-weighting those channels genuinely changes which grid point
    wins. (GPTQ has no such lever: its rows are independent given H, so a
    per-row weight is exactly a no-op -- scaling a quadratic cannot move its
    argmin. That asymmetry is why task-salience failed on both but this may not.)

    The right weight is the SECOND-order quantity a_i = E[(dL/dy_i)^2], the
    output-space diagonal Fisher. The earlier task-salience arm used a
    first-order quantity E|x . dL/dx|, which is the same order-of-term mistake
    that killed gradient-based bit allocation."""
    W = m.weight.data.float()
    x = x.float()
    x_scale = x.abs().mean(0) if sal is None else sal.to(x.device).float()
    y_orig = x @ W.t()
    # e.g. WavLM's gru_rel_pos_linear (head_dim=64 -> 8): in_features doesn't
    # divide the usual 128 group; fall back to one group spanning the whole
    # row rather than asserting (GPTQ's own grouping degrades the same way
    # for a short row -- W[:, 0:group] just clips to all of it).
    eff_group = group if W.shape[1] % group == 0 else W.shape[1]

    best_loss, best_Wq, best_s = float("inf"), None, None
    for i in range(n_grid):
        ratio = i / n_grid
        s = x_scale.pow(ratio).clamp(min=1e-4)
        s = s / (s.max() * s.min()).sqrt()          # keep overall scale ~unit
        Wq = pseudo_quantize_tensor(W * s, n_bit=bits, q_group_size=eff_group) / s
        d2 = (y_orig - x @ Wq.t()).pow(2)
        # out_w is a per-output-channel weight. A positive constant factor cannot
        # change the argmin, so its scale is irrelevant, but the caller normalizes it
        # to mean=1 to keep the loss values across the grid easy to read.
        loss = (d2.mean(0) * out_w).sum().item() if out_w is not None else d2.mean().item()
        if loss < best_loss:
            best_loss, best_Wq, best_s = loss, Wq, s
    return (best_Wq, best_s) if return_scale else best_Wq


def quantize_layer(model, layer, calib, bits, group, n_grid=20, salience=None,
                   clip=False, out_w=None):
    """AWQ every Linear in one encoder layer, in place. Same signature as
    talq.quant.gptq.quantize_layer so talq.quant.precompute_awq is a drop-in swap.

    Runs a full-model forward so captured activations reflect whatever the
    caller already did to earlier layers (sequential composition for the
    uniform "ALL" variant, fp32 prefix for isolated per-layer files) --
    identical calling convention to the GPTQ pipeline.

    salience: optional {linear_name: (in_features,)} overriding stock mean|x|
    per Linear (see awq_search_layer). Names are layer-relative, e.g.
    "attention.q_proj", matching layer.named_modules()."""
    linears = {n: m for n, m in layer.named_modules() if isinstance(m, nn.Linear)}
    acts = {n: [] for n in linears}
    handles = [m.register_forward_pre_hook(
        lambda _m, args, n=n: acts[n].append(args[0].detach().reshape(-1, args[0].shape[-1])))
        for n, m in linears.items()]
    with torch.no_grad():
        model(calib)
    for h in handles:
        h.remove()
    for n, m in linears.items():
        if not acts[n]:
            print(f"  [skip] {n}: no activations captured, kept fp32", flush=True)
            continue
        x = torch.cat(acts[n], dim=0)
        use_clip = clip and not any(t in n for t in ("q_proj", "k_proj"))
        Wq, s = awq_search_layer(m, x, bits, group, n_grid,
                                 sal=None if salience is None else salience.get(n),
                                 return_scale=True,
                                 out_w=None if out_w is None else out_w.get(n))
        if use_clip:
            # exactly llm-awq's auto_clip order: fix the scale -> search the clip on
            # top of it -> clamp -> requantize -> /s. The original excludes q/k_proj
            # because of the qk bmm.
            eg = group if m.weight.shape[1] % group == 0 else m.weight.shape[1]
            W_eff, x_eff = m.weight.data.float() * s, x.float() / s
            mv = search_clip(W_eff, x_eff, bits, eg)
            Wq = pseudo_quantize_tensor(torch.clamp(W_eff, -mv, mv),
                                        n_bit=bits, q_group_size=eg) / s
        m.weight.data = Wq.to(m.weight.dtype)
    return linears


def _selfcheck():
    torch.manual_seed(0)
    m = nn.Linear(256, 256)
    x = torch.randn(64, 256)
    x[:, :32] *= 20  # a salient block of channels, like real activation stats

    # ratio=0 (uniform RTN, no AWQ) vs best-of-grid must not be worse
    W = m.weight.data.float()
    y = x @ W.t()
    plain = pseudo_quantize_tensor(W, n_bit=3, q_group_size=64)
    loss_plain = (y - x @ plain.t()).pow(2).mean().item()
    Wq = awq_search_layer(m, x, bits=3, group=64, n_grid=20)
    loss_awq = (y - x @ Wq.t()).pow(2).mean().item()
    assert loss_awq <= loss_plain + 1e-9, f"AWQ search worse than plain RTN: {loss_awq} > {loss_plain}"
    print(f"selfcheck OK: plain-RTN MSE {loss_plain:.5f} -> AWQ-search MSE {loss_awq:.5f} "
          f"({'better' if loss_awq < loss_plain else 'tied'})")

    class Wrap(nn.Module):
        def __init__(self, lin):
            super().__init__()
            self.lin = lin

        def forward(self, x):
            return (self.lin(x),)

    wrap = Wrap(m)
    calib = torch.randn(4, 8, 256)
    calib[..., :32] *= 20
    quantize_layer(wrap, wrap, calib, bits=4, group=64)
    assert wrap.lin.weight.shape == (256, 256)
    print("selfcheck OK: quantize_layer runs end-to-end on a toy module")


if __name__ == "__main__":
    import argparse
    from talq.paths import EMILIA_EN
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="facebook/wav2vec2-base-960h")
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--group", type=int, default=128)
    p.add_argument("--nclips", type=int, default=48)
    p.add_argument("--audio-dir", default=str(EMILIA_EN))
    a = p.parse_args()

    _selfcheck()

    from transformers import Wav2Vec2ForCTC
    from talq.quant.gptq import load_speech, encoder_repr, cosine, sqnr_db

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = Wav2Vec2ForCTC.from_pretrained(a.model, dtype=torch.float32).eval().to(dev)
    clips = load_speech(a.audio_dir, a.nclips + 4).to(dev)
    calib, heldout = clips[:a.nclips], clips[a.nclips:]
    ref = encoder_repr(model, heldout)

    for i, layer in enumerate(model.wav2vec2.encoder.layers):
        quantize_layer(model, layer, calib, a.bits, a.group)
        q = encoder_repr(model, heldout)
        print(f"[layer {i:2d} done]  encoder cosine = {cosine(ref, q):.5f}"
              f"   SQNR = {sqnr_db(ref, q):.2f} dB", flush=True)

    q = encoder_repr(model, heldout)
    print(f"final cosine = {cosine(ref, q):.5f}  SQNR = {sqnr_db(ref, q):.2f} dB")
    print("PASS" if cosine(ref, q) > 0.95 else "WARN low fidelity")
