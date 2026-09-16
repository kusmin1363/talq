"""Direct GPTQ for wav2vec2 -- no optimum/GPTQModel.

Those libraries hardcode `input_ids` (text) and cannot take `input_values`
(float waveform), so speech encoders are unquantizable through them. The whole
"connection to the model" here is a forward-pre-hook on each nn.Linear that
captures its input activations during calibration; then per-layer closed-form
GPTQ replaces the weight. That's it -- no HF quant pipeline involved.

Run standalone: `python -m talq.quant.gptq` -> quantizes layer 0's 6 Linears and
prints per-layer output fidelity (cosine / SQNR) vs the fp32 layer. This is the
"does the first layer give the same result" check the design hinges on.
"""
import math
import torch
import torch.nn as nn


def pack_codes(q, bits):
    """(rows, cols) int codes in [0, 2^bits) -> uint8 with 8//bits codes per byte."""
    per = 8 // bits
    pad = (-q.shape[1]) % per
    if pad:
        q = torch.nn.functional.pad(q, (0, pad))
    q = q.to(torch.uint8)
    out = torch.zeros(q.shape[0], q.shape[1] // per, dtype=torch.uint8, device=q.device)
    for j in range(per):
        out |= q[:, j::per] << (j * bits)
    return out


def unpack_codes(p, bits, cols):
    """Inverse of pack_codes; crops the pad back off."""
    per = 8 // bits
    mask = (1 << bits) - 1
    out = torch.empty(p.shape[0], p.shape[1] * per, dtype=torch.uint8, device=p.device)
    for j in range(per):
        out[:, j::per] = (p >> (j * bits)) & mask
    return out[:, :cols]


def dequant_packed(st, group=128):
    """Packed state dict entry -> fp32 weight. This is what an int kernel would fuse."""
    rows, cols = st["shape"]
    q = unpack_codes(st["qweight"], st["bits"], cols).float()
    scale = st["scale"].float().repeat_interleave(group, dim=1)[:, :cols]
    zero = st["zero"].float().repeat_interleave(group, dim=1)[:, :cols]
    return (q - zero) * scale


def load_packed(path, model):
    """Restore a packed real-quant checkpoint (dequant on load).

    ponytail: real quant STORAGE only -- compute still runs fp32 after load;
    hooking an int matmul kernel is the (deliberately skipped) next step.
    """
    ckpt = torch.load(path, map_location="cpu")
    sd = {k: v.float() for k, v in ckpt["rest"].items()}
    for k, st in ckpt["packed"].items():
        sd[k] = dequant_packed(st, ckpt["group"])
    model.load_state_dict(sd)
    return model


class GPTQ:
    """Closed-form 4-bit quantizer for one nn.Linear (Frantar et al. 2022).

    Accumulate H = sum(x xᵀ) over calibration, then quantize column by column,
    propagating each column's rounding error into the not-yet-quantized columns
    via H⁻¹ -- that error feedback is the only thing that makes GPTQ better than
    round-to-nearest.
    """

    def __init__(self, layer: nn.Linear):
        self.layer = layer
        self.rows, self.cols = layer.weight.shape  # (out_features, in_features)
        self.H = None  # allocated on first add_batch, on the activation's device
        self.n = 0

    def add_batch(self, inp: torch.Tensor):
        # inp: (..., in_features). wav2vec2 hands (batch, time, d); flatten the
        # leading dims so every frame is one Hessian sample.
        inp = inp.reshape(-1, self.cols).float()
        if self.H is None:
            self.H = torch.zeros(self.cols, self.cols, device=inp.device)
        b = inp.shape[0]
        self.H *= self.n / (self.n + b)   # running mean over all calib frames
        self.n += b
        # NOT in-place: .float() is a no-op when the activation is already fp32
        # and .reshape returns a view, so `inp *= ...` used to scale the caller's
        # tensor. Registered as a forward PRE-hook, that corrupted the very
        # activation the Linear was about to consume (q/k/v_proj share one input
        # tensor, so the scaling compounded across them).
        inp = inp * math.sqrt(2 / self.n)
        self.H += inp.t() @ inp

    @torch.no_grad()
    def quantize(self, bits=4, group=128, percdamp=0.01):
        W = self.layer.weight.data.clone().float()
        H = self.H.clone()
        maxq = 2 ** bits - 1

        # A frame dimension that's always zero (dead conv channel etc.) gives a
        # zero H column -> singular. Speech feats have many; pin them and zero
        # the matching weights so they don't blow up the inverse.
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        # Damping: without it Cholesky fails on near-singular H, which is far more
        # common for speech than for text. λ = 1% of mean(diag).
        damp = percdamp * torch.diag(H).mean()
        diag = torch.arange(self.cols)
        H[diag, diag] += damp

        # Hinv (upper) = chol(inv(chol(H))). The i-th diagonal is the leverage of
        # column i; error is divided by it before being pushed forward.
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        Hinv = torch.linalg.cholesky(H, upper=True)

        Q = torch.zeros_like(W)
        Qi = torch.zeros(self.rows, self.cols, dtype=torch.uint8, device=W.device)
        scales, zeros = [], []
        scale = zero = None
        for i in range(self.cols):
            if i % group == 0:  # per-(row, group) asymmetric scale from current W
                g = W[:, i:i + group]
                xmax, xmin = g.amax(1, keepdim=True), g.amin(1, keepdim=True)
                xmax = torch.clamp(xmax, min=0)   # make sure 0 is representable
                xmin = torch.clamp(xmin, max=0)
                scale = (xmax - xmin) / maxq
                scale[scale == 0] = 1e-8
                zero = torch.round(-xmin / scale)
                scales.append(scale)
                zeros.append(zero)

            w = W[:, i:i + 1]
            q = torch.clamp(torch.round(w / scale) + zero, 0, maxq)
            dq = (q - zero) * scale       # dequantized column
            Q[:, i:i + 1] = dq
            Qi[:, i] = q[:, 0].to(torch.uint8)
            err = (w - dq) / Hinv[i, i]   # scaled residual
            # push residual into the remaining columns (GPTQ correction step)
            W[:, i + 1:] -= err @ Hinv[i:i + 1, i + 1:]

        # real-quant state: packed int codes + per-group fp16 scale / uint8 zero.
        # The fake-quant weight below stays the fp32-scale dequant (numbers unchanged);
        # the packed form is what actually goes to disk with --save-packed.
        self.qweight = pack_codes(Qi, bits)
        self.scale = torch.cat(scales, 1).half()
        self.zero = torch.cat(zeros, 1).to(torch.uint8)
        self.bits = bits

        self.layer.weight.data = Q.to(self.layer.weight.dtype)
        return Q

    def packed_state(self):
        return {"qweight": self.qweight.cpu(), "scale": self.scale.cpu(),
                "zero": self.zero.cpu(), "shape": (self.rows, self.cols),
                "bits": self.bits}


def sqnr_db(ref, test):
    return 10 * torch.log10((ref ** 2).sum() / ((ref - test) ** 2).sum()).item()


def cosine(ref, test):
    return torch.nn.functional.cosine_similarity(
        ref.flatten(), test.flatten(), dim=0).item()


def quantize_layer(model, layer, calib, bits, group, chunk=32):
    """GPTQ every Linear in one encoder layer, in place.

    Runs a full-model forward so the captured inputs already reflect the
    already-quantized earlier layers -- that's what makes it *sequential* GPTQ
    (each layer's Hessian sees the quantized prefix), not 12 independent RTNs.

    The forward is chunked over calib clips: a 384-clip batch asks for ~88GB of
    activations in one go. add_batch keeps a running mean over frames, so
    chunking accumulates the identical H = (2/n) sum(x x^T) -- and every clip is
    the same length here, so there is no padding to shift either.
    """
    linears = {n: m for n, m in layer.named_modules() if isinstance(m, nn.Linear)}
    gptq = {n: GPTQ(m) for n, m in linears.items()}
    handles = [m.register_forward_pre_hook(
        lambda _m, args, g=gptq[n]: g.add_batch(args[0]))
        for n, m in linears.items()]
    with torch.no_grad():
        for i in range(0, calib.shape[0], chunk):
            model(calib[i:i + chunk])
    for h in handles:
        h.remove()
    for n in list(gptq):
        if gptq[n].H is None:
            # Linear never fired during the calib forward (e.g. WavLM's
            # gru_rel_pos_linear on inactive paths) -> no Hessian, leave fp32.
            print(f"  [skip] {n}: no activations captured, kept fp32", flush=True)
            del gptq[n]
            continue
        # bits is a scalar or {layer-relative name: bits}. If a dict, every Linear
        # is quantized at its own bit width (faithfulness check for per-Linear
        # mixed precision).
        b = bits[n] if isinstance(bits, dict) else bits
        gptq[n].quantize(bits=b, group=group)
    return gptq   # caller may harvest packed_state(); safe to ignore


@torch.no_grad()
def encoder_repr(model, x):
    """Head-free encoder output (last_hidden_state) -- the fidelity target."""
    return model.wav2vec2(x).last_hidden_state


def load_speech(audio_dir, n, seconds=2.0, sr=16000, seed=0):
    """Load n real-speech clips -> (n, seconds*sr) tensor, normalized like wav2vec2.

    On-manifold input is the point: Gaussian noise drives every layer off the
    speech manifold, so its end-to-end error propagation is meaningless. Language
    is irrelevant here -- fidelity compares fp32 vs quantized on the SAME clip.
    """
    import glob, os, random
    import numpy as np
    import soundfile as sf

    files = sorted(glob.glob(os.path.join(audio_dir, "**/*.wav"), recursive=True))
    random.Random(seed).shuffle(files)
    L = int(seconds * sr)
    clips = []
    for f in files:
        if len(clips) >= n:
            break
        w, fsr = sf.read(f, dtype="float32")
        if w.ndim > 1:
            w = w.mean(1)
        if fsr != sr:  # linear resample: fidelity metric, not ASR, so it's plenty
            w = np.interp(np.linspace(0, len(w) - 1, int(len(w) * sr / fsr)),
                          np.arange(len(w)), w).astype("float32")
        w = np.pad(w, (0, L - len(w)))[:L] if len(w) < L else w[:L]
        w = (w - w.mean()) / np.sqrt(w.var() + 1e-7)  # wav2vec2 zero-mean unit-var
        clips.append(w)
    assert len(clips) >= n, f"only {len(clips)} clips in {audio_dir}, need {n}"
    return torch.tensor(np.stack(clips))


def main():
    import argparse, json
    from transformers import Wav2Vec2ForCTC
    from talq.paths import EMILIA_EN, RESULTS_ROOT
    torch.manual_seed(0)

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="facebook/wav2vec2-base-960h")
    p.add_argument("--out", default=str(RESULTS_ROOT / "wav2vec2-gptq4"))
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--group", type=int, default=128)
    p.add_argument("--nclips", type=int, default=48)  # ~99 frames each > 3072 -> H full rank
    p.add_argument("--audio-dir", default=str(EMILIA_EN),
                   help="real-speech calib; pass '' to fall back to synthetic noise")
    p.add_argument("--save-model", action="store_true")
    p.add_argument("--save-packed", action="store_true",
                   help="real-quant checkpoint: packed int codes on disk (no int kernel)")
    a = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = Wav2Vec2ForCTC.from_pretrained(a.model, dtype=torch.float32).eval().to(dev)

    if a.audio_dir:
        clips = load_speech(a.audio_dir, a.nclips + 4).to(dev)
        calib, heldout = clips[:a.nclips], clips[a.nclips:]  # held-out = distinct clips
    else:
        # ponytail: synthetic fallback. Off-manifold, so end-to-end numbers are
        # meaningless (only the per-layer LOCAL fidelity survives) -- diagnostic only.
        calib = torch.randn(a.nclips, 32000, device=dev)
        heldout = torch.randn(4, 32000, device=dev)

    ref = encoder_repr(model, heldout)  # fp32 reference BEFORE any quantization

    layers = model.wav2vec2.encoder.layers
    # ponytail: re-runs a full forward per layer -> O(L) forwards. Fine at L=12;
    # if it ever dominates, cache each layer's input hidden states once instead.
    packed = {}
    for i, layer in enumerate(layers):
        gptq = quantize_layer(model, layer, calib, a.bits, a.group)
        if a.save_packed:
            for n, g in gptq.items():
                packed[f"wav2vec2.encoder.layers.{i}.{n}.weight"] = g.packed_state()
        # per-layer fidelity of the WHOLE encoder as we deepen the quantized prefix
        q = encoder_repr(model, heldout)
        print(f"[layer {i:2d} done]  encoder cosine = {cosine(ref, q):.5f}"
              f"   SQNR = {sqnr_db(ref, q):.2f} dB", flush=True)

    q = encoder_repr(model, heldout)
    n_linears = sum(isinstance(m, nn.Linear)
                    for _, m in model.wav2vec2.encoder.named_modules())
    res = {
        "model": a.model, "bits": a.bits, "group": a.group, "nclips": a.nclips,
        "n_linears": n_linears,
        "final_cosine": cosine(ref, q), "final_sqnr_db": sqnr_db(ref, q),
    }
    print(json.dumps(res, indent=2), flush=True)
    # Report, don't crash: the number IS the experiment. Bar is a soft sanity flag.
    print("PASS" if res["final_cosine"] > 0.95 else "WARN low fidelity", flush=True)

    import os
    os.makedirs(a.out, exist_ok=True)
    with open(f"{a.out}/fidelity.json", "w") as f:
        json.dump(res, f, indent=2)
    if a.save_model:
        # fake-quant weights (dequantized fp32 on the 4-bit grid): a full-size
        # checkpoint, not a packed one. Fine for a fidelity study, useless for
        # deployment -- add real int4 packing only if on-device size is measured.
        model.save_pretrained(a.out)

    if a.save_packed:
        # REAL quant on disk: encoder Linears as packed int codes + fp16 group
        # scale / uint8 zero; everything else (conv frontend, layernorms, biases,
        # CTC head) fp16. Loadable via load_packed() -- dequant on load, so
        # inference still runs fp32 (int matmul kernel deliberately out of scope).
        ckpt = {"bits": a.bits, "group": a.group, "packed": packed,
                "rest": {k: v.half() for k, v in model.state_dict().items()
                         if k not in packed}}
        path = f"{a.out}/packed_int{a.bits}.pt"
        torch.save(ckpt, path)
        mb = os.path.getsize(path) / 1e6
        enc_mb = sum(t.numel() * t.element_size()
                     for st in packed.values() for t in
                     (st["qweight"], st["scale"], st["zero"])) / 1e6
        print(f"packed checkpoint -> {path}", flush=True)
        print(f"  on-disk {mb:.1f} MB total | encoder quantized part {enc_mb:.1f} MB "
              f"(fp32 would be {sum(st['shape'][0]*st['shape'][1] for st in packed.values())*4/1e6:.1f} MB)",
              flush=True)
        # roundtrip check: pack -> save -> load -> dequant must reproduce the
        # fake-quant weights (up to fp16 scale rounding, rel ~5e-4)
        sd = model.state_dict()
        worst = 0.0
        for k, st in packed.items():
            dq = dequant_packed(st, a.group).to(sd[k].device)
            worst = max(worst, ((dq - sd[k]).norm() / sd[k].norm()).item())
        assert worst < 1e-3, f"pack/dequant roundtrip broken: rel err {worst:.2e}"
        print(f"PACK ROUNDTRIP OK (max rel err {worst:.2e})", flush=True)
    print("OK: full encoder quantized", flush=True)


if __name__ == "__main__":
    main()
