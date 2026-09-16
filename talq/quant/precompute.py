"""Precompute all quantized-backbone variants for the phase-2 sweep.

Quantization depends only on (backbone, calib, layer, bits) -- fully independent
of the probes -- so this runs while probes train. Per variant we store just the
quantized Linear weights of the touched layer(s); the sweep loads the fp32
backbone, swaps these in, attaches a frozen probe, and measures.

Variants per backbone:
  isolated : layer k in 0..11  x  bits in {2,3,4}   (rest fp32; H sees fp32 prefix)
  uniform  : ALL layers        x  bits in {2,3,4}   (sequential GPTQ, breakage floor)

Calib: SAME as phase 1 -- Emilia EN, first 48 clips, 2s crops. Resumable: a
variant whose file already exists is skipped (login-node kill insurance).
"""
import argparse
import copy
import glob
import os
import types

import torch
import torch.nn as nn

from talq.paths import CALIB_ROOT, EMILIA_EN, QUANT_GPTQ
from talq.quant.gptq import quantize_layer
from talq.eval.calib_io import load_emilia, crop, load_calib


def _eager_mhsa(self, hidden_states, attention_mask, gated_position_bias,
                output_attentions):
    """Drop-in for WavLMAttention.torch_multi_head_self_attention that CALLS
    q/k/v/out_proj modules (so GPTQ forward hooks fire) instead of handing
    their weights to F.multi_head_attention_forward. Math is identical:
    q scaled by head_dim^-0.5 after in-proj, position bias added to scores,
    key padding masked, dropout is a no-op in eval mode."""
    B, T, D = hidden_states.shape
    H = self.num_heads
    hd = D // H
    q = self.q_proj(hidden_states) * hd ** -0.5
    k = self.k_proj(hidden_states)
    v = self.v_proj(hidden_states)
    q, k, v = (t.view(B, T, H, hd).transpose(1, 2) for t in (q, k, v))
    attn = q @ k.transpose(-1, -2) + gated_position_bias.view(B, H, T, T)
    if attention_mask is not None:
        attn = attn.masked_fill(attention_mask.ne(1)[:, None, None, :],
                                torch.finfo(attn.dtype).min)
    ctx = (attn.softmax(-1) @ v).transpose(1, 2).reshape(B, T, D)
    return self.out_proj(ctx), None


def patch_wavlm_eager(model, check_x):
    """Rebind the MHA wrapper on every WavLM attention; verify equivalence."""
    from transformers.models.wavlm.modeling_wavlm import WavLMAttention
    with torch.no_grad():
        ref = model(check_x).last_hidden_state
    n = 0
    for mod in model.modules():
        if isinstance(mod, WavLMAttention):
            mod.torch_multi_head_self_attention = types.MethodType(_eager_mhsa, mod)
            n += 1
    with torch.no_grad():
        got = model(check_x).last_hidden_state
    err = (ref - got).abs().max().item()
    rel = err / max(ref.abs().max().item(), 1e-12)
    # torch's fused F.multi_head_attention_forward and this manual matmul path are
    # not bit-identical. Measured layer by layer the difference accumulates smoothly
    # with no spike, so it is not a masking/formula bug. With requires_grad_(False)
    # it drops into the fast-path kernel and the difference grows further.
    # An absolute threshold does not transfer across models, since the hidden scale
    # differs (wavlm-base 6.2 vs wavlm-large 3.6, and large additionally applies one
    # more final LN through stable_layer_norm, which amplifies the difference). So we
    # judge by a scale-independent relative error.
    # Measured: wavlm-base 1.5e-3, wavlm-large 2.3e-2.
    # Functionally verified too -- wavlmL fp32 PER was 0.0351 both before and after the patch.
    assert rel < 5e-2, (f"eager attention diverges from stock: "
                        f"max abs {err} (relative {rel:.2e})")
    print(f"patched {n} WavLM attentions to eager (max abs diff {err:.2e})", flush=True)

BACKBONES = {
    "w2v2": "facebook/wav2vec2-base",
    "wavlm": "microsoft/wavlm-base",
    # ponytail: key has no '_' after "w2v2" so glob "w2v2_*" never matches these
    "w2v2960h": "facebook/wav2vec2-base-960h",  # CTC-finetuned encoder
    "hubert": "facebook/hubert-base-ls960",  # masked-prediction SSL (3rd objective)
    "wavlmL": "microsoft/wavlm-large",       # Large axis: 24 layers, 1024 dim
    "hubertL": "facebook/hubert-large-ll60k",
}


def layer_linears(layer, prefix):
    """{full_name: weight fp32 clone} for every Linear in one encoder layer."""
    return {f"{prefix}.{n}.weight": m.weight.data.clone().cpu()
            for n, m in layer.named_modules() if isinstance(m, nn.Linear)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=list(BACKBONES), required=True)
    ap.add_argument("--emilia", default=str(EMILIA_EN))
    ap.add_argument("--calib", default=str(CALIB_ROOT / "emilia_calib.pt"))
    ap.add_argument("--bits", default="4,3,2")
    ap.add_argument("--calib-tag", default="emilia48x2s",
                    help="calib name in the saved meta. Must be changed when "
                         "building with a task calib -- otherwise it cannot be "
                         "told apart from the emilia version.")
    ap.add_argument("--out", default=str(QUANT_GPTQ))
    a = ap.parse_args()
    torch.manual_seed(0)
    os.makedirs(a.out, exist_ok=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # phase-1 calib, exactly. The Emilia tars are not on this machine, but the
    # saved .pt is bitwise identical to stack([crop(w) for w in load_emilia(
    # shards, 48)]) -- make_calib_sweep asserted that when it was written -- so
    # this preserves the original calibration values, it does not approximate
    # them. load_calib falls back to reading the tars if the .pt is absent.
    calib = load_calib(a.calib, a.emilia, dev, n=48)

    from transformers import AutoModel
    model = AutoModel.from_pretrained(BACKBONES[a.backbone],
                                      dtype=torch.float32).eval().to(dev)
    layers = model.encoder.layers                     # bare SSL model: no wrapper
    if a.backbone.startswith("wavlm"):    # base + wavlmL
        # stock WavLM feeds q/k/v/out WEIGHTS to F.multi_head_attention_forward,
        # so module hooks never fire -> swap in the equivalent eager path
        patch_wavlm_eager(model, calib[:2])
    orig = copy.deepcopy(model.state_dict())
    bits_list = [int(b) for b in a.bits.split(",")]

    for bits in bits_list:
        # isolated: one layer quantized, rest fp32
        for k in range(len(layers)):
            path = f"{a.out}/{a.backbone}_L{k}_b{bits}.pt"
            if os.path.exists(path):
                print(f"skip {path}", flush=True)
                continue
            model.load_state_dict(orig)
            quantize_layer(model, layers[k], calib, bits=bits, group=128)
            torch.save({"backbone": BACKBONES[a.backbone], "layer": k, "bits": bits,
                        "calib": a.calib_tag,
                        "weights": layer_linears(layers[k], f"encoder.layers.{k}")},
                       path)
            print(f"saved {path}", flush=True)

        # uniform: sequential over all layers
        path = f"{a.out}/{a.backbone}_ALL_b{bits}.pt"
        if os.path.exists(path):
            print(f"skip {path}", flush=True)
            continue
        model.load_state_dict(orig)
        w = {}
        for k, layer in enumerate(layers):
            quantize_layer(model, layer, calib, bits=bits, group=128)
            w.update(layer_linears(layer, f"encoder.layers.{k}"))
        torch.save({"backbone": BACKBONES[a.backbone], "layer": "ALL", "bits": bits,
                    "calib": a.calib_tag, "weights": w}, path)
        print(f"saved {path}", flush=True)

    print("DONE " + a.backbone, flush=True)


if __name__ == "__main__":
    main()
