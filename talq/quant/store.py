"""Store a fake-quantized model losslessly, so a variant can be re-scored later.

Why keep them: every arm run quantizes, evaluates, and throws the weights away.
When the PR/ASR probes arrive from the other cluster we would have to re-quantize
all ~450 variants just to fill two columns. Saving costs disk; re-quantizing costs
GPU-days.

How: fake quantization leaves each group of 128 input channels holding at most 2^b
distinct float values. So per weight tensor we store the sorted unique values plus
an index per element. Reconstruction is `table[idx]`, which is bit-exact by
construction -- no assumption about HOW the quantizer arrived at those values.

That last point is deliberate. Reconstructing (q - z) * s instead would require
reproducing GPTQ's error compensation and AWQ's clip/scale search exactly, and a
silent mismatch there is precisely the class of bug that cost us a day. Here the
save asserts bit-exactness before writing.

Only encoder Linear weights are stored -- they are the only tensors quantization
touches. Everything else is restored from the pretrained checkpoint.

  from talq.quant.store import save_quant, load_quant
  save_quant(model, "quantized/gptq/hubert_recon_b3.pt", {"variant": "recon_b3"})
  load_quant(model, "quantized/gptq/hubert_recon_b3.pt")     # model must be fp32-loaded
"""
import os

import torch
import torch.nn as nn


def _linears(model):
    for k, layer in enumerate(model.encoder.layers):
        for name, mod in layer.named_modules():
            if isinstance(mod, nn.Linear):
                yield f"L{k}.{name}", mod


def _encode(w, group=128):
    """(rows, cols) fp32 -> per-(row, group) value table + uint8 index.

    A global unique() does not work: GPTQ/AWQ scale per ROW per GROUP, so a
    768x768 3-bit tensor holds ~30k distinct values globally but only 8 per
    (row, group) block. Tables must therefore be per block.

    Vectorised via sort, so this is one pass rather than rows*groups unique() calls.
    """
    R, C = w.shape
    # The quantizer treats a whole row as one group when there are fewer columns
    # than group. WavLM's gated relative-position-bias Linear has 64 columns, which
    # is not divisible by 128. For any other non-divisible case, taking the whole
    # row as one block is likewise safe (K grows, but the 255 check below catches it).
    g = group if (C >= group and C % group == 0) else C
    G = C // g
    group = g
    x = w.reshape(R, G, group)
    s, order = x.sort(-1)
    new = torch.ones_like(s, dtype=torch.bool)
    new[..., 1:] = s[..., 1:] != s[..., :-1]
    rank = new.cumsum(-1) - 1                     # table index in sorted order
    idx = torch.empty_like(rank)
    idx.scatter_(-1, order, rank)                 # back to the original order
    K = int(new.sum(-1).max())
    if K > 255:
        raise ValueError(f"{K} unique values per block -- looks like an unquantized tensor")
    table = torch.zeros(R, G, K, dtype=torch.float32)
    # rank runs 0..K-1 within each block, so scatter collects the values
    table.scatter_(-1, rank.clamp(max=K - 1), s)
    return table, idx.to(torch.uint8), K, g          # g: block size actually used


def _decode(table, idx, shape, group=128):
    R, C = shape
    G = C // group     # take the actual group value stored at save time as is
    out = table.gather(-1, idx.long().reshape(R, G, group))
    return out.reshape(R, C)


def save_quant(model, path, meta=None, group=128, verify=True):
    """Write encoder Linear weights as per-block (value table, uint8 index)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    blob = {}
    for key, mod in _linears(model):
        w = mod.weight.data.detach().cpu().float()
        table, idx, K, g = _encode(w, group)
        blob[key] = {"table": table, "idx": idx, "shape": tuple(w.shape),
                     "K": K, "group": g}
        if verify:
            assert torch.equal(_decode(table, idx, w.shape, g), w), \
                f"{key}: reconstruction is not bit-exact"
    torch.save({"meta": meta or {}, "weights": blob}, path)
    return path


def load_quant(model, path, strict=True):
    """Restore weights saved by save_quant into an fp32-loaded model."""
    d = torch.load(path, map_location="cpu", weights_only=False)
    have = dict(_linears(model))
    missing = set(d["weights"]) - set(have)
    if strict and missing:
        raise KeyError(f"keys absent from the model: {sorted(missing)[:3]}...")
    for key, rec in d["weights"].items():
        if key not in have:
            continue
        mod = have[key]
        w = _decode(rec["table"], rec["idx"], rec["shape"], rec["group"])
        assert w.shape == mod.weight.shape, f"{key}: {w.shape} != {mod.weight.shape}"
        mod.weight.data.copy_(w.to(mod.weight.device, mod.weight.dtype))
    return d["meta"]


def stored_size(path):
    return os.path.getsize(path) / 2 ** 20        # MB


if __name__ == "__main__":                        # self-check
    torch.manual_seed(0)
    lin = nn.Linear(256, 128)
    w = lin.weight.data
    for r in range(w.shape[0]):                   # a different scale per row x group (as in practice)
        for g in range(0, w.shape[1], 128):
            blk = w[r, g:g + 128]
            lo, hi = blk.min(), blk.max()
            step = (hi - lo) / 7                  # 3 bit = 8 steps
            w[r, g:g + 128] = ((blk - lo) / step).round() * step + lo
    m = type("M", (), {})()
    m.encoder = type("E", (), {})()
    m.encoder.layers = [nn.ModuleDict({"proj": lin})]
    p = save_quant(m, "/tmp/_qs_test.pt", {"t": 1})
    before = lin.weight.data.clone()
    lin.weight.data.zero_()
    load_quant(m, p)
    assert torch.equal(lin.weight.data, before), "reconstruction mismatch"
    fp32_mb = before.numel() * 4 / 2 ** 20
    print(f"self-check passed  {torch.unique(before).numel()} global unique values  "
          f"{stored_size(p):.3f} MB / fp32 {fp32_mb:.3f} MB = "
          f"{stored_size(p) / fp32_mb:.2f}x")
