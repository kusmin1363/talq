"""AWQ counterpart to talq.quant.precompute -- same variants, same schema, swap
GPTQ's Hessian-corrected quantize_layer for talq.quant.awq's activation-aware RTN.

Variants per backbone (identical layout to quants/, under quants_awq/):
  isolated : layer k in 0..11  x  bits in {2,3,4}   (rest fp32; sees fp32 prefix)
  uniform  : ALL layers        x  bits in {2,3,4}   (sequential AWQ, breakage floor)

Same file schema ({"backbone","layer","bits","calib","weights"}) as quants/, so
talq.eval.sweep_probes, talq.eval.grid_eval and the other consumers all
work unmodified by just pointing --quants at quants_awq.

Calib: calib/emilia_calib.pt (the raw Emilia tars this project used to build
that cache aren't present on H200), same deterministic 48x2s tensor
talq.quant.precompute's GPTQ variants were built from, so the two quant sets are
comparable apples-to-apples.

  run: python -m talq.quant.precompute_awq --backbone w2v2
"""
import argparse
import copy
import os

import torch

from talq.paths import CALIB_ROOT, EMILIA_EN, QUANT_AWQ
from talq.quant.awq import quantize_layer
from talq.eval.calib_io import load_calib
from talq.quant.precompute import BACKBONES, layer_linears, patch_wavlm_eager


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=list(BACKBONES), required=True)
    ap.add_argument("--calib", default=str(CALIB_ROOT / "emilia_calib.pt"))
    ap.add_argument("--emilia", default=str(EMILIA_EN))
    ap.add_argument("--bits", default="4,3,2")
    ap.add_argument("--calib-tag", default="emilia48x2s",
                    help="calib name in the saved meta. Must be changed when "
                         "building with a task calib -- otherwise it cannot be "
                         "told apart from the emilia version.")
    ap.add_argument("--out", default=str(QUANT_AWQ))
    ap.add_argument("--salience", default=None,
                    help="task-salience collector output; replaces AWQ's mean|x| "
                         "salience with |x . dL_task/dx|. Calib stays the same "
                         "Emilia tensor, so task is the only variable changed.")
    ap.add_argument("--clip", action="store_true",
                    help="apply llm-awq auto_clip(mse_range). It is the other half "
                         "of the original AWQ; until 2026-08-12 we had ported only "
                         "auto_scale.")
    ap.add_argument("--only-all", action="store_true",
                    help="build just the uniform {bk}_ALL_b{bits}.pt (the "
                         "deployable single-kernel model); skip isolated layers")
    a = ap.parse_args()
    torch.manual_seed(0)
    os.makedirs(a.out, exist_ok=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    calib = load_calib(a.calib, a.emilia, dev)

    from transformers import AutoModel
    model = AutoModel.from_pretrained(BACKBONES[a.backbone],
                                      dtype=torch.float32).eval().to(dev)
    layers = model.encoder.layers
    if a.backbone.startswith("wavlm"):
        patch_wavlm_eager(model, calib[:2])
    orig = copy.deepcopy(model.state_dict())
    bits_list = [int(b) for b in a.bits.split(",")]

    # salience is stored flat as {"L{k}.{linear}": vec}; quantize_layer wants it
    # keyed the way layer.named_modules() reports, i.e. layer-relative.
    sal_all, algo = None, "awq"
    if a.salience:
        d = torch.load(a.salience, map_location="cpu")
        assert d["backbone"] == a.backbone, \
            f"salience is for {d['backbone']}, not {a.backbone}"
        sal_all = [{} for _ in layers]
        for key, v in d["salience"].items():
            k, name = key[1:].split(".", 1)
            sal_all[int(k)][name] = v
        algo = f"awq_task_{d['task']}"
        print(f"task salience: {a.salience} ({d['task']}, "
              f"{sum(len(s) for s in sal_all)} Linears)", flush=True)

    def sal_of(k):
        return None if sal_all is None else sal_all[k]

    for bits in bits_list:
        for k in range(len(layers)) if not a.only_all else []:
            path = f"{a.out}/{a.backbone}_L{k}_b{bits}.pt"
            if os.path.exists(path):
                print(f"skip {path}", flush=True)
                continue
            model.load_state_dict(orig)
            quantize_layer(model, layers[k], calib, bits=bits, group=128,
                           salience=sal_of(k), clip=a.clip)
            torch.save({"backbone": BACKBONES[a.backbone], "layer": k, "bits": bits,
                        "calib": a.calib_tag, "algo": algo + ("_clip" if a.clip else ""),
                        "weights": layer_linears(layers[k], f"encoder.layers.{k}")},
                       path)
            print(f"saved {path}", flush=True)

        path = f"{a.out}/{a.backbone}_ALL_b{bits}.pt"
        if os.path.exists(path):
            print(f"skip {path}", flush=True)
            continue
        model.load_state_dict(orig)
        w = {}
        for k, layer in enumerate(layers):
            quantize_layer(model, layer, calib, bits=bits, group=128,
                           salience=sal_of(k), clip=a.clip)
            w.update(layer_linears(layer, f"encoder.layers.{k}"))
        torch.save({"backbone": BACKBONES[a.backbone], "layer": "ALL", "bits": bits,
                    "calib": a.calib_tag, "algo": algo + ("_clip" if a.clip else ""), "weights": w}, path)
        print(f"saved {path}", flush=True)

    print("DONE " + a.backbone, flush=True)


if __name__ == "__main__":
    main()
