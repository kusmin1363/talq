"""Re-measure the reference grid under **full** SUPERB official-protocol evaluation.

Grid: 5 backbone x {fp32, GPTQ b2/b3/b4, AWQ b2/b3/b4} = 35 config.
task: PR / ASR / KS / IC / ASV -- for these five the split was checked against the
      s3prl original and matches the SUPERB official one. SID (iden_split, 1,251
      speakers) and ER (5-fold) need the task heads retrained, so they are pulled
      out into a separate stage.

What differs from the existing talq.eval.arm_eval calls is that every "knob that reduced the
sample" is gone:
    eval_ctc n=32   -> all 2,608 test-clean utterances
    CAP 500         -> all 3,081 ks / 3,793 ic
    sv_stride 8     -> all 37,720 official ASV trial pairs (s3prl voxceleb1_test_v2.txt)
    no A/B split    -> training never sees the eval, so there is no reason to split

No requantization. quants/{bk}_ALL_b{b}.pt and quants_awq/ already exist for all 5
backbones (verified). The weights are loaded on and only evaluated.

  python -m talq.eval.grid_eval --backbones hubert --smoke
  python -m talq.eval.grid_eval --backbones w2v2,wavlm,hubert,hubertL,wavlmL
"""
import argparse, csv, os, time
import torch

import talq.eval.arm_eval as AE
from talq.paths import CALIB_ROOT, DATA_ROOT, PROBES, QUANT_ROOT, RESULTS_ROOT
from talq.eval.superb_data import ic_index, ks_index, load_lexicon_official
from talq.eval.probe_train import librispeech_index, CHARS

TASKS = ["pr", "asr", "ks", "ic", "asv"]
METRIC = {"pr": "PER", "asr": "WER", "ks": "KS_ERR", "ic": "IC_ERR", "asv": "EER"}
BIG = 10 ** 9


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def official_eval(data=DATA_ROOT, cap=None):
    """The full SUPERB official test split. cap is for the smoke run."""
    lex, ph = load_lexicon_official()
    LS = f"{data}/librispeech/LibriSpeech"
    d = {
        "pr": (librispeech_index(f"{LS}/test-clean", "pr", lex, phones=ph), ph, " "),
        "asr": (librispeech_index(f"{LS}/test-clean", "asr"), CHARS, ""),
        "ks": ks_index("test")[0],
        "ic": ic_index("test")[0],
    }
    trials = [(int(l), a, b) for l, a, b in
              (ln.split() for ln in open(f"{data}/voxceleb1/voxceleb1_test_v2.txt"))]
    d["asv"] = (trials, f"{data}/voxceleb1/test")
    if cap:
        d["pr"] = (d["pr"][0][:cap],) + d["pr"][1:]
        d["asr"] = (d["asr"][0][:cap],) + d["asr"][1:]
        d["ks"] = d["ks"][:cap]; d["ic"] = d["ic"][:cap]
        d["asv"] = (trials[:cap], d["asv"][1])
    return d


def setup(bk, dev):
    from transformers import AutoModel
    import talq.quant.task_fisher as T
    m = AutoModel.from_pretrained(T.BACKBONES[bk], dtype=torch.float32).eval().to(dev)
    m.requires_grad_(False)
    if bk.startswith("wavlm"):          # startswith -- with bk=="wavlm" wavlmL would silently drop out
        from talq.eval.calib_io import load_calib
        from talq.quant.precompute import patch_wavlm_eager
        patch_wavlm_eager(m, load_calib(f"{CALIB_ROOT}/emilia_calib.pt", None, dev)[:2])
    probes = AE.load_probes(T.BACKBONES[bk], PROBES, dev, tasks=TASKS)
    orig = {k: v.clone() for k, v in m.state_dict().items()}
    return m, probes, orig


def configs(bk):
    out = [("fp32", None)]
    for q, dirn in (("gptq", "quants"), ("awq", "quants_awq")):
        for b in (4, 3, 2):
            p = f"{QUANT_ROOT}/{dirn}/{bk}_ALL_b{b}.pt"
            if os.path.exists(p):
                out.append((f"{q}_b{b}", p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", default="w2v2,wavlm,hubert,hubertL,wavlmL")
    ap.add_argument("--csv", default=f"{RESULTS_ROOT}/grid_official.csv")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    dev = "cuda"
    cap = 40 if a.smoke else None
    if a.smoke:
        a.csv = a.csv.replace(".csv", "_smoke.csv")

    done = set()
    fresh = not os.path.exists(a.csv)
    if not fresh:
        for r in csv.DictReader(open(a.csv)):
            done.add((r["backbone"], r["config"]))
    fh = open(a.csv, "a", newline="")
    fields = ["backbone", "config", "quantizer", "bits"] + [METRIC[t] for t in TASKS] + ["sec"]
    w = csv.DictWriter(fh, fieldnames=fields)
    if fresh:
        w.writeheader(); fh.flush()

    d = official_eval(cap=cap)
    log(f"eval set pr {len(d['pr'][0])}  asr {len(d['asr'][0])}  ks {len(d['ks'])}  "
        f"ic {len(d['ic'])}  asv {len(d['asv'][0])} pairs")

    for bk in a.backbones.split(","):
        todo = [c for c in configs(bk) if (bk, c[0]) not in done]
        if not todo:
            log(f"[skip] {bk}"); continue
        log(f"===== {bk} ({len(todo)} config) =====")
        m, probes, orig = setup(bk, dev)
        log(f"  probe {sorted(probes)}")
        for name, path in todo:
            t0 = time.time()
            m.load_state_dict(orig)
            if path:
                sd = torch.load(path, map_location="cpu", weights_only=False)["weights"]
                m.load_state_dict({k: v.to(dev) for k, v in sd.items()}, strict=False)
            res = AE.eval_all(m, probes, d, dev, n_ctc=BIG)
            q, bits = (name.split("_b") + [""])[:2] if path else ("none", "32")
            rw = {"backbone": bk, "config": name, "quantizer": q, "bits": bits,
                  "sec": round(time.time() - t0)}
            rw.update({METRIC[t]: round(res[t], 4) for t in TASKS if t in res})
            w.writerow(rw); fh.flush()
            log(f"  {name:<9} " + "  ".join(f"{METRIC[t]}={res[t]:.4f}"
                                            for t in TASKS if t in res)
                + f"  ({rw['sec']}s)")
        del m
        torch.cuda.empty_cache()
    print("GRID_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
