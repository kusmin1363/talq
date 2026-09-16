# Task heads

TALQ never fine-tunes the backbone. Each downstream task has a light head trained
once on the frozen FP32 backbone and then held fixed for every quantization
measurement. The heads are therefore part of the **measuring instrument**, not the
method — reproducing our numbers requires these exact heads, not equivalent ones.

```bash
export TALQ_CKPT_ROOT=/path/to/checkpoints
```

## Expected layout

```
$TALQ_CKPT_ROOT/
├── probes/          one head per (backbone, task)          ~390 MB
└── probes_superb/   the heads whose SUPERB recipe splits     ~32 MB
                     the data further: ER by session fold
```

Files are named `{hf-model-name}_{task}.pt`, e.g. `wavlm-base_pr.pt`,
`hubert-large-ll60k_er_fold3.pt`.

## Why ER is separate

`probes/` contains a single pooled ER head trained across all IEMOCAP sessions. It
**leaks the held-out session of every fold** and must not be used for ER. Every ER
number in the paper comes from `probes_superb/{backbone}_er_fold{1..5}.pt`, each
trained with its own session removed. The code enforces this: `talq/paths.py`
exposes the two directories separately and `talq.eval.sensitivity` reads ER only
from `PROBES_SUPERB`.

## One cell uses a different head

HuBERT Large on KS is scored with `hubert-large-ll60k_ks_superb_baldev.pt`. Within
that cell, FP32, uniform, TALQ and the baseline must **all** be read from that
head's outputs — mixing heads inside a cell invalidates the comparison. The shipped
`results/sw_{gptq,awq}_kshead/` carry exactly those rows.

## Training your own

`python -m talq.eval.probe_train` trains a head from scratch. It checkpoints every
1,000 steps to `<out>.pt.resume` so a killed job can continue with `--resume`;
those files are training scratch and are not needed to evaluate.

Heads you train yourself will not reproduce the paper's numbers exactly — the
quantization deltas are small relative to head-training variance, which is why the
heads are published rather than left as a training recipe.
