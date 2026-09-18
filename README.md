# TALQ — Task-Adaptive Layer-wise Quantization for Speech Foundation Models

Reference implementation for the ICASSP submission.

A speech SSL encoder is pretrained once and reused across many downstream tasks,
but each deployed instance usually serves only one — so its weights can be
specialised without multiplying on-device storage. Existing post-training
quantization for these encoders nonetheless applies a **task-independent**
precision policy.

TALQ decouples weight quantization from bit allocation. A PTQ backend first builds
frozen 2-, 3- and 4-bit candidates for every Transformer layer; allocation then
optimises only `|B|` selection logits per layer, through Gumbel–Softmax, against a
rate–distortion objective that preserves the task head's outputs while penalising
the average bit width. The candidates are never retrained and **no downstream
labels are used**.

Because the distortion term is measured through a task head, the resulting
allocation is task-dependent — which is the paper's main empirical finding:
quantization-sensitive layers move with the task, not just with the backbone. In
WavLM they sit at layers 2–5 for speaker verification and 8–11 for phoneme
recognition.

## Results

At average-bit caps of 3.33 and 3.67, across GPTQ and AWQ, six SUPERB tasks and
five backbones, TALQ beats a speech adaptation of TAQ-KL in 17 of 24 task-level
comparisons, with the larger margins at the tighter cap. Within 5% relative error
of uniform 4-bit it needs only 3.21 and 3.38 bits on average — 1.25× and 1.18×
additional linear-weight compression.

## Reproduce the paper in one command

Everything the figures and tables need is in `results/` (9.5 MB). No GPU, no
datasets, no checkpoints.

```bash
pip install -e .
./scripts/make_figures.sh
```

Both figures come out identical to the published versions in `assets/` apart from
the creation timestamp that PDF embeds, so `md5sum` will differ while the drawing
does not. To check:

```bash
python - <<'EOF'
import re
pat = re.compile(rb"/CreationDate \(D:\d{14}Z\)")
for a, b in [("figs/fig_sens_multi_2bit.pdf", "assets/fig1_sensitivity.pdf"),
             ("figs/fig_alloc_multi_gptq_b3.33.pdf", "assets/fig2_allocation.pdf")]:
    x, y = (pat.sub(b"", open(f, "rb").read()) for f in (a, b))
    print(f"{b}: {'identical' if x == y else 'DIFFERS'}")
EOF
```

Full procedure, including re-running the sweep from scratch: [`docs/REPRODUCE.md`](docs/REPRODUCE.md).

## Layout

| Path | Contents |
|---|---|
| `talq/alloc.py` | The method: allocation logits, Gumbel–Softmax, the RD objective, full-split evaluation |
| `talq/search.py` | Composing a model from per-layer frozen candidates |
| `talq/sweep.py` | The λ sweep that traces the rate–distortion curve |
| `talq/quant/` | GPTQ and AWQ for speech encoders, and the candidate caches |
| `talq/eval/` | SUPERB task heads, official full-split evaluation, per-layer sensitivity |
| `talq/baselines/` | TAQ-KL / TAQ-IS re-implementation, scored on TALQ's own axis |
| `talq/data_prep/` | Task-conditioned calibration set construction |
| `figures/` | Regenerates every figure and table in the paper |
| `results/` | The measurements themselves |
| `docs/` | Data, checkpoints, reproduction |

GPTQ and AWQ are implemented directly against `nn.Linear` forward-pre-hooks. The
usual libraries hardcode `input_ids`, so a float-waveform speech encoder cannot go
through them at all.

## What is not here

- **Quantized candidate caches** (~790 GB). Regenerate with `talq.quant.precompute`.
- **Datasets** — all public; see [`docs/DATA.md`](docs/DATA.md).
- **Pretrained backbones** — pulled from the Hugging Face hub at run time.
- **Trained task heads** — distributed separately, ~420 MB; see [`docs/CHECKPOINTS.md`](docs/CHECKPOINTS.md).
  These are the measuring instrument, not the method: heads you train yourself will
  not reproduce the numbers exactly, because the quantization deltas are small
  relative to head-training variance.

Paths are configured entirely through environment variables (`TALQ_DATA_ROOT`,
`TALQ_CKPT_ROOT`, `TALQ_QUANT_ROOT`, `TALQ_S3PRL`); see `talq/paths.py`.

## Citation

```bibtex
TODO
```

## License

MIT (see [`LICENSE`](LICENSE)), with one exception:
[`talq/eval/superb_heads.py`](talq/eval/superb_heads.py) is derived from s3prl and
stays under Apache-2.0, as its header records. `pseudo_quantize_tensor` in
`talq/quant/awq.py` is reproduced from llm-awq (MIT).

See [`NOTICE`](NOTICE) for the short version and
[`THIRD_PARTY.md`](THIRD_PARTY.md) for what was taken from where and what changed.
