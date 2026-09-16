# Third-party code

This repository is MIT-licensed (see LICENSE) and incorporates code from the
projects below. Their terms apply to the portions identified here.

`talq/eval/superb_heads.py` is the one file NOT covered by the MIT LICENSE: it
stays under Apache-2.0, carries the required header, and the full licence text
is in LICENSES/Apache-2.0.txt.

## s3prl — Apache License 2.0

Copyright (c) Speech Lab, NTU, Taiwan.
https://github.com/s3prl/s3prl

`talq/eval/superb_heads.py` is a port of s3prl 0.4.18:

| Upstream | Here |
|---|---|
| `s3prl/downstream/sv_voxceleb1/model.py` — `TDNN`, `XVector`, `SP`, `UtteranceExtractor`, `AMSoftmaxLoss`, `Model` | `ASVProbe` and its components |
| `s3prl/downstream/model.py` — `UtteranceLevel`, `MeanPooling` | `UttProbe` |

Changes made: the modules were restructured into two probe classes with this
project's constructor signatures; `AMSoftmaxLoss` computes its masked logsumexp by
masking rather than by s3prl's Python loop (numerically identical, faster); the
frame-arithmetic and initialisation rationale is documented inline. Architectures
and hyperparameters (s=30.0, m=0.4, agg_dim=1500, proj=256, TDNN contexts and
dilations) are unchanged, so the heads stay numerically equivalent to s3prl's.

`talq/eval/superb_data.py` follows s3prl's split recipes but embeds no s3prl data.
It reads the CTC lexicon, the phoneme vocabulary and the ASV dev-split metadata
from a user-supplied s3prl checkout at run time (`TALQ_S3PRL`); those files are not
redistributed here.

## llm-awq — MIT License

Copyright (c) 2023 MIT HAN Lab.
https://github.com/mit-han-lab/llm-awq

`pseudo_quantize_tensor` in `talq/quant/awq.py` is reproduced verbatim from
`awq/quantize/quantizer.py`. It is copied rather than imported because llm-awq's
`awq.quantize` package unconditionally imports a compiled CUDA extension that is
not built in this environment. The clip search in the same file follows llm-awq's
`auto_clip_layer` ordering (fix the scale first, then search the clip); the
surrounding AWQ implementation is this project's own, written against speech
encoder `nn.Linear` forward-pre-hooks.

## GPTQ — not incorporated

`talq/quant/gptq.py` implements GPTQ (Frantar et al., 2022,
https://arxiv.org/abs/2210.17323) directly from the paper. No code from the
reference implementation is included; the citation is scholarly, not a licence
obligation.

## TAQ — not incorporated

`talq/baselines/taq.py` re-implements TAQ (LeVi et al., arXiv:2511.06516v4) from
the paper text. The authors published no reference implementation, so no code
could be or was copied. Every intentional divergence is recorded in that file's
DEVIATIONS block.
