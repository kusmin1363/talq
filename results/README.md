# Shipped results

Raw measurements, small enough to version (9.5 MB). Every figure and table in the
paper rebuilds from these alone — no GPU, no dataset, no checkpoints:

```bash
python figures/table1.py          # Table 1 and Table 2
python figures/fig1_sensitivity.py
python figures/fig2_allocation.py
```

## Provenance

Every allocation cell records the candidate directory it was measured against in
its own `quant_dir` field. All of them read `quants_tc1024_{task}` — the 1,024-second
per-task calibration candidates the paper describes. Nothing here was measured
against an earlier or shorter calibration set.

| Path | What it is |
|---|---|
| `sw_gptq/`, `sw_awq/` | TALQ λ sweep, one directory per λ per backbone. `{backbone}/l{λ}/{backbone}_b3.json` holds the learned allocation, its achieved average bit width, and the full-split SUPERB metric for every task. `{backbone}/base/` additionally carries the uniform 2/3/4-bit reference for that backbone. |
| `sw_{gptq,awq}_erf1..5/` | The same sweep for ER, run once per IEMOCAP session fold. ER is never read from the pooled directories: the pooled head leaks each fold's held-out session. |
| `sw_{gptq,awq}_kshead/` | KS for HuBERT Large only, measured with the task head that cell uses. FP32, uniform, TALQ and the baseline must all be read from here for that cell — mixing heads within a cell invalidates the comparison. |
| `taq3600_b34*/` | TAQ-KL baseline allocations, evaluated against the same frozen candidates as TALQ so the two are comparable. Same `_erf*` / `_kshead` split convention. `scores.csv` carries the per-layer importance the allocation was derived from — see *Auditing the baseline* below. |
| `sens_1024_iso.csv` | Figure 1. One row per (backbone, task, layer, fold): that layer alone quantized to 2 bits, everything else FP32. `sens_1024_iso_{backbone}.csv` are the per-backbone shards it was merged from — kept because they were written by separate processes. |
| `grid6.csv` | FP32 and uniform GPTQ/AWQ 2/3/4-bit reference grid, 5 backbones × 7 configurations. |
| `fp32_ours.json` | FP32 ER baselines measured under the fold protocol. |
| `ks_head_hubertL.json` | The HuBERT Large KS head's own reference rows. |
| `seed_stability.csv` | Allocation stability across random seeds. |
| `uniform_bank/` | The uniform 2/3/4-bit reference measured once per backbone and reused across λ cells. |
| `pool3600/` | The 3,600-second audio pool that allocation is decided on, as ordered file lists — one per task, one per ER fold. Not stored anywhere else: the pool is re-derived from the corpora each run, so these lists are the only way to read it without running the code. See `pool3600/README.md`. |
| `extra/` | Measurements on the same axis that are **not** in the paper — a third allocation method, seed and initialisation sensitivity, the `{3,4}` bit set. See `extra/README.md`. |

## Regenerating these

See `docs/REPRODUCE.md`. Rebuilding the sweep needs the task heads and the
precomputed candidates; the candidates alone are roughly 790 GB and are not
distributed.

## Auditing the baseline

The paper claims TALQ beats TAQ-KL, so the baseline's own allocation has to be
checkable rather than taken on trust. Each `taq3600_b34*/scores.csv` holds the
gradient-derived per-layer importance for that evaluation group — one row per
(backbone, task, method), with the score vector, the `topk` fraction, and the
`hi_layers` it selects.

Two things re-derive from those numbers alone:

- `hi_layers` is the top `round(n_layers * topk)` layers by score, ties going to the
  lower index (`talq.baselines.taq.allocate_topk`). Checked for all 102 rows.
- each `taq_spec_{backbone}.csv` bit sequence is the top *n* by the same ordering,
  where *n* is fixed by the budget rather than by `topk` — at 3.3333 over `{3,4}`
  with 12 layers that is 4 layers at 4 bits, and at 3.6667 it is 8. Checked for all
  194 (row, budget) pairs.

So a reader can go from the scores to the allocation to the evaluated number
without re-running anything on a GPU.

`method` is `taq-is` (input-statistics importance, independent of the task head),
`taq-kl` (KL through the head — the baseline the paper reports), or `taq-kl-cos`, an
exploratory cosine variant. The five `taq-kl-cos` rows have no `taq_spec` rows: no
budget spec was ever built for them, so they were never evaluated.

One field needed repair. `taq_grid` wrote its `calib` provenance through
`os.path.basename`, which is right when the value is a path but truncates the
descriptive tag that `--calib-pool` produces: `train_pool(er/fold1, 3600s, n=600)`
was stored as `fold1, 3600s, n=600)`. The ER rows here carry the full string
restored from the fold number and count they still held; the code no longer
truncates it.
