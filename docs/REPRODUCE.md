# Reproduction

Three tiers, cheapest first. Tier 1 is what most readers want and needs nothing
but this repository.

## Tier 1 — the paper's figures and tables, CPU only, minutes

Everything required ships in `results/` (9.5 MB).

```bash
pip install -e .
./scripts/make_figures.sh          # writes figs/
```

or individually:

```bash
python figures/table1.py                   # Table 1 and Table 2
python figures/table1.py --latex           # the .tex rows verbatim
python figures/fig1_sensitivity.py
python figures/fig2_allocation.py --out-dir sw_gptq --quantizer GPTQ \
       --budgets 3.3333 --backbones w2v2,hubert,wavlm
```

Expected Rel. GMean in Table 1: GPTQ 1.049 (≤3.67) and 1.077 (≤3.33); AWQ 1.141
and 1.252. Both figures come out byte-identical to `assets/`.

Note `figures/fig2_allocation.py` defaults to a different sweep directory; pass
`--out-dir sw_gptq` to get the panel that is in the paper.

## Tier 2 — re-run allocation learning

Needs one GPU, the task heads (`docs/CHECKPOINTS.md`) and the precomputed
candidates (Tier 3, ~790 GB).

```bash
export TALQ_CKPT_ROOT=... TALQ_QUANT_ROOT=... TALQ_DATA_ROOT=... TALQ_S3PRL=...

python -m talq.alloc --backbones w2v2 --tasks pr --smoke        # path check

./scripts/run_sweep.sh                     # --dry-run: cells, plan, estimate
python -m talq.sweep                       # run; resumable, keyed on artifacts
python -m talq.sweep --progress            # counts finished cells, not log lines
```

The sweep is what writes `results/sw_{gptq,awq}*/`. `--progress` deliberately
counts artifacts: a per-task OOM is caught, saves an empty result and exits 0, so
the completion log alone will overcount.

The TAQ-KL baseline is `talq.baselines.taq_grid` (allocations) evaluated through
`talq.baselines.taq_vs_talq`, which scores it against the **same frozen
candidates** on the **same protocol** as TALQ. Comparing across the two evaluators
is not valid — composing per-layer candidates and re-quantizing sequentially give
different values, because the damage is super-additive.

## Tier 3 — from scratch

```bash
# 1. per-task calibration sets (1,024 s each; ER split five ways by fold)
python -m talq.data_prep.make_task_calib --sec 1024

# 2. frozen 2/3/4-bit candidates for every layer   <-- ~790 GB, GPU-days
python -m talq.quant.precompute
python -m talq.quant.precompute_awq

# 3. task heads (or download them; see docs/CHECKPOINTS.md)
python -m talq.eval.probe_train

# 4. FP32 + uniform reference grid  -> results/grid6.csv
python -m talq.eval.grid_eval

# 5. per-layer sensitivity          -> results/sens_1024_iso.csv  (Figure 1)
python figures/measure_sensitivity.py

# 6. the sweep, then Tier 1
```

Candidates depend only on (backbone, calibration set, layer, bit width) — nothing
about the task heads — so step 2 can run while step 3 trains.

## Environment

```bash
conda env create -f environment.yml    # or: pip install -r requirements.txt
```

`figures/` needs only numpy, pandas and matplotlib. Everything else needs torch,
transformers and soundfile.
