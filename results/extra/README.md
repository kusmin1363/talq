# Additional measurements

Runs not in the paper, measured on the same 1,024-second per-task candidates
(`quants_tc1024_*`), task heads and full-split evaluation as `../sw_gptq/`.

| Path | What it is |
|---|---|
| `sensdp1024/`, `sensdp1024_b34/` | Sensitivity-ranked dynamic-programming allocation: score each layer independently, then solve the separable budget problem exactly. `figures/table1.py` carries a `sensdp()` reader for these. |
| `sw_gptq_seed1/`, `sw_gptq_seed2/` | The paper's sweep at seeds 1 and 2 (the paper is seed 0). Pair with `../seed_stability.csv`. |
| `sw_gptq_initunif/` | The same sweep with `--init uniform` instead of (0.05, 0.05, 0.90). |
| `sw34_gptq/`, `sw34_gptq_erf1..5/` | The sweep restricted to 3- and 4-bit candidates. The λ grid differs (`0.001 … 0.3`) and is not comparable cell-for-cell with the paper's (`0.001 … 5`). |
| `b34probe/`, `full_tc1024/` | A `{3,4}` probe run, and the first 1,024-second sweep, which `../sw_gptq/` superseded. |
