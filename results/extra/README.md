# Additional measurements

Runs that are **not** in the paper but were measured on the same axis as it: the
same 1,024-second per-task candidates (`quants_tc1024_*`), the same task heads and
the same full-split evaluation. Every cell here records its `quant_dir`, and all
894 files check out — so these numbers can be put in a table next to `../sw_gptq/`
without an apples-to-oranges problem.

They exist to answer questions the paper does not.

## What is here

### `sensdp1024/`, `sensdp1024_b34/` — a third allocation method

Sensitivity-ranked dynamic-programming allocation: score each layer's damage
independently, then solve the separable budget problem exactly. Neither TALQ's
joint optimisation nor TAQ-KL's two-level split.

This is the control for the question *"is the gain from task-awareness, or from
optimising jointly instead of ranking?"* — both TALQ and TAQ-KL are task-aware, so
the paper alone cannot separate the two. `figures/table1.py` already carries a
`sensdp()` reader for these files (`--table` does not call it; wire it in if you
want the extra rows).

`_b34` is the same over the `{3,4}` bit set.

### `sw_gptq_seed1/`, `sw_gptq_seed2/` — seed sensitivity

The paper's sweep at seeds 1 and 2; the paper itself is seed 0. Allocation is a
stochastic search over a discrete space, so "layer 7 got 4 bits" is only meaningful
if it survives a reseed. Pair these with `../seed_stability.csv`.

### `sw_gptq_initunif/` — does the initialisation matter?

Identical to the paper's sweep except `--init uniform` (1/3, 1/3, 1/3) instead of
(0.05, 0.05, 0.90).

The paper starts near 4 bits and descends under the budget penalty. A uniform start
sits at an interior point of the simplex where the effective weight is a convex
combination of the three candidates, so independent rounding errors partly cancel —
that point measures *better* than any deployable one-hot allocation while not being
deployable at all. This run is what that costs.

### `sw34_gptq/`, `sw34_gptq_erf1..5/` — the `{3,4}` bit set

The same sweep restricted to 3- and 4-bit candidates, dropping 2-bit entirely.

Note the λ grid differs (`0.001 … 0.3`, denser at the low end) and is not
comparable cell-for-cell with the paper's (`0.001 … 5`): with 2-bit removed the
floor is 3.00 bits, so everything at λ ≥ 0.3 collapses onto it and a wider grid
would only add duplicates.

**Not paper material.** The paper makes no `{3,4}` claim; this sweep is here only
so the bit set can be varied.

### `b34probe/`, `full_tc1024/`

A `{3,4}` probe run, and the first 1,024-second sweep, which `../sw_gptq/`
superseded. Kept for provenance; prefer `../sw_gptq/` for anything current.

## What is deliberately absent

An earlier round of work — cross-task allocation transfer, objective/forward-mode
ablations, upper and lower bounds, and a MAPSSWE paired significance test for
ASR/PR — was measured against the **96-second** candidates, before the 1,024-second
set existed. Those numbers are internally consistent but cannot be placed beside
the paper's, so they are not shipped. Re-running them on this axis is the honest
fix if any of them is needed.
