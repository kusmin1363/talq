#!/usr/bin/env bash
# Regenerate every figure and table in the paper from the shipped results.
# CPU only, no datasets, no checkpoints -- everything needed is in results/.
#
# Verified: the two figures come out byte-identical to the versions in assets/.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=${1:-figs}
mkdir -p "$OUT"

echo "== Table 1 and Table 2 =="
python figures/table1.py
python figures/table1.py --latex > "$OUT/tables.tex"

echo
echo "== Figure 1: per-layer 2-bit sensitivity =="
python figures/fig1_sensitivity.py --out "$OUT"

echo
echo "== Figure 2: allocations at the 3.33-bit cap =="
# The paper's panel: GPTQ, three Base encoders, the 3.3333 cap.
python figures/fig2_allocation.py \
    --out-dir sw_gptq --quantizer GPTQ --budgets 3.3333 \
    --backbones w2v2,hubert,wavlm --out "$OUT"

echo
echo "Wrote $OUT/"
