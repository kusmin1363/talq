"""Sensitivity map for the paper -- 3 backbones on one panel (backbone = colour).

Stacking three of them vertically would force the reader to cross-check "the
sensitive layers differ per task" and "that pattern is common across backbones"
by eye. Putting 3 lanes per task and separating the backbones by colour brings
both claims into **one glance** -- three colours gathered in the same column
means backbone-common, scattered means backbone-specific.

  python figures/fig1_sensitivity.py             # w2v2/hubert/wavlm, 2-bit
  python figures/fig1_sensitivity.py --bits 3
  python figures/fig1_sensitivity.py --topk 3 --no-values

The row order is the allocation depth-centroid order (ASV<KS<ER<IC<ASR<PR), so
going from top to bottom one sees a staircase in which the sensitive band moves
to the right (to deeper layers).
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import argparse, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from talq.paths import REPO_ROOT, RESULTS_ROOT

ROOT = REPO_ROOT
TASKS = ["ASV", "KS", "ER", "IC", "ASR", "PR"]
COL = {"w2v2": "#C0392B", "hubert": "#1E8449", "wavlm": "#1F4E9C"}
ALPHA = [1.0, 0.72, 0.48, 0.28]          # rank 1..4 intensity

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "axes.labelsize": 10, "xtick.labelsize": 9.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", default="w2v2,hubert,wavlm")
    ap.add_argument("--bits", type=int, default=2)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--csv", default=f"{RESULTS_ROOT}/sens_1024_iso.csv")
    ap.add_argument("--out", default=f"{ROOT}/figs")
    ap.add_argument("--name", default="",
                    help="output file name without the extension")
    ap.add_argument("--width", type=float, default=7.2)
    ap.add_argument("--values", action="store_true",
                    help="write the degradation % in the cell. The lanes get 3x thicker, so off by default")
    ap.add_argument("--lane-h", type=float, default=0,
                    help="lane height (inches). 0 = automatic, depending on --values")
    a = ap.parse_args()

    bks = [b for b in a.backbones.split(",") if b]
    c = pd.read_csv(a.csv)
    os.makedirs(a.out, exist_ok=True)
    k, nb = a.topk, len(bks)

    S = {}
    raw_iso = {"task", "delta", "fold"}.issubset(c.columns)
    for bk in bks:
        s = c[(c.backbone == bk) & (c.bits == a.bits)].copy()
        if s.empty:
            raise SystemExit(f"not found: {bk} {a.bits}-bit")
        if raw_iso:
            s["task"] = s["task"].str.upper()
            nl = int(s["layer"].max()) + 1
            S[bk] = {}
            for task in TASKS:
                # For ER, average the delta of the five folds per layer.
                v = (s[s.task == task].groupby("layer")["delta"].mean()
                     .reindex(range(nl)))
                if v.isna().any():
                    raise SystemExit(f"missing: {bk}/{task} {a.bits}-bit")
                S[bk][task] = v.to_numpy(dtype=float)
        else:
            s = s.sort_values("layer")
            S[bk] = {task: s["d_" + task].astype(float).to_numpy()
                     for task in TASKS}
            nl = len(s)
    if len({len(S[bk][task]) for bk in bks for task in TASKS}) != 1:
        raise SystemExit("the layer count does not match across backbone/task")
    nl = len(S[bks[0]][TASKS[0]])

    lane = 0.80 / nb                       # lane height inside one task cell (1.0)
    # Dropping the numbers keeps thin lanes readable, so shrink the height a lot.
    lh = a.lane_h or (0.32 if a.values else 0.155)
    body = lh * len(TASKS) * nb
    leg_h = 0.78                                  # every backbone on one horizontal row
    fig = plt.figure(figsize=(a.width, body + leg_h + 0.75))
    gs = fig.add_gridspec(2, 1, height_ratios=[body, leg_h], hspace=0.30)
    ax = fig.add_subplot(gs[0])

    for ti, t in enumerate(TASKS):
        for bi, bk in enumerate(bks):
            v = S[bk][t]
            y = ti + 0.10 + bi * lane      # backbone order from the top
            order = np.argsort(-v)[:k]
            for rk, col in enumerate(order):
                ax.add_patch(Rectangle((col - 0.5, y), 1.0, lane * 0.92,
                                       facecolor=COL[bk], alpha=ALPHA[rk],
                                       edgecolor="white", lw=0.6, zorder=2))
                if a.values:
                    ax.text(col, y + lane * 0.46, f"{v[col]:.0f}",
                            ha="center", va="center", fontsize=5.8, zorder=3,
                            color="white" if rk < 2 else "#101C1E")
        # task separator
        if ti:
            ax.axhline(ti, color="#9AA5A6", lw=0.7, zorder=1)

    # layer grid
    for x in range(nl + 1):
        ax.axvline(x - 0.5, color="#E4EAEA", lw=0.6, zorder=0)

    ax.set_xlim(-0.5, nl - 0.5)
    ax.set_ylim(len(TASKS), 0)
    ax.set_xticks(range(nl))
    ax.set_xticklabels([str(i + 1) for i in range(nl)])
    ax.set_yticks([i + 0.5 for i in range(len(TASKS))])
    ax.set_yticklabels(TASKS, fontweight="bold", fontsize=10.5)
    ax.set_xlabel("layer  (input → output)", labelpad=1)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    # Legend = one horizontal row. Laying the backbone(nb) x rank(k) blocks side
    # by side lets the two axes, colour (= backbone) and intensity (= rank), be
    # compared in one place.
    # Only the middle of the 3 rows is used, to make the bars thin (the axes
    # height is exactly the bar thickness)
    # The rank explanation was moved to the paper caption. Instead of leaving
    # that slot empty, the bars are spread evenly over the full width
    # (left/centre/right) so each backbone ramp looks large.
    gl = gs[1].subgridspec(3, nb, height_ratios=[0.55, 1.0, 1.1],
                           wspace=0.28, hspace=0)
    for bi, bk in enumerate(bks):
        cax = fig.add_subplot(gl[1, bi])
        rgb = np.array(matplotlib.colors.to_rgb(COL[bk]))
        cax.imshow(np.array([[(1 - al) + al * rgb for al in ALPHA[:k]]]),
                   aspect="auto")
        bkname = {"w2v2": "wav2vec 2.0", "hubert": "HuBERT",
                  "wavlm": "WavLM"}.get(bk, bk)
        cax.set_title(bkname, fontsize=10, fontweight="bold", pad=3)
        cax.set_yticks([])
        cax.set_xticks(range(k))
        cax.set_xticklabels([str(r + 1) for r in range(k)], fontsize=8.5)
        cax.tick_params(length=0, pad=2)
        for sp in cax.spines.values():
            sp.set_visible(False)
    sub = ("; numbers = % error increase vs. fp32" if a.values else "")
    ax.set_title(f"Top-{k} quantization-sensitive layers per task  "
                 f"({a.bits}-bit GPTQ{sub})",
                 fontsize=10.5, fontweight="bold", pad=7)
    name = (a.name or
            f"fig_sens_multi_{a.bits}bit" + ("_val" if a.values else ""))
    for ext in ("pdf", "png"):
        fig.savefig(f"{a.out}/{name}.{ext}")
    print(f"  {a.out}/{name}.pdf / .png")


if __name__ == "__main__":
    main()
