"""Learned bit-sequence map -- 3 backbones on one panel (backbone = colour, intensity = bits).

It is the **same layout** as figures/fig1_sensitivity.py. Where the sensitivity map shows
"which layer is vulnerable at low bit width", this figure shows "where the
training actually spent the bits". Putting the two side by side contrasts the
motivation (the sensitive layers differ per task) and the result (so the
allocation is learned differently too) in the same coordinate system.

2-bit is left blank. Painting only 3/4-bit makes "the places that used bits"
immediately readable, and visually matches the sensitivity map painting only
the top-k.

The operating point supports two protocols. ``cap`` picks the cell with the
**highest bit width among avg_bit <= budget**, and ``nearest`` picks the cell
closest to the target mean bit width without looking at downstream performance.
The latter is the operating-point selection protocol of Table 1.

ER applies the same lambda to the five-fold results. After picking the common
lambda closest to the target mean bit width, the figure shows the fold-wise
median bit of each layer.

The row order is the allocation depth-centroid order (ASV<KS<ER<IC<ASR<PR). If
the painted band moves to the right (to deeper layers) as one goes from top to
bottom, that is exactly the task-dependence.

Giving several budgets stretches the panels horizontally. The larger the
budget, the more dark cells, but **the positions stay the same per task** at a
glance.

  python figures/fig2_allocation.py                  # GPTQ, 3.0/3.25/3.5 caps
  python figures/fig2_allocation.py --budgets 3.0 --out-dir full_awq
  python figures/fig2_allocation.py --out-dir sw_gptq --selection nearest \
      --budgets 3.333,3.583 --out ../paper/mine/images
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import argparse, glob, json, os, re, statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from talq.paths import REPO_ROOT, RESULTS_ROOT

ROOT = REPO_ROOT
TASKS = ["asv", "ks", "er", "ic", "asr", "pr"]
LABEL = {"asv": "ASV", "ks": "KS", "er": "ER", "ic": "IC",
         "asr": "ASR", "pr": "PR"}
COL = {"w2v2": "#C0392B", "hubert": "#1E8449", "wavlm": "#1F4E9C",
       "wavlmL": "#7D3C98", "hubertL": "#B7791F"}
BITS = [4, 3, 2]
ALPHA = {4: 1.0, 3: 0.45, 2: 0.0}        # 2-bit is left empty

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "axes.labelsize": 10, "xtick.labelsize": 9.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def points(od, bk, task):
    """List of (avg_bit, lambda, allocation). Performance values are not read."""
    out = []
    for fp in glob.glob(f"{RESULTS_ROOT}/{od}/{bk}/l*/{bk}_b3.json"):
        m = re.search(r"/l([0-9.]+)/", fp)
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        if not m or task not in d.get("allocs", {}):
            continue
        v = d["allocs"][task]
        ab = sum(v) / len(v)
        out.append((ab, float(m.group(1)), v))
    return out


def snap(budget, nl):
    """Snap the nominal budget onto the layer grid.

    3.33 or 3.3333 is a truncated decimal of 10/3 = 3.3333333..., so comparing
    with `avg <= budget` under the cap rule **drops the allocation that sits
    exactly on the budget** and falls one cell below. On 2026-09-12 all 120 rows
    of the table came out short that way.
    """
    return round(budget * nl) / nl


def pick(od, bk, task, budget, selection):
    """Pick the allocation of the target operating point without using performance."""
    pts = points(od, bk, task)
    if selection == "nearest":
        return min(pts, key=lambda p: (abs(p[0] - budget), p[1])) if pts else None
    nl = len(pts[0][2]) if pts else 12
    pts = [p for p in pts if p[0] <= snap(budget, nl) + 1e-9]
    return max(pts, key=lambda p: (p[0], -p[1])) if pts else None


def pick_er(od, bk, budget, selection, folds):
    """Pick ER's common lambda by bit width and return the layer-wise median."""
    by_fold = [points(f"{od}_erf{fold}", bk, "er")
               for fold in range(1, folds + 1)]
    maps = [{lam: (ab, vec) for ab, lam, vec in fold_pts}
            for fold_pts in by_fold]
    common = set.intersection(*(set(m) for m in maps)) if maps else set()
    candidates = []
    for lam in common:
        av = statistics.mean(m[lam][0] for m in maps)
        vecs = [m[lam][1] for m in maps]
        median_vec = [int(statistics.median(v[i] for v in vecs))
                      for i in range(len(vecs[0]))]
        candidates.append((av, lam, median_vec))
    if selection == "nearest":
        return (min(candidates, key=lambda p: (abs(p[0] - budget), p[1]))
                if candidates else None)
    nl = len(candidates[0][2]) if candidates else 12
    candidates = [p for p in candidates if p[0] <= snap(budget, nl) + 1e-9]
    return max(candidates, key=lambda p: (p[0], -p[1])) if candidates else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", default="w2v2,hubert,wavlm")
    ap.add_argument("--out-dir", default="full_tc")
    ap.add_argument("--budgets", default="3.0,3.25,3.5")
    ap.add_argument("--selection", choices=["cap", "nearest"], default="cap")
    ap.add_argument("--er-folds", type=int, default=5)
    ap.add_argument("--quantizer", choices=["GPTQ", "AWQ"], default=None)
    ap.add_argument("--out", default=f"{ROOT}/figs")
    ap.add_argument("--width", type=float, default=0)
    ap.add_argument("--lane-h", type=float, default=0.155)
    ap.add_argument("--xlabel", action="store_true",
                    help="add the x-axis label. It overlaps the legend when the figure is small.")
    a = ap.parse_args()
    buds = [float(x) for x in a.budgets.split(",") if x]

    bks = [b for b in a.backbones.split(",") if b]
    nb, nB = len(bks), len(buds)
    qn = a.quantizer or ("AWQ" if "awq" in a.out_dir.lower() else "GPTQ")
    has_er_folds = os.path.isdir(f"{RESULTS_ROOT}/{a.out_dir}_erf1")
    A = {(bd, bk, t):
         (pick_er(a.out_dir, bk, bd, a.selection, a.er_folds)
          if t == "er" and has_er_folds
          else pick(a.out_dir, bk, t, bd, a.selection))
         for bd in buds for bk in bks for t in TASKS}
    miss = [k for k, v in A.items() if v is None]
    if miss:
        raise SystemExit(f"not found: {miss[:4]}")
    nl = len(A[(buds[0], bks[0], TASKS[0])][2])

    lane = 0.80 / nb
    body = a.lane_h * len(TASKS) * nb
    leg_h = 0.65
    W = a.width or (7.2 if nB == 1 else 3.35 * nB + 0.5)
    fig = plt.figure(figsize=(W, body + leg_h + 0.58))
    gs = fig.add_gridspec(2, nB, height_ratios=[body, leg_h],
                          wspace=0.10, hspace=0.25)

    axes = []
    for pi, bd in enumerate(buds):
        ax = fig.add_subplot(gs[0, pi])
        realized = []
        for ti, t in enumerate(TASKS):
            for bi, bk in enumerate(bks):
                ab, lam, vec = A[(bd, bk, t)]
                realized.append(ab)
                y = ti + 0.10 + bi * lane
                for col, b in enumerate(vec):
                    if ALPHA[b] <= 0:
                        continue
                    ax.add_patch(Rectangle((col - 0.5, y), 1.0, lane * 0.88,
                                           facecolor=COL[bk], alpha=ALPHA[b],
                                           edgecolor="white", lw=0.5, zorder=2))
            if ti:
                ax.axhline(ti, color="#B9C2C3", lw=0.6, zorder=1)
        for x in range(nl + 1):
            ax.axvline(x - 0.5, color="#EDF1F1", lw=0.5, zorder=0)
        ax.set_xlim(-0.5, nl - 0.5)
        ax.set_ylim(len(TASKS), 0)
        ticks = range(nl) if nl == 12 else range(0, nl, 2)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(i + 1) for i in ticks], fontsize=8.5)
        if pi == 0:
            ax.set_yticks([i + 0.5 for i in range(len(TASKS))])
            ax.set_yticklabels([LABEL[t] for t in TASKS],
                               fontweight="bold", fontsize=10)
        else:
            ax.set_yticks([])
        ax.tick_params(length=0, pad=2)
        for sp in ax.spines.values():
            sp.set_visible(False)
        op = (f"target {bd:.2f}" if a.selection == "nearest"
              else f"$\\leq${bd:.2f}")
        title = f"{op} bit (mean {np.mean(realized):.2f})"
        if nB == 1:
            title = f"Learned bit allocation per task ({qn}; {title})"
        ax.set_title(title, fontsize=10.5 if nB == 1 else 9.5,
                     fontweight="bold", pad=6)
        ax.set_xlabel("layer  (input $\\rightarrow$ output)", labelpad=1)
        axes.append(ax)

    # Legend: one ramp row per backbone, gathered under the middle panel.
    # The x-axis label is not added by default. When the figure is used small,
    # not even one line of margin is left between the tick labels and the
    # legend, so it overlaps wherever it is placed (figure coordinates or axes
    # coordinates alike; shrinking the height leaves the font size as it is, so
    # it cuts in). The 1..12 ticks and the caption are enough, so it is turned
    # on with --xlabel only when needed.
    if a.xlabel:
        fig.canvas.draw()
        bb = [ax.get_position() for ax in axes]
        ctr = (min(b.x0 for b in bb) + max(b.x1 for b in bb)) / 2
        b0 = bb[0]
        axes[0].set_xlabel("layer  (input $\\rightarrow$ output)", fontsize=9.5)
        axes[0].xaxis.set_label_coords((ctr - b0.x0) / (b0.x1 - b0.x0), -0.13)

    # A ramp taking the whole figure width looks larger than the body text.
    # Insert empty columns to narrow it.
    gl = gs[1, :].subgridspec(3, nb, height_ratios=[0.45, 1.0, 0.65],
                              wspace=0.28, hspace=0)
    for bi, bk in enumerate(bks):
        cax = fig.add_subplot(gl[1, bi])
        rgb = np.array(matplotlib.colors.to_rgb(COL[bk]))
        cax.imshow(np.array([[(1 - ALPHA[b]) + ALPHA[b] * rgb for b in BITS]]),
                   aspect="auto")
        # 2-bit is a blank cell in the body, so it is a white cell in the legend
        # too. Draw a border to make clear it is the '2-bit category', not 'no data'.
        i2 = BITS.index(2)
        cax.add_patch(Rectangle((i2 - 0.5, -0.5), 1.0, 1.0, fill=False,
                                edgecolor="#9AA5A6", lw=0.9, ls=(0, (2, 1.5)),
                                clip_on=False, zorder=5))
        bkname = {"w2v2": "wav2vec 2.0", "hubert": "HuBERT",
                  "wavlm": "WavLM"}.get(bk, bk)
        cax.set_title(bkname, fontsize=9.5, fontweight="bold", pad=2)
        cax.set_yticks([])
        cax.set_xticks(range(len(BITS)))
        cax.set_xticklabels([f"{b}b" for b in BITS], fontsize=8)
        cax.tick_params(length=0, pad=2)
        for sp in cax.spines.values():
            sp.set_visible(False)

    if nB > 1:
        fig.suptitle(f"Learned bit allocation per task ({qn})",
                     fontsize=10.5, fontweight="bold", y=0.995)
    os.makedirs(a.out, exist_ok=True)
    name = f"fig_alloc_multi_{qn.lower()}_" + "_".join(f"b{b:.2f}" for b in buds)
    for ext in ("pdf", "png"):
        fig.savefig(f"{a.out}/{name}.{ext}")
    print(f"  {a.out}/{name}.pdf / .png")
    for bd in buds:
        print(f"  target {bd:.3f}")
        for t in TASKS:
            row = []
            for bk in bks:
                ab, lam, vec = A[(bd, bk, t)]
                row.append(f"{bk}={ab:.3f}b (lambda={lam:g})")
            print(f"    {LABEL[t]:3}: " + ", ".join(row))


if __name__ == "__main__":
    main()
