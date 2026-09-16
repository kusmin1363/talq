"""Directly search for the per-task 'truly optimal' bit allocation.

The allocation used so far (the earlier ranking-based allocator) solves exactly the additive
surrogate
    min_b sum_k damage_t(k, b_k)
The problem is that this surrogate differs from the true objective
    min_b  m_t( quant_all(b) )
and the measurements have already shown that gap -- uniform is always inside
the feasible set, so on the surrogate DP cannot be worse than uniform, yet in
practice it lost. That difference is super-additivity (one layer at a time is
safe, but lowering them together collapses).

Here we discard the surrogate and search on the true metric. That solves two
things at once:
  * it sidesteps the noise of the sensitivity map (the positive-control pr-asr
    correlation is only 0.325)
  * it splits search set A / reporting set B, breaking the circularity
    (choosing on the same eval set that is then reported on)

Stages:
  faith    -- does the composition approximation give the same ranking as
              sequential GPTQ (kill-switch)
  spread   -- does the allocation change performance at all (kill-switch, the
              real go/no-go)
  greedy   -- start from uniform 4bit and demote one layer at a time by actual
              measurement on set A
  diagonal -- cross-evaluate the 7 resulting allocations on set B

  python -m talq.search --stage faith --backbone hubert
"""
import argparse
import copy
import json
import os
import random
import time

import torch

from talq.paths import CALIB_ROOT, DATA_ROOT, PROBES, QUANT_GPTQ, RESULTS_ROOT

BITS = [2, 3, 4]


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ---------------------------------------------------------------- A/B split
def split_eval(d, half):
    """The eval set into two non-overlapping pieces. Cut even/odd, not
    front/back -- some lists are sorted by speaker and by session, so cutting by
    prefix gives the pieces different distributions."""
    s = slice(None, None, 2) if half == "A" else slice(1, None, 2)
    out = {}
    for t, v in d.items():
        if t in ("pr", "asr"):
            out[t] = (v[0][s],) + tuple(v[1:])
        elif t == "asv":
            out[t] = (v[0][s], v[1])
        else:
            out[t] = v[s]
    return out


# ------------------------------------------------------------- composition
def load_layer_quants(bk, nl):
    """quants/{bk}_L{k}_b{b}.pt -> {(k,b): {param_name: tensor}}"""
    q = {}
    for k in range(nl):
        for b in BITS:
            p = f"{QUANT_GPTQ}/{bk}_L{k}_b{b}.pt"
            if not os.path.exists(p):
                raise SystemExit(f"missing: {p}")
            q[(k, b)] = torch.load(p, map_location="cpu", weights_only=False)["weights"]
    return q


def compose(m, orig, bitvec, LQ, dev):
    """Lay the per-layer quantized weights on as bitvec says. 32 is the fp32 original."""
    sd = m.state_dict()
    for k, b in enumerate(bitvec):
        if b == 32:
            for n in LQ[(k, BITS[0])]:
                sd[n].copy_(orig[n])
        else:
            for n, w in LQ[(k, b)].items():
                sd[n].copy_(w.to(dev))


def sequential(m, orig, bitvec, calib, group, dev):
    """Actual sequential GPTQ quantization."""
    from talq.quant.gptq import quantize_layer
    m.load_state_dict(orig)
    for k, layer in enumerate(m.encoder.layers):
        if bitvec[k] < 32:
            quantize_layer(m, layer, calib, bits=bitvec[k], group=group)


# ------------------------------------------------------------- allocation
def budget_sum(nl, budget):
    s = int(round(budget * nl))
    assert 2 * nl <= s <= 4 * nl, f"budget {budget} impossible"
    return s


def random_alloc(nl, total, rng):
    """Draw an allocation whose sum is total, uniformly.

    n2+n3+n4 = nl,  2n2+3n3+4n4 = total  =>  n3 + 2n4 = total - 2nl
    so sweeping n4 determines (n2,n3,n4). Weighting by the permutation count
    (multinomial coefficient) of each combination is what makes it 'uniform over
    allocations' -- drawing combinations uniformly undersamples the side with
    large n3."""
    from math import factorial
    rem = total - 2 * nl
    cand = []
    for n4 in range(nl + 1):
        n3 = rem - 2 * n4
        n2 = nl - n3 - n4
        if n3 < 0 or n2 < 0:
            continue
        cand.append(((n2, n3, n4),
                     factorial(nl) // (factorial(n2) * factorial(n3) * factorial(n4))))
    if not cand:
        raise SystemExit(f"sum {total} not composable (nl={nl})")
    combos, ws = zip(*cand)
    n2, n3, n4 = rng.choices(combos, weights=ws)[0]
    v = [2] * n2 + [3] * n3 + [4] * n4
    rng.shuffle(v)
    return v


def neighbors_swap(bitvec, rng=None):
    """Budget-preserving neighbors: lower one layer by 1 bit and raise another by 1 bit.
    greedy descends in only one direction from uniform 4bit, so it gets stuck in a
    path-dependent local optimum (measured: on the asr metric spec_ic beat
    spec_asr). Swap neighbors leave that path while preserving sum exactly."""
    out = []
    for i, bi in enumerate(bitvec):
        if bi <= 2:
            continue
        for j, bj in enumerate(bitvec):
            if i == j or bj >= 4:
                continue
            nb = list(bitvec)
            nb[i] -= 1
            nb[j] += 1
            out.append(((i, j), nb))
    if rng is not None:
        rng.shuffle(out)
    return out


def neighbors_demote(bitvec):
    """Every neighbor reachable by a 1-bit demotion."""
    out = []
    for k, b in enumerate(bitvec):
        if b > 2:
            nb = list(bitvec)
            nb[k] = b - 1
            out.append((k, nb))
    return out


# --------------------------------------------------------------------- common
def setup(bk, dev, sv_stride):
    from transformers import AutoModel
    from talq.eval import arm_eval
    from talq.quant import task_fisher as T
    m = AutoModel.from_pretrained(T.BACKBONES[bk], dtype=torch.float32).eval().to(dev)
    m.requires_grad_(False)
    if bk.startswith("wavlm"):
        # quants/ are weights talq.quant.precompute made on the eager MHA path. Read
        # with stock WavLM's fused attention they run a different forward than
        # the one they were fitted to, and on top of that q/k/v are not caught
        # by the hook, so sequential quantization skips those three wholesale.
        # Written with startswith -- with bk == "wavlm", wavlmL silently drops out.
        from talq.eval.calib_io import load_calib
        from talq.quant.precompute import patch_wavlm_eager
        patch_wavlm_eager(m, load_calib(f"{CALIB_ROOT}/emilia_calib.pt", None, dev)[:2])
    probes = arm_eval.load_probes(T.BACKBONES[bk], f"{PROBES}", dev)
    d = arm_eval.load_eval_data(f"{DATA_ROOT}", sv_stride=sv_stride)
    orig = {k: v.clone() for k, v in m.state_dict().items()}
    return m, probes, d, orig, len(m.encoder.layers)


def score(m, probes, data, tasks, n_ctc):
    from talq.eval import arm_eval
    return arm_eval.eval_all(m, {t: probes[t] for t in tasks if t in probes},
                             data, "cuda", n_ctc=n_ctc)


# ------------------------------------------------------- stage: faith
def stage_faith(a):
    """Does the composition approximation give the same 'ranking' as sequential GPTQ.

    The search runs in composition space while deployment is sequential, so even
    if the absolute values differ a little, the search result is valid as long as
    the ranking is preserved. If the ranking breaks, this whole line of work is
    void."""
    import statistics as st
    from talq.eval.calib_io import load_calib
    dev = "cuda"
    m, probes, d, orig, nl = setup(a.backbone, dev, a.sv_stride)
    dA = split_eval(d, "A")
    LQ = load_layer_quants(a.backbone, nl)
    calib = load_calib(f"{CALIB_ROOT}/emilia_calib.pt", None, dev)
    total = budget_sum(nl, a.budget)
    rng = random.Random(0)
    vecs = [[3] * nl, [4] * nl] + [random_alloc(nl, total, rng) for _ in range(a.n_faith - 2)]
    tasks = a.tasks.split(",")
    log(f"faith: {len(vecs)} allocations x {len(tasks)} task, composed vs sequential")
    res = {"composed": [], "sequential": []}
    for i, v in enumerate(vecs):
        compose(m, orig, v, LQ, dev)
        rc = score(m, probes, dA, tasks, a.n_ctc)
        sequential(m, orig, v, calib, a.group, dev)
        rs = score(m, probes, dA, tasks, a.n_ctc)
        res["composed"].append(rc); res["sequential"].append(rs)
        log(f"  [{i+1}/{len(vecs)}] {''.join(map(str,v))} "
            + "  ".join(f"{t}:{rc.get(t,float('nan')):.4f}/{rs.get(t,float('nan')):.4f}" for t in tasks))
    def spear(x, y):
        rx = {i: r for r, i in enumerate(sorted(range(len(x)), key=lambda i: x[i]))}
        ry = {i: r for r, i in enumerate(sorted(range(len(y)), key=lambda i: y[i]))}
        ax = [rx[i] for i in range(len(x))]; ay = [ry[i] for i in range(len(y))]
        mx, my = st.mean(ax), st.mean(ay)
        num = sum((p-mx)*(q-my) for p, q in zip(ax, ay))
        den = (sum((p-mx)**2 for p in ax) * sum((q-my)**2 for q in ay)) ** .5
        return num/den if den > 0 else 0.0
    out = {}
    log("\n  task   Spearman(composed,sequential)   mean|reldev|")
    for t in tasks:
        c = [r[t] for r in res["composed"] if t in r]
        s = [r[t] for r in res["sequential"] if t in r]
        if len(c) != len(vecs):
            continue
        rho = spear(c, s)
        dev_ = st.mean(abs(x-y)/y for x, y in zip(c, s) if y > 0)
        out[t] = {"spearman": rho, "reldev": dev_}
        log(f"  {t:5s} {rho:>16.3f} {dev_:>16.1%}")
    med = st.median([v["spearman"] for v in out.values()]) if out else 0.0
    mdev = st.median([v["reldev"] for v in out.values()]) if out else 1.0
    ok = med >= a.faith_rho and mdev <= a.faith_dev
    log(f"\n  median Spearman={med:.3f} (threshold >={a.faith_rho})  "
        f"reldev={mdev:.1%} (threshold <={a.faith_dev:.0%})  ->  {'PASS' if ok else 'KILL'}")
    json.dump({"per_task": out, "median_spearman": med, "median_reldev": mdev,
               "pass": ok, "vecs": vecs},
              open(f"{RESULTS_ROOT}/alloc_faith_{a.backbone}.json", "w"), indent=1)
    return 0 if ok else 3


# ------------------------------------------------------ stage: spread
def stage_spread(a):
    """Does the allocation change performance at all. The real go/no-go.

    Evaluate N random allocations at the same budget and look at the spread. If
    uniform-3bit is already in the top group and there is no spread, there is
    nothing to gain from any allocator you build."""
    import statistics as st
    dev = "cuda"
    m, probes, d, orig, nl = setup(a.backbone, dev, a.sv_stride)
    dA = split_eval(d, "A")
    LQ = load_layer_quants(a.backbone, nl)
    total = budget_sum(nl, a.budget)
    rng = random.Random(1)
    tasks = [t for t in a.tasks.split(",") if t in probes]
    vecs = [[3] * nl] + [random_alloc(nl, total, rng) for _ in range(a.n_spread)]
    log(f"spread: uniform-3 + {a.n_spread} random, budget={a.budget} (sum {total})")
    rows = []
    for i, v in enumerate(vecs):
        compose(m, orig, v, LQ, dev)
        r = score(m, probes, dA, tasks, a.n_ctc)
        rows.append({"vec": v, **r})
        if i % 5 == 0 or i == len(vecs) - 1:
            log(f"  {i+1}/{len(vecs)}")
    out = {}
    log(f"\n  {'task':5s} {'uniform':>9s} {'best':>9s} {'median':>9s} {'worst':>9s} "
        f"{'uni_rank':>11s} {'best_gain':>9s}")
    live = 0
    for t in tasks:
        vals = [r[t] for r in rows if t in r]
        if len(vals) < len(vecs):
            continue
        u = vals[0]; rest = sorted(vals[1:])
        rank = sum(1 for x in rest if x < u) + 1        # what place uniform comes in (lower is better)
        gain = (u - rest[0]) / u if u > 0 else 0.0
        out[t] = {"uniform": u, "best": rest[0], "median": st.median(rest),
                  "worst": rest[-1], "uniform_rank": rank, "n": len(rest),
                  "best_gain": gain}
        # Do not kill here on 'is there a random allocation better than uniform' --
        # this draws 24 out of a 73,789-wide space, so even what greedy could find
        # is not met at random. What we look at here is 'does the allocation
        # change performance at all'.
        rel_spread = (rest[-1] - rest[0]) / st.median(rest) if st.median(rest) > 0 else 0.0
        out[t]["rel_spread"] = rel_spread
        alive = rel_spread >= a.spread_min
        live += alive
        log(f"  {t:5s} {u:>9.4f} {rest[0]:>9.4f} {st.median(rest):>9.4f} {rest[-1]:>9.4f} "
            f"{rank:>6d}/{len(rest)+1} {gain:>+8.1%} spread{rel_spread:>6.1%}"
            f"{'  <-alive' if alive else ''}")
    ok = live >= a.spread_tasks
    log(f"\n  tasks where the allocation changes performance {live}/{len(out)} (threshold >={a.spread_tasks})  ->  "
        f"{'PASS' if ok else 'KILL'}")
    json.dump({"per_task": out, "n_random": a.n_spread, "budget": a.budget,
               "live_tasks": live, "pass": ok,
               "rows": [{"vec": r["vec"], **{k: v for k, v in r.items() if k != "vec"}}
                        for r in rows]},
              open(f"{RESULTS_ROOT}/alloc_spread_{a.backbone}.json", "w"), indent=1)
    return 0 if ok else 3


# ------------------------------------------------------ stage: greedy
def stage_greedy(a):
    """Start from uniform 4bit and at every step pick the cheapest demotion by
    actual measurement on set A.

    No additive surrogate is used -- every step is measured from the 'current
    composed state', so super-additivity is reflected automatically."""
    dev = "cuda"
    m, probes, d, orig, nl = setup(a.backbone, dev, a.sv_stride)
    dA = split_eval(d, "A")
    LQ = load_layer_quants(a.backbone, nl)
    total = budget_sum(nl, a.budget)
    tasks = [t for t in a.tasks.split(",") if t in probes]
    dB = split_eval(d, "B")
    compose(m, orig, [3] * nl, LQ, dev)
    uniA = score(m, probes, dA, tasks, a.n_ctc)
    uniB = score(m, probes, dB, tasks, a.n_ctc)
    log("  uniform3 baseline  A: " + "  ".join(f"{t}:{uniA[t]:.4f}" for t in tasks))
    log("  uniform3 baseline  B: " + "  ".join(f"{t}:{uniB[t]:.4f}" for t in tasks))
    allocs, trace = {}, {}
    for t in tasks:
        v = [4] * nl
        hist = []
        t0 = time.time()
        while sum(v) > total:
            best = None
            for k, nb in neighbors_demote(v):
                compose(m, orig, nb, LQ, dev)
                s = score(m, probes, dA, [t], a.n_ctc)[t]
                if best is None or s < best[0]:
                    best = (s, k, nb)
            v = best[2]
            hist.append({"sum": sum(v), "demoted_layer": best[1], "score": best[0]})
            log(f"  {t}: sum{sum(v):>3d}  L{best[1]} demoted  {t}={best[0]:.4f}")
        allocs[t] = v
        trace[t] = hist
        compose(m, orig, v, LQ, dev)
        bt = score(m, probes, dB, [t], a.n_ctc)[t]
        gA = (uniA[t] - hist[-1]["score"]) / uniA[t] if uniA[t] > 0 else 0.0
        gB = (uniB[t] - bt) / uniB[t] if uniB[t] > 0 else 0.0
        trace[t + "_summary"] = {"alloc": v, "A": hist[-1]["score"], "B": bt,
                                 "uniA": uniA[t], "uniB": uniB[t],
                                 "gainA": gA, "gainB": gB}
        log(f"  == {t} done {''.join(map(str,v))}  ({time.time()-t0:.0f}s)  "
            f"A {hist[-1]['score']:.4f} vs uni {uniA[t]:.4f} ({gA:+.1%}) | "
            f"B {bt:.4f} vs uni {uniB[t]:.4f} ({gB:+.1%})")
    json.dump({"backbone": a.backbone, "budget": a.budget, "allocs": allocs,
               "uniform_A": uniA, "uniform_B": uniB, "trace": trace},
              open(f"{RESULTS_ROOT}/alloc_greedy_{a.backbone}{a.out_tag}.json", "w"), indent=1)
    log("\nsearched allocations:")
    for t, v in allocs.items():
        log(f"  {t:5s} {''.join(map(str,v))}")
    return 0


# -------------------------------------------------------- stage: polish
def stage_polish(a):
    """first-improvement local search starting from the greedy solution (budget-preserving swap).

    greedy is a monotone path descending one layer at a time from 4bit, so it is
    stuck in a local optimum. Here we sweep the swap neighbors in random order
    and move as soon as the set-A score improves (first-improvement, not
    best-improvement -- there are up to L*(L-1) neighbors, so evaluating all of
    them every time would explode in cost).

    The evaluation budget is cut per task by whichever of --polish-max-evals and
    --polish-max-sec is reached first. For asv the EER is an evaluation over
    trial pairs, ~49s per run, so the time limit is the binding constraint."""
    dev = "cuda"
    src = f"{RESULTS_ROOT}/alloc_greedy_{a.backbone}{a.out_tag}.json"
    if not os.path.exists(src):
        log(f"missing: {src}"); return 3
    G = json.load(open(src))
    allocs = dict(G["allocs"])
    m, probes, d, orig, nl = setup(a.backbone, dev, a.sv_stride)
    dA, dB = split_eval(d, "A"), split_eval(d, "B")
    LQ = load_layer_quants(a.backbone, nl)
    tasks = [t for t in allocs if t in probes]
    compose(m, orig, [3] * nl, LQ, dev)
    uniA = score(m, probes, dA, tasks, a.n_ctc)
    uniB = score(m, probes, dB, tasks, a.n_ctc)
    rng = random.Random(0)
    out = {}
    for t in tasks:
        v = list(allocs[t])
        compose(m, orig, v, LQ, dev)
        cur = score(m, probes, dA, [t], a.n_ctc)[t]
        start = cur
        t0, ev, moves = time.time(), 0, 0
        improved = True
        while improved:
            improved = False
            for (i, j), nb in neighbors_swap(v, rng):
                if ev >= a.polish_max_evals or time.time() - t0 > a.polish_max_sec:
                    break
                compose(m, orig, nb, LQ, dev)
                sc = score(m, probes, dA, [t], a.n_ctc)[t]
                ev += 1
                if sc < cur - 1e-9:
                    v, cur, improved, moves = nb, sc, True, moves + 1
                    log(f"  {t}: L{i}-1 L{j}+1  {t}={sc:.4f}  (eval {ev})")
                    break
            if ev >= a.polish_max_evals or time.time() - t0 > a.polish_max_sec:
                log(f"  {t}: budget exhausted (eval {ev}, {time.time()-t0:.0f}s)")
                break
        compose(m, orig, v, LQ, dev)
        bt = score(m, probes, dB, [t], a.n_ctc)[t]
        allocs[t] = v
        out[t] = {"alloc": v, "A": cur, "A_start": start, "B": bt,
                  "uniA": uniA[t], "uniB": uniB[t], "moves": moves, "evals": ev,
                  "gainB": (uniB[t] - bt) / uniB[t] if uniB[t] > 0 else 0.0}
        log(f"  == {t} polish {''.join(map(str,v))}  {moves} moves/{ev}eval  "
            f"A {start:.4f}->{cur:.4f} | B {bt:.4f} vs uni {uniB[t]:.4f} "
            f"({out[t]['gainB']:+.1%})")
    json.dump({"backbone": a.backbone, "budget": a.budget, "allocs": allocs,
               "uniform_A": uniA, "uniform_B": uniB, "detail": out},
              open(f"{RESULTS_ROOT}/alloc_polish_{a.backbone}{a.out_tag}.json", "w"), indent=1)
    # Update the greedy json so diagonal uses the polish result (the original is kept as .greedy_raw)
    if not os.path.exists(src + ".greedy_raw"):
        os.rename(src, src + ".greedy_raw")
        G["allocs"] = allocs
        json.dump(G, open(src, "w"), indent=1)
    return 0


# ---------------------------------------------------- stage: diagonal
def stage_diagonal(a):
    """Cross-evaluate the 7 searched allocations on set B (the half not used for search).

    On metric t, is spec_t better than the other 6. Choosing on A and reporting
    on B rules out eval-set overfitting -- exactly the point on which the current
    diagonal result is being doubted."""
    import statistics as st
    from math import comb
    dev = "cuda"
    p = f"{RESULTS_ROOT}/alloc_greedy_{a.backbone}{a.out_tag}.json"
    if not os.path.exists(p):
        log(f"missing: {p}"); return 3
    G = json.load(open(p))
    allocs = G["allocs"]
    m, probes, d, orig, nl = setup(a.backbone, dev, a.sv_stride)
    dB = split_eval(d, "B")
    LQ = load_layer_quants(a.backbone, nl)
    tasks = [t for t in allocs if t in probes]
    total = budget_sum(nl, a.budget)
    log(f"diagonal: {len(tasks)} spec x {len(tasks)} metric, set-B evaluation")
    tab = {}
    for tau in tasks:                       # the task the spec was made for
        compose(m, orig, allocs[tau], LQ, dev)
        tab[tau] = score(m, probes, dB, tasks, a.n_ctc)
        log(f"  spec={tau:5s} " + "  ".join(f"{t}:{tab[tau].get(t,float('nan')):.4f}" for t in tasks))
    compose(m, orig, [3] * nl, LQ, dev)
    uni = score(m, probes, dB, tasks, a.n_ctc)
    log(f"  uniform3    " + "  ".join(f"{t}:{uni.get(t,float('nan')):.4f}" for t in tasks))
    ranks, hit = [], 0
    log(f"\n  metric  own-spec rank  vs uniform")
    for t in tasks:
        vals = {tau: tab[tau][t] for tau in tasks if t in tab[tau]}
        if len(vals) != len(tasks):
            continue
        order = sorted(vals, key=lambda z: vals[z])
        r = order.index(t) + 1
        ranks.append(r); hit += (r == 1)
        bu = (uni[t] - vals[t]) / uni[t] if uni.get(t, 0) > 0 else 0.0
        log(f"  {t:5s} {r:>8d}/{len(tasks)} {bu:>+12.1%}")
    n = len(ranks)
    exp = (len(tasks) + 1) / 2
    lo = min(hit, n - hit)
    p_hit = min(1.0, 2 * sum(comb(n, i) for i in range(lo + 1)) / 2 ** n) if n else 1.0
    log(f"\n  1st place {hit}/{n} (chance {1/len(tasks):.1%})  mean rank {st.mean(ranks):.2f} "
        f"(chance {exp:.2f})")
    json.dump({"backbone": a.backbone, "budget": a.budget, "table": tab,
               "uniform": uni, "ranks": dict(zip(tasks, ranks)),
               "hit": hit, "n": n, "mean_rank": st.mean(ranks) if ranks else None},
              open(f"{RESULTS_ROOT}/alloc_diag_{a.backbone}.json", "w"), indent=1)
    return 0


def stage_merge(a):
    """Merge the greedy jsons that were run in parallel per task into one."""
    import glob
    pat = f"{RESULTS_ROOT}/alloc_greedy_{a.backbone}_t*.json"
    files = sorted(glob.glob(pat))
    if not files:
        log(f"no files to merge: {pat}"); return 3
    allocs, uniA, uniB, trace = {}, {}, {}, {}
    for f in files:
        d = json.load(open(f))
        allocs.update(d.get("allocs", {}))
        uniA.update(d.get("uniform_A", {}))
        uniB.update(d.get("uniform_B", {}))
        trace.update(d.get("trace", {}))
        log(f"  + {os.path.basename(f)}: {sorted(d.get('allocs', {}))}")
    out = f"{RESULTS_ROOT}/alloc_greedy_{a.backbone}.json"
    json.dump({"backbone": a.backbone, "budget": a.budget, "allocs": allocs,
               "uniform_A": uniA, "uniform_B": uniB, "trace": trace},
              open(out, "w"), indent=1)
    log(f"merged -> {out}  ({len(allocs)} task: {sorted(allocs)})")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["faith", "spread", "greedy", "polish", "diagonal", "merge"])
    ap.add_argument("--backbone", default="hubert")
    ap.add_argument("--budget", type=float, default=3.0)
    ap.add_argument("--tasks", default="pr,asr,er,ks,ic,sid,asv")
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--n-ctc", type=int, default=64)
    ap.add_argument("--sv-stride", type=int, default=8)
    ap.add_argument("--n-faith", type=int, default=8)
    ap.add_argument("--n-spread", type=int, default=24)
    ap.add_argument("--out-tag", default="",
                    help="Suffix for the result json file name. Keeps parallel per-task "
                         "runs from overwriting each other. Merge later with --stage merge.")
    ap.add_argument("--polish-max-evals", type=int, default=200)
    ap.add_argument("--polish-max-sec", type=float, default=1800)
    # kill-switch thresholds
    ap.add_argument("--faith-rho", type=float, default=0.5,
                    help="lower bound on the median composed vs sequential Spearman")
    ap.add_argument("--faith-dev", type=float, default=0.25,
                    help="upper bound on the median relative deviation")
    ap.add_argument("--spread-min", type=float, default=0.15,
                    help="minimum relative spread of the random allocations at the same "
                         "budget. Below this the allocation does not change performance, "
                         "so the line of work itself dies.")
    ap.add_argument("--spread-tasks", type=int, default=3,
                    help="minimum number of tasks that must have headroom")
    a = ap.parse_args()
    torch.manual_seed(0)
    return {"faith": stage_faith, "spread": stage_spread,
            "greedy": stage_greedy, "polish": stage_polish,
            "diagonal": stage_diagonal, "merge": stage_merge}[a.stage](a)


if __name__ == "__main__":
    raise SystemExit(main())
