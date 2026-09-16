"""1024s main sweep runner -- replaces the earlier shell chain and sweep runner.

Why it was rewritten (2026-09-08). The old scheduler gated concurrency on
**worker count** alone, so 91 of the 100 cells of the GPTQ large stage 2 came out
silently empty. Three things overlapped:

  1. jobs_override.json {"large":4} overrode --jobs 2 with 4. large peaks at 25GB
     per process, so 4-way is 100GB > 70.2GB -- a guaranteed OOM.
  2. talq.alloc catches per-task OOM, prints only [FAIL] and **exits with rc=0**.
     the earlier sweep runner looked only at rc, so a 5-task cell was recorded as
     "done" in 40s.
  3. The earlier sweep runner's comment said "the real worker count is decided by run_pool from
     the GPU headroom", but run_pool had no such code. That comment is what made
     override 4 look safe.

So this runner has three rules.

  **Admit on GB, not on count.** What decides memory is not the backbone but the
  task's audio length (measured: wavlmL on ks/ic is 10.1GB, hubertL on pr is
  23.5GB). So a job is split into (backbone, lambda, weight class). heavy=asv/asr/pr
  (8s), light=ks/ic/er (1-6s). Splitting pays one more setup per cell (about 120s,
  4%) but raises large from 2-way to 3-4-way. Leaving the 5 tasks in one lump hits
  the peak sooner or later anyway, so workers would always have to be sized on the
  worst case.

  **Do not trust rc.** When a job ends, re-read the cell JSON and see whether the
  task actually landed in allocs. If it is empty, requeue (--max-retry). Tasks
  already present are not touched (talq.alloc's todo filter), so correct results
  pass automatically.

  **Learn from OOM.** If the log has OutOfMemoryError, raise that class's
  reservation and record it in runner_state/reservations.json. From the next run on,
  admit with that value.

If two processes write the same cell JSON at once the later one overwrites the
earlier (talq.alloc reads the whole file and writes it back whole). So a **mutex per
cell file** keeps the heavy and light job of the same (out_dir, bk, lambda) from
being launched at the same time.

  python -m talq.sweep --dry-run        # remaining cells, reservation plan, estimated time
  python -m talq.sweep                  # run (resumable, keyed on artifacts)
  python -m talq.sweep --progress       # the same table as talq.sweep_progress
"""
import argparse
import json
import os
import subprocess
import sys
import time

from talq import tb_log
from talq.paths import LOGS_ROOT, REPO_ROOT, RESULTS_ROOT, TB_ROOT

PY = sys.executable

T5 = ["asv", "asr", "pr", "ic", "ks"]          # the 5 non-ER tasks
LAMS = [0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 5]
FOLDS = [1, 2, 3, 4, 5]
BASE_BKS = ["w2v2", "wavlm", "hubert"]
LARGE_BKS = ["wavlmL", "hubertL"]

# The classes follow **talq.eval.salience.MAX_S (the audio crop length)**. Both
# memory and time are decided there. Do not guess; look at that table:
#   8.0s  asv asr pr -> heavy   large measured peak 25.9GB / 57 min per job
#   6.0s  ic         -> mid     large measured peak 19.9GB /  9 min per job
#   6.0s  er         -> er      large measured 15.6GB+ , base 9.3GB
#   1.0s  ks         -> light   large measured  9.1GB
# Putting er in the same class as ks cost us on 2026-09-09. er is 6s, the same class
# as ic, but it was reserved against the 1s ks figure (11GB) and kept OOMing. It is
# kept apart from ic because the measurements differ even at the same 6s
# (large 15.6 vs 19.9).
HEAVY = {"asv", "asr", "pr"}
MID = {"ic"}
CLASSES = ["heavy", "mid", "er", "light"]

# Seconds per cell. Inherited from the earlier sweep runner's cost table (w2v2
# measured under 5-way contention).
COST = {"asv": 1320, "asr": 1310, "pr": 940, "ic": 520, "er": 500, "ks": 300}
SETUP = 120                                    # model / eval-set load per invocation
BK_SCALE = {"w2v2": 1.0, "wavlm": 1.21, "hubert": 1.0, "wavlmL": 1.3, "hubertL": 1.3}

# GPU reservations (GB). Measured basis:
#   large/heavy  peaks 26.83 / 22.77 / 26.05 in the OOM logs, 23.5 for hubertL(pr) while running
#   large/light  10.1 for wavlmL(ks/ic) while running
#   base/*       the earlier sweep runner's comment "about 11GB per process" is the
#                5-task peak = heavy
# If runner_state/reservations.json exists, that one wins (OOM self-correction).
# mid has no completed-run measurement yet (ic only ever died at the cap). Reserve
# generously, let it finish once, then lower it from the "GPU peak" in the log.
# Reserving too little is the more expensive side.
RESERVE = {("base", "heavy"): 12.0, ("base", "mid"): 13.0,
           ("base", "er"): 10.0, ("base", "light"): 6.0,
           ("large", "heavy"): 27.0, ("large", "mid"): 20.0,
           ("large", "er"): 18.0, ("large", "light"): 10.0}
GPU_TOTAL_GB = 70.2                            # H200 usable share (per nvidia-smi)
GPU_FRAC = 0.80                                # use only up to this fraction. Aiming at 100%
# means one fragmentation event kills a neighbour -- that is how 13 of 20 cells were
# ruined on 2026-09-08. The remaining 20% is left as the share eaten by the caching
# allocator's fragmentation and by the CUDA context.
NCPU = 12                                      # container cpuset (the host has 224)

STATE = f"{REPO_ROOT}/runner_state"


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


class RunnerTB:
    """The runner's own time series. What an operator watches is not learning curves
    but **occupancy and throughput**: how much GPU is in use, how many are running
    and how many slots are left, when an OOM happened. The loss of each individual
    job is left by the run on the talq.alloc side.

    The x axis is **elapsed seconds** since the runner started. Using the poll
    interval (20s by default) as the step would change the meaning of the x axis the
    moment the interval changes, and using minutes would stack three points on the
    same x. TB also keeps a separate wall time axis, so it can be viewed in absolute
    time as well.
    """

    def __init__(self, root):
        self.tb = tb_log.TB(root, "_runner") if root else tb_log.TB.off()
        self.t0 = time.time()

    def _x(self):
        return int(time.time() - self.t0)

    def sample(self, used, occupied, budget, n_run, n_pend, cells_left, cells_tot):
        x = self._x()
        if used is not None:
            self.tb.scalar("gpu/used_gb", used, x)
        self.tb.scalar("gpu/occupied_gb", occupied, x)      # measured + not-yet-reached reservation
        self.tb.scalar("gpu/budget_gb", budget, x)
        self.tb.scalar("gpu/headroom_gb", budget - occupied, x)
        self.tb.scalar("jobs/running", n_run, x)
        self.tb.scalar("jobs/pending", n_pend, x)
        self.tb.scalar("progress/cells_done", cells_tot - cells_left, x)
        self.tb.scalar("progress/cells_left", cells_left, x)
        if cells_tot:
            self.tb.scalar("progress/pct", 100.0 * (cells_tot - cells_left) / cells_tot, x)

    def job_done(self, j, peak, ok, oom, mins):
        x = self._x()
        self.tb.scalar("job/wall_min", mins, x)
        if peak:
            self.tb.scalar("job/peak_gb", peak, x)
            self.tb.scalar("job/peak_over_reserve", peak - j["res"], x)
        self.tb.scalar("job/ok", 1 if ok else 0, x)
        # Log the 0 too. Emitting only on OOM gives a series with a single point, and
        # the graph then does not show when it did not happen.
        self.tb.scalar("job/oom", 1 if oom else 0, x)
        self.tb.text("event", f"{'done' if ok else 'OOM' if oom else 'requeue'} "
                              f"{j['label']} peak={peak or float('nan'):.1f}GB "
                              f"res={j['res']:g}GB {mins:.0f}min", x)
        self.tb.flush()

    def reserve(self, tab):
        x = self._x()
        for (sz, c), v in tab.items():
            self.tb.scalar(f"reserve/{sz}_{c}_gb", v, x)

    def close(self):
        self.tb.close()


def size_of(bk):
    return "large" if bk in LARGE_BKS else "base"


def cls_of(task):
    return ("heavy" if task in HEAVY else "er" if task == "er"
            else "mid" if task in MID else "light")


# ---------------------------------------------------------------- planning

def stages(quantizers, sizes):
    """List of (label, out_dir, quantizer, backbones, tasks, er_fold).

    The order inherits the earlier shell chain's rationale as is -- the order in which
    usable results come out earliest. In each bundle the 5 non-ER tasks have to
    finish first for the table draft to grow. This runner does however **not block
    the stages serially**. They are only a priority, and if the earlier stage does
    not fit in memory the gaps are filled with small jobs from a later stage.
    """
    bk_of = {"base": BASE_BKS, "large": LARGE_BKS}
    out = []
    for q in quantizers:
        for sz in sizes:
            bks = bk_of[sz]
            out.append((f"{q} {sz} non-ER", f"sw_{q}", q, bks, T5, None))
            for n in FOLDS:
                out.append((f"{q} {sz} ERf{n}", f"sw_{q}_erf{n}", q, bks,
                            ["er"], n))
    return out


def prime_bank(qs, szs):
    """**Merge** the already-measured 3/4-bit uniform baselines into the base JSON.

    base_missing() demands all of (2,3,4), so the baseline job comes up anyway, but
    talq.alloc skips bits that are already there, so in practice only 2 bits are
    measured. Two thirds of the baseline cost disappears here.

    The bank generator that produced results/uniform_bank/{q}.json is never re-run
    to emit these baselines. Its emit path **rewrites the base JSON whole**, so
    emitting with --uniform-bits 3,4 erases the already-measured 2 bits (on
    2026-09-08 40 slots across 20 files were actually lost. A copy was left in the
    cell JSONs, so it was recovered). Here only the missing slots are filled.
    """
    banks = {q: json.load(open(f"{RESULTS_ROOT}/uniform_bank/{q}.json"))
             for q in set(qs)}
    n = 0
    for lab, out, q, bks, tasks, fold in stages(qs, szs):
        for bk in bks:
            b = banks[q].get(bk)
            if not b:
                continue
            fp = base_fp(out, bk)
            cur = json.load(open(fp)) if os.path.exists(fp) else {
                "backbone": bk, "budget": 3, "allocs": {}, "probs": {},
                "eval": {}, "hist": {}, "avg": {}, "uniform": {}}
            for bit in ("3", "4"):
                slot = cur.setdefault("uniform", {}).setdefault(bit, {})
                for t in tasks:
                    if t in slot:
                        continue
                    # The ER baseline differs per fold. The average must not be used.
                    v = (b["er_per_fold"].get(bit, {}).get(str(fold))
                         if t == "er" and fold else
                         b["uniform"].get(bit, {}).get(t))
                    if v is not None:
                        slot[t] = v
                        n += 1
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            json.dump(cur, open(fp, "w"), indent=1)
    log(f"  filled {n} slots from the bank (3/4 bit)")


def qdir(q):
    return f"quants_1024/{'GPTQ' if q == 'gptq' else 'AWQ'}/quants_tc1024_{{task}}"


def cell_fp(out, bk, lam):
    return f"{RESULTS_ROOT}/{out}/{bk}/l{lam:g}/{bk}_b3.json"


def base_fp(out, bk):
    return f"{RESULTS_ROOT}/{out}/{bk}/base/{bk}_b3.json"


def done_tasks(out, bk, lam):
    """The tasks that **actually landed** in the cell JSON. The artifact is the
    criterion, not the log."""
    try:
        return set(json.load(open(cell_fp(out, bk, lam))).get("allocs", {}))
    except Exception:
        return set()


def base_missing(out, bk, tasks, bits=(2, 3, 4)):
    """(bit, task) missing from the uniform baseline. Even one raises a baseline job."""
    try:
        u = json.load(open(base_fp(out, bk))).get("uniform", {})
    except Exception:
        u = {}
    return [t for t in tasks if any(t not in u.get(str(b), {}) for b in bits)]


def seed_cell(out, bk, lam):
    """Plant the uniform baseline into the cell directory (inherited from
    the earlier sweep runner's seeding step). It is full-split evaluation and therefore the most expensive,
    so it is measured once per (backbone, task) and copied per lambda.
    **Always call it inside the cell mutex** -- a running job's JSON must not be
    overwritten."""
    b = base_fp(out, bk)
    if not os.path.exists(b):
        return
    u = json.load(open(b)).get("uniform", {})
    fp = cell_fp(out, bk, lam)
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    r = json.load(open(fp)) if os.path.exists(fp) else {
        "backbone": bk, "budget": 3.0, "allocs": {}, "probs": {},
        "eval": {}, "uniform": {}, "hist": {}, "avg": {}}
    for k, v in u.items():
        r["uniform"].setdefault(k, {}).update(v)
    json.dump(r, open(fp, "w"), indent=1)


def build_jobs(quantizers, sizes, lams, res_tab):
    """The remaining work as a job list. job = dict.

    kind='base'  one (out, bk). Every cell job of that (out,bk) comes after it.
    kind='cell'  (out, bk, lam, cls) -- only tasks of the same class are bundled.
    """
    jobs, prio = [], 0
    for lab, out, q, bks, tasks, fold in stages(quantizers, sizes):
        prio += 1
        for bk in bks:
            need = base_missing(out, bk, tasks)
            if need:
                jobs.append(dict(
                    kind="base", key=f"{out}/{bk}/base", label=f"base {bk} [{lab}]",
                    out=out, q=q, bk=bk, lam=None, tasks=need, cls="heavy",
                    fold=fold, prio=prio, mutex=base_fp(out, bk), tries=0,
                    cost=(len(need) * 450 + SETUP) * BK_SCALE[bk],
                    res=res_tab[(size_of(bk), "heavy")]))
            for lam in lams:
                left = [t for t in tasks if t not in done_tasks(out, bk, lam)]
                if not left:
                    continue
                for c in CLASSES:
                    ts = [t for t in left if cls_of(t) == c]
                    if not ts:
                        continue
                    jobs.append(dict(
                        kind="cell", key=f"{out}/{bk}/l{lam:g}/{c}",
                        label=f"{out.replace('sw_','')} {bk}/l{lam:g} {c}"
                              f"[{','.join(ts)}]",
                        out=out, q=q, bk=bk, lam=lam, tasks=ts, cls=c,
                        fold=fold, prio=prio, mutex=cell_fp(out, bk, lam), tries=0,
                        cost=(sum(COST[t] for t in ts) + SETUP) * BK_SCALE[bk],
                        res=res_tab[(size_of(bk), c)],
                        dep=f"{out}/{bk}/base" if need else None))
    # Within a priority, the long ones first. If the short ones go in first, only the
    # long ones are left in the tail.
    jobs.sort(key=lambda j: (j["prio"], j["kind"] != "base", -j["cost"]))
    return jobs


# ---------------------------------------------------------------- execution

def cmd_of(j):
    q = qdir(j["q"])
    if j["kind"] == "base":
        c = [PY, "-m", "talq.alloc", "--backbones", j["bk"],
             "--tasks", ",".join(j["tasks"]), "--uniform-bits", "2,3,4",
             "--quant-dir", q, "--baseline-only",
             "--out", f"{RESULTS_ROOT}/{j['out']}/{j['bk']}/base"]
    else:
        # --init is not given. talq.alloc adjusts it automatically to the bitset length.
        c = [PY, "-m", "talq.alloc", "--backbones", j["bk"],
             "--tasks", ",".join(j["tasks"]), "--objective", "kl",
             "--forward", "gumbel", "--project", "none", "--rate", "linear",
             "--lam", str(j["lam"]), "--beta", "0", "--steps", "2000",
             "--bs", "16", "--pool-sec", "3600", "--uniform-bits", "2,3,4",
             "--quant-dir", q, "--bitset", "2,3,4",
             "--out", os.path.dirname(cell_fp(j["out"], j["bk"], j["lam"]))]
    if j["fold"]:
        # Leaving out --er-fold makes quant-dir resolve to ..._er and look for files
        # that do not exist.
        c += ["--er-fold", str(j["fold"])]
    if j.get("init"):
        # For the init ablation only. Main sweep jobs do not put this key in, so
        # talq.alloc's default 0.05,0.05,0.9 + automatic bitset adjustment path stands.
        c += ["--init", str(j["init"])]
    if j.get("seed"):
        # For the seed reproducibility experiment only. Main sweep jobs do not put
        # this key in, so seed 0 (the default) stands -- not one character of the
        # command line differs from the existing artifacts.
        c += ["--seed", str(j["seed"])]
    return c


def log_path(j):
    tag = "base" if j["kind"] == "base" else f"l{j['lam']:g}_{j['cls']}"
    return f"{LOGS_ROOT}/{j['out']}_{j['bk']}_{tag}.log"


def gpu_used_gb():
    """(total used, {pid: used}) -- both in GB. Measured with nvidia-smi.

    Counting only the reservation sum misses processes outside the runner.
    Conversely, looking only at the measurement overestimates the headroom because a
    just-launched job has not reached its peak yet. So the two are combined:
        occupied = measured total + sum(our job's reservation - that job's current use)
    """
    try:
        g = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"], text=True, timeout=20)
        tot = float(g.strip().splitlines()[0]) / 1024.0
        p = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"], text=True, timeout=20)
        per = {}
        for ln in p.strip().splitlines():
            if "," in ln:
                pid, mb = ln.split(",")
                per[int(pid)] = float(mb) / 1024.0
        return tot, per
    except Exception as e:
        log(f"  [warning] nvidia-smi failed ({type(e).__name__}) -- admitting on the reservation table alone")
        return None, {}


def tail_since(j):
    """Only the part this attempt wrote to the log. The log is append, so OOMs left
    by the old chain are mixed in. Scanning the whole thing would keep raising the
    reservation from those old OOMs."""
    try:
        with open(log_path(j), errors="replace") as f:
            f.seek(j.get("_off", 0))
            return f.read()
    except Exception:
        return ""


def verify(j):
    """Judged on the artifact, not on rc. (remaining tasks, whether OOM)"""
    if j["kind"] == "base":
        left = base_missing(j["out"], j["bk"], j["tasks"])
    else:
        left = [t for t in j["tasks"]
                if t not in done_tasks(j["out"], j["bk"], j["lam"])]
    return left, "OutOfMemoryError" in tail_since(j)


def peak_gb(j):
    """The reserved peak talq.alloc prints at the end. + about 0.6GB of CUDA context
    is the value nvidia-smi picks up. Kept so that an over- or under-sized
    reservation can be seen by eye. Only this attempt's span is read (the previous
    attempt's peak must not be read)."""
    try:
        for ln in reversed(tail_since(j).splitlines()):
            if "GPU peak" in ln and "reserved" in ln:
                return float(ln.split("reserved")[1].split("GB")[0]) + 0.6
    except Exception:
        pass
    return None


def load_reservations():
    tab = dict(RESERVE)
    try:
        for k, v in json.load(open(f"{STATE}/reservations.json")).items():
            sz, c = k.split("/")
            tab[(sz, c)] = float(v)
    except FileNotFoundError:
        pass
    except Exception as e:
        log(f"ignoring reservations.json ({type(e).__name__}: {e})")
    return tab


def save_reservations(tab):
    os.makedirs(STATE, exist_ok=True)
    json.dump({f"{s}/{c}": v for (s, c), v in tab.items()},
              open(f"{STATE}/reservations.json", "w"), indent=1)


def run(jobs, a, res_tab):
    budget = GPU_TOTAL_GB * a.gpu_frac
    # OMP is matched to the worker count. 8 threads x 4 workers on 12 cores is a 3x
    # oversubscription and the threads push each other out. The expected worker count
    # is derived back from the reservations and divided by.
    exp_workers = max(1, int(budget / min(res_tab.values())))
    omp = a.omp or max(2, min(8, NCPU // max(1, min(exp_workers, 5))))
    env = dict(os.environ, OMP_NUM_THREADS=str(omp),
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
               # The child (talq.alloc) writes under the same root. Empty turns the
               # child off too.
               DA_TB_DIR=a.tb_dir or "")
    tb = RunnerTB(a.tb_dir)
    tb.reserve(res_tab)
    cells_tot = sum(len(j["tasks"]) for j in jobs if j["kind"] == "cell")
    log(f"GPU budget {budget:.1f}GB = {a.gpu_frac:.0%} of {GPU_TOTAL_GB}GB total  "
        f"OMP_NUM_THREADS={omp}")
    log("reservation table " + "  ".join(f"{s}/{c}={v:g}GB"
                                         for (s, c), v in sorted(res_tab.items())))

    pend, running, held, failed, last_start = list(jobs), [], set(), [], 0.0
    blocked_rounds = 0
    while pend or running:
        tot_used, per_pid = gpu_used_gb()
        if tot_used is None:                    # if nvidia-smi died, reservation table only
            occupied = sum(j["res"] for j, _, _ in running)
        else:
            # Measured + the shortfall of our jobs that have not reached their peak
            # yet. External processes are already in the measured side, so they are
            # picked up in the calculation automatically.
            occupied = tot_used + sum(
                max(0.0, j["res"] - per_pid.get(p.pid, 0.0))
                for j, p, _ in running)
        cells_left = sum(len(j["tasks"]) for j in pend + [x[0] for x in running]
                         if j["kind"] == "cell")
        tb.sample(tot_used, occupied, budget, len(running), len(pend),
                  cells_left, cells_tot)
        started_any = staggered = False
        for idx, j in enumerate(list(pend)):
            if len(running) >= a.max_jobs:
                break
            # Cell JSON mutex: if two processes write the same file the later one
            # overwrites the earlier
            if j["mutex"] in held:
                continue
            if j.get("dep") and base_fp(j["out"], j["bk"]) in held:
                continue          # that (out,bk)'s baseline job is still running
            if j.get("dep") and base_missing(j["out"], j["bk"], j["tasks"]):
                continue          # the baseline is not filled in yet
            fits = occupied + j["res"] <= budget
            if not fits:
                # If a big job keeps being pushed back, only small jobs get eaten and
                # the tail grows long.
                if idx == 0:
                    blocked_rounds += 1
                if blocked_rounds >= a.head_block:
                    break         # stop admitting until the head one gets in
                continue
            # Simultaneous launches are forbidden. Allocations pile up in the model
            # load + Mixer construction stretch, which is what caused the 2026-09-08
            # OOM. Give the previous job time to take its peak.
            if time.time() - last_start < a.stagger:
                staggered = True      # not a memory problem. Do not misdiagnose it
                break
            os.makedirs(os.path.dirname(log_path(j)), exist_ok=True)
            if j["kind"] == "cell":
                seed_cell(j["out"], j["bk"], j["lam"])
            f = open(log_path(j), "a")
            f.write(f"\n===== talq.sweep {time.strftime('%F %T')} "
                    f"try{j['tries']+1} {' '.join(cmd_of(j))}\n")
            f.flush()
            j["_off"] = f.tell()       # start of the span verify/peak_gb will look at
            # The per-process cap is a **blast radius limit**, not a precision
            # tripwire. Giving it exactly the reservation kills a job whole just for
            # going a few % over the peak (on 2026-09-08 base/light died that way at
            # 7.6 vs 7.0). For large heavy that burns 80 minutes before dying. Put it
            # 10% above the reservation -- since the budget is 80% of the card, even
            # if everyone uses up to the cap the card as a whole does not overflow.
            cap = j["res"] * (a.cap_margin + 0.25 * j["tries"])
            jenv = dict(env, DA_MEM_GB=f"{cap:.1f}")
            p = subprocess.Popen(cmd_of(j), cwd=REPO_ROOT, env=jenv, stdout=f,
                                 stderr=subprocess.STDOUT)
            pend.remove(j)
            running.append((j, p, f))
            held.add(j["mutex"])
            occupied += j["res"]           # the job just launched still uses 0, but
            last_start = time.time()       # it is counted as taking its reservation
            j["_t0"] = last_start
            started_any = True
            blocked_rounds = 0
            log(f"  start {j['label']}  ({j['res']:g}GB, occupied {occupied:.0f}/"
                f"{budget:.0f}GB, running {len(running)}, left {len(pend)})")
        if not started_any and not running and pend:
            if staggered:
                time.sleep(max(1.0, a.stagger - (time.time() - last_start)))
            else:
                log("  [stalled] no job is admitted. Check whether a process outside "
                    "the runner is holding the GPU. Retrying in 60s")
                time.sleep(60)
        for j, p, f in [t for t in running if t[1].poll() is not None]:
            f.close()
            running.remove((j, p, f))
            held.discard(j["mutex"])
            left, oom = verify(j)
            peak = peak_gb(j)
            tb.job_done(j, peak, not left, oom,
                        (time.time() - j.get("_t0", time.time())) / 60)
            if peak:
                log(f"  peak {peak:.1f}GB / reservation {j['res']:g}GB  {j['label']}")
            if oom:
                # Take the measured peak as the floor. Multiplying blindly by 15%
                # each time makes **several OOMs from the same cause count as
                # independent evidence** and the reservation inflates compounded
                # (2026-09-08: both peaks were 15.8GB but the reservation went
                # 14->16.1->18.5). There is a cap too -- light costing more than
                # heavy makes no sense.
                k = (size_of(j["bk"]), j["cls"])
                want = max(peak, res_tab[k]) * 1.25
                want = min(want, res_tab[(k[0], "heavy")])
                if want > res_tab[k] + 0.05:
                    res_tab[k] = round(want, 1)
                    save_reservations(res_tab)
                    tb.reserve(res_tab)
                    why = f" (measured peak {peak:.1f}GB)" if peak else ""
                    log(f"  [OOM] {j['label']} -> reservation {k[0]}/{k[1]} "
                        f"raised to {res_tab[k]:g}GB{why}")
                else:
                    log(f"  [OOM] {j['label']} -- keeping reservation {res_tab[k]:g}GB "
                        f"(already at or above the measured peak {peak:.1f}GB)")
            if not left:
                log(f"  done {j['label']}")
                continue
            j["tries"] += 1
            if j["tries"] >= a.max_retry:
                failed.append((j["label"], left))
                log(f"  [gave up] {j['label']} remaining {left} ({j['tries']} retries)")
                continue
            j["tasks"] = left
            j["res"] = res_tab[(size_of(j["bk"]), j["cls"])]
            pend.insert(0, j)
            log(f"  requeue {j['label']} remaining {left} (rc={p.returncode}, "
                f"attempt {j['tries']})")
        if running:
            time.sleep(a.poll)
    tb.close()
    log("=== runner finished ===")
    if failed:
        log(f"{len(failed)} jobs never filled:")
        for lab, left in failed:
            log(f"  {lab} -> {left}")
    return 1 if failed else 0


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quantizers", default="gptq,awq")
    ap.add_argument("--sizes", default="large,base",
                    help="large first is the default -- Table 1/2 is blocked on large2, "
                         "and large eats a lot of memory, so pushing it back lengthens the tail.")
    ap.add_argument("--lams", default=",".join(f"{l:g}" for l in LAMS))
    ap.add_argument("--max-jobs", type=int, default=8,
                    help="A cap only. The actual concurrency is decided by the GB reservations.")
    ap.add_argument("--stagger", type=float, default=90,
                    help="Launch interval (seconds). If model load + Mixer construction overlap it blows up.")
    ap.add_argument("--poll", type=float, default=20)
    ap.add_argument("--max-retry", type=int, default=4)
    ap.add_argument("--head-block", type=int, default=4,
                    help="If the head job is pushed back this many times, stop admitting and wait. "
                         "Letting only small jobs through starves the big ones.")
    ap.add_argument("--omp", type=int, default=None,
                    help="The default is derived back from the worker count (on a 12-core basis).")
    ap.add_argument("--cap-margin", type=float, default=1.10,
                    help="Per-process GPU cap = reservation x this value. Do not leave it at 1.0 "
                         "-- a job that goes slightly over its reservation dies whole and wastes time.")
    ap.add_argument("--gpu-frac", type=float, default=GPU_FRAC,
                    help="Use GPU memory only up to this fraction (0.80 by default). Do not aim "
                         "at 100%% -- the remaining share is eaten by caching allocator "
                         "fragmentation and the CUDA context. Before raising it, look at "
                         "'peak X / reservation Y' in the log first.")
    ap.add_argument("--reserve", default=None, metavar="large/heavy=24,...",
                    help="Manual reservation adjustment. Lower it **only after seeing a measured "
                         "peak**. The runner only raises by 15%% on OOM and never lowers on its "
                         "own -- lowering is a judgement a human makes from the evidence.")
    ap.add_argument("--tb-dir", default=os.environ.get("DA_TB_DIR", f"{TB_ROOT}"),
                    help="TensorBoard event root. The runner time series goes under _runner/ there, "
                         "the per-job learning curves stack at the cell path as is. "
                         "A single tensorboard --logdir tbruns shows both.")
    ap.add_argument("--no-tb", action="store_true",
                    help="Turn off TensorBoard logging on both the runner and the child jobs.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--progress", action="store_true")
    a = ap.parse_args()

    qs = [x for x in a.quantizers.split(",") if x]
    szs = [x for x in a.sizes.split(",") if x]
    lams = [float(x) for x in a.lams.split(",") if x]
    if a.no_tb:
        a.tb_dir = ""
    res_tab = load_reservations()
    for kv in (a.reserve or "").split(","):
        if kv.strip():
            k, v = kv.split("="); sz, c = k.strip().split("/")
            res_tab[(sz, c)] = float(v)
    if not a.progress:
        log("planting the 3/4-bit baselines from the uniform bank ...")
        prime_bank(qs, szs)
    jobs = build_jobs(qs, szs, lams, res_tab)

    if a.progress:
        os.execv(PY, [PY, "-m", "talq.sweep_progress"])

    ncell = sum(len(j["tasks"]) for j in jobs if j["kind"] == "cell")
    work = sum(j["cost"] for j in jobs)
    budget = GPU_TOTAL_GB * a.gpu_frac
    # Parallel efficiency: take the average concurrency from the reservations and
    # subtract a 20% contention loss (the same coefficient as the measured base 5-way
    # PAR_EFF 3.9 = 5 x 0.78).
    par = max(1.0, sum(budget / j["res"] * j["cost"] for j in jobs)
              / max(work, 1) * 0.78)
    log(f"{len(jobs)} jobs (baseline {sum(1 for j in jobs if j['kind']=='base')}, "
        f"cell {sum(1 for j in jobs if j['kind']=='cell')}) / {ncell} cells")
    log(f"estimate: cumulative work {work/3600:.1f}h / parallel efficiency {par:.1f} "
        f"-> about {work/par/3600:.1f} hours wall clock")
    if a.dry_run:
        by = {}
        for j in jobs:
            by.setdefault(j["prio"], []).append(j)
        for p in sorted(by):
            g = by[p]
            print(f"  [{p:2d}] {g[0]['out']:16s} {g[0]['bk'] if len({x['bk'] for x in g})==1 else '·':8s}"
                  f" job {len(g):3d}  cells {sum(len(x['tasks']) for x in g):3d}"
                  f"  {sum(x['cost'] for x in g)/3600:6.1f}h")
        print("\n  first 12 jobs:")
        for j in jobs[:12]:
            print(f"    {j['res']:4g}GB  {j['cost']/3600:5.2f}h  {j['label']}")
        return
    sys.exit(run(jobs, a, res_tab))


if __name__ == "__main__":
    main()
