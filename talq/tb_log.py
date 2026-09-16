"""TensorBoard event logging -- for the run records the server admin requires.

There is one design principle. **Never kill the experiment.** If tensorboard is
missing or writing an event fails, training goes on unchanged (ignore and
proceed, warn once). That is why every public method here swallows exceptions.

Directory convention:
    {DA_TB_DIR or talq.paths.TB_ROOT}/{run tag}/{YYYYmmdd-HHMMSS}
The run tag is the cell path as-is -- e.g. sw_gptq_erf2/wavlmL/l1/er.
Retrying the same cell gets a different timestamp, so it does not overwrite the
previous curves (the runner re-queues after OOM, so overwriting really happens).

    tensorboard --logdir <logs root>/tbruns --port 6006   # talq.paths.TB_ROOT
"""
import os
import time

_WARNED = False


def _mk(dirpath):
    """One SummaryWriter. On failure give None and warn only once."""
    global _WARNED
    try:
        from torch.utils.tensorboard import SummaryWriter
        os.makedirs(dirpath, exist_ok=True)
        return SummaryWriter(dirpath)
    except Exception as e:                      # missing module/disk/permission all included
        if not _WARNED:
            _WARNED = True
            print(f"[tb] logging off ({type(e).__name__}: {e}) -- "
                  f"pip install tensorboard turns it on", flush=True)
        return None


class TB:
    """None-safe wrapper. `TB.off()` is an instance that does nothing."""

    def __init__(self, root=None, tag=None):
        self.w = None
        self.tag = tag or ""
        if root and tag:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            self.w = _mk(os.path.join(root, tag, stamp))
            if self.w:
                print(f"[tb] {os.path.join(root, tag, stamp)}", flush=True)

    @classmethod
    def off(cls):
        return cls()

    def scalar(self, k, v, step):
        if self.w is None:
            return
        try:
            self.w.add_scalar(k, float(v), step)
        except Exception:
            self.w = None                       # once it fails, give up on that run

    def text(self, k, s, step=0):
        if self.w is None:
            return
        try:
            self.w.add_text(k, str(s), step)
        except Exception:
            self.w = None

    def flush(self):
        if self.w is not None:
            try:
                self.w.flush()
            except Exception:
                self.w = None

    def close(self):
        if self.w is not None:
            try:
                self.w.close()
            except Exception:
                pass
            self.w = None


def run_tag(out_dir, backbone, task, er_fold=None):
    """The result path as-is for the run name. The path relative to results/ is
    already (sweep/backbone/lambda), so TensorBoard's left tree matches the
    sweep structure."""
    p = os.path.abspath(out_dir).replace("\\", "/")
    if "/results/" in p:
        p = p.split("/results/", 1)[1]
    else:
        p = os.path.basename(p)
    t = f"{task}f{er_fold}" if task == "er" and er_fold else task
    # The cell path already contains the backbone (results/sw_gptq/wavlmL/l1). Do not put it in twice.
    return f"{p}/{t}" if backbone in p.split("/") else f"{p}/{backbone}/{t}"
