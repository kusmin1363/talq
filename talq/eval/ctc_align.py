"""CTC forced alignment (Viterbi), no torchaudio needed.

Why: the content axis fails because the CTC gradient is spike-dominated --
measured effective sample size is 3-6% of frames, top 10% of frames carry
83-98% of the weight. That is structural to an alignment-free objective: blank
absorbs most frames and the loss peaks on a few. Forced alignment assigns every
frame a target label, so a frame-wise CE built on it spreads gradient over all
speech frames instead of a handful.

Standard CTC alignment lattice: the target y = [y1..yL] is expanded to
  z = [blank, y1, blank, y2, ..., yL, blank]   (length 2L+1)
and a frame can stay on the same z-index, step +1, or step +2 (the latter only
when skipping a blank between two DIFFERENT labels).
"""
import torch


def expand_targets(y, blank=0):
    """y: (L,) LongTensor -> z: (2L+1,) with blanks interleaved."""
    L = y.shape[0]
    z = y.new_full((2 * L + 1,), blank)
    z[1::2] = y
    return z


@torch.no_grad()
def forced_align(logprobs, y, blank=0):
    """Viterbi-align one utterance.

    logprobs: (T, V) log-softmax outputs of the CTC head
    y:        (L,)   target label ids (no blanks), all != blank
    returns:  (T,)   per-frame label id from z (blank included)

    Returns None when alignment is impossible (T < 2L+1 is the classic case),
    so callers can skip that utterance rather than train on a bogus path.
    """
    T, V = logprobs.shape
    z = expand_targets(y, blank)
    S = z.shape[0]
    if T < S:
        return None

    neg = torch.finfo(logprobs.dtype).min
    # alpha[s] = best score reaching z[s] at current t; bp[t, s] = previous s
    alpha = logprobs.new_full((S,), neg)
    alpha[0] = logprobs[0, z[0]]
    if S > 1:
        alpha[1] = logprobs[0, z[1]]
    bp = torch.zeros(T, S, dtype=torch.long, device=logprobs.device)

    # a +2 step is only legal into a real label whose predecessor label differs
    can_skip = torch.zeros(S, dtype=torch.bool, device=logprobs.device)
    if S > 2:
        can_skip[2:] = (z[2:] != blank) & (z[2:] != z[:-2])

    for t in range(1, T):
        stay = alpha
        step = torch.cat([alpha.new_full((1,), neg), alpha[:-1]])
        skip = torch.cat([alpha.new_full((2,), neg), alpha[:-2]])
        skip = torch.where(can_skip, skip, alpha.new_full((S,), neg))
        cand = torch.stack([stay, step, skip])          # (3, S)
        best, arg = cand.max(0)
        prev = torch.arange(S, device=logprobs.device) - arg
        bp[t] = prev
        alpha = best + logprobs[t, z]

    # must finish on the last label or the trailing blank
    end = S - 1 if alpha[S - 1] >= alpha[S - 2] else S - 2
    path = torch.zeros(T, dtype=torch.long, device=logprobs.device)
    s = end
    for t in range(T - 1, -1, -1):
        path[t] = z[s]
        s = bp[t, s]
    return path


@torch.no_grad()
def align_batch(logprobs, targets, in_lens, blank=0):
    """Batched wrapper.

    logprobs: (B, T, V) log-softmax
    targets:  list of (L_b,) LongTensors
    in_lens:  (B,) valid frame counts
    returns:  labels (B, T) long, mask (B, T) bool -- mask is False on padding
              and on utterances that could not be aligned.
    """
    B, T, _ = logprobs.shape
    labels = torch.zeros(B, T, dtype=torch.long, device=logprobs.device)
    mask = torch.zeros(B, T, dtype=torch.bool, device=logprobs.device)
    for b in range(B):
        n = int(in_lens[b])
        p = forced_align(logprobs[b, :n], targets[b].to(logprobs.device), blank)
        if p is None:
            continue
        labels[b, :n] = p
        mask[b, :n] = True
    return labels, mask
