# Derived from s3prl (https://github.com/s3prl/s3prl)
# Copyright (c) Speech Lab, NTU, Taiwan
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at LICENSES/Apache-2.0.txt in this repository, or at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
#
# NOTE: This file is the one exception to this project's MIT LICENSE; see NOTICE.
#
# Modifications from s3prl 0.4.18, as required by Apache-2.0 section 4(b):
#   * The upstream modules were restructured into two probe classes (ASVProbe,
#     UttProbe) taking this project's constructor signatures.
#   * AMSoftmaxLoss computes its masked logsumexp by masking rather than by
#     s3prl's Python loop over classes -- numerically identical, faster.
#   * A layer-mixing weight `w` was added so every probe in this project is read
#     the same way.
#   * Frame arithmetic and initialisation rationale were documented inline.
#   Architectures and hyperparameters (s=30.0, m=0.4, agg_dim=1500, proj=256, the
#   TDNN contexts and dilations) are unchanged, so the heads remain numerically
#   equivalent to s3prl's.

"""SUPERB-spec downstream heads, faithful to s3prl's published recipes.

Why this file exists: talq.eval.probe_train's SVProbe is a hybrid -- it trains like SID
(mean-pool -> linear -> speaker CE) but is scored like ASV (cosine EER). SUPERB
keeps those apart, and its ASV head is much stronger: XVector (5 TDNN layers) +
statistics pooling + AMSoftmax. Our whole "why does speaker behave differently"
analysis rests on the sv probe, so that probe must not be the odd one out.

Ported from s3prl 0.4.18:
  downstream/sv_voxceleb1/model.py   (TDNN, XVector, SP, UtteranceExtractor,
                                      AMSoftmaxLoss, Model)   -> ASVProbe
  downstream/model.py                (UtteranceLevel + MeanPooling) -> UttProbe
Config values are from sv_voxceleb1/config.yaml and speech_commands /
fluent_commands / voxceleb1 / emotion config.yaml (projector_dim 256).

The layer-mixing weight `w` follows the same convention as talq.eval.probe_train's
LinearCTC (softmax over L+1 hidden states) so every probe in this project is
read the same way and collect/analysis code needs no special cases.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedSum(nn.Module):
    """SUPERB layer mixing: softmax(w) . hidden_states. Same as LinearCTC.fuse."""

    def __init__(self, n_layers):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n_layers))

    def fuse(self, hs):                      # hs: (L, B, T, D)
        return (hs * torch.softmax(self.w, 0)[:, None, None, None]).sum(0)


class TDNN(nn.Module):
    """Time-delay layer: Linear over a dilated context window, then ReLU.

    s3prl builds this with F.unfold on a (B,1,T,H) view; keeping that exact
    formulation matters because it also defines how many frames are consumed
    (T shrinks by (context_size-1)*dilation)."""

    def __init__(self, input_dim, output_dim, context_size, dilation,
                 batch_norm=False, dropout_p=0.0):
        super().__init__()
        self.input_dim, self.context_size, self.dilation = input_dim, context_size, dilation
        self.kernel = nn.Linear(input_dim * context_size, output_dim)
        # He initialization. PyTorch's default (kaiming_uniform, a=sqrt(5)) shrinks
        # the variance by ~1/3 per layer once 5 ReLU layers are stacked, and ReLU
        # cuts half again, so right after init the xvector output max dies down to
        # 0.1. Then any input yields the same direction, the cosine between
        # utterances becomes 1.0000 (std 3e-5), and training cannot even start.
        nn.init.kaiming_normal_(self.kernel.weight, nonlinearity="relu")
        nn.init.zeros_(self.kernel.bias)
        self.nonlinearity = nn.ReLU()
        self.bn = nn.BatchNorm1d(output_dim) if batch_norm else None
        self.drop = nn.Dropout(dropout_p) if dropout_p else None

    def forward(self, x):                    # (B, T, H) -> (B, T', out)
        assert x.shape[-1] == self.input_dim, f"{x.shape[-1]} != {self.input_dim}"
        x = F.unfold(x.unsqueeze(1), (self.context_size, self.input_dim),
                     stride=(1, self.input_dim), dilation=(self.dilation, 1))
        x = self.nonlinearity(self.kernel(x.transpose(1, 2)))
        if self.drop is not None:
            x = self.drop(x)
        if self.bn is not None:
            x = self.bn(x.transpose(1, 2)).transpose(1, 2)
        return x


class XVector(nn.Module):
    """5 TDNN layers, contexts (5,3,3,1,1) / dilations (1,2,3,1,1). Consumes 14
    frames total, so inputs shorter than that are invalid (8s audio = ~400)."""

    def __init__(self, input_dim, agg_dim=1500, dropout_p=0.0, batch_norm=False):
        super().__init__()
        cfg = [(input_dim, 5, 1), (input_dim, 3, 2), (input_dim, 3, 3),
               (input_dim, 1, 1), (agg_dim, 1, 1)]
        self.module = nn.Sequential(*[
            TDNN(input_dim, out, ctx, dil, batch_norm, dropout_p)
            for out, ctx, dil in cfg])

    def forward(self, x):
        return self.module(x)


class StatsPooling(nn.Module):
    """[mean ; std] over time -> 2*agg_dim, honouring per-utterance lengths.

    The mask is NOT optional during training. pad_batch zero-pads a batch to its
    longest clip, and pooling over the padding makes both statistics a function of
    how much padding an utterance happened to get. s3prl's SP slices to the valid
    length for exactly this reason. Passing lens=None (batch of one, no padding)
    is the fast path used at eval time."""

    def forward(self, x, lens=None):         # (B, T, H) -> (B, 2H)
        if lens is None:
            return torch.cat([x.mean(1), x.std(1)], -1)
        out = []
        for i, n in enumerate(lens):
            v = x[i, :max(int(n), 2)]        # std needs at least 2 frames
            out.append(torch.cat([v.mean(0), v.std(0)]))
        return torch.stack(out)


class UtteranceExtractor(nn.Module):
    """Linear-ReLU-Linear-ReLU for training; inference() stops after the FIRST
    ReLU. That asymmetry is s3prl's, not a bug -- the speaker embedding scored
    by cosine/EER is the 1-layer output, so we must reproduce it exactly."""

    def __init__(self, input_dim, out_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        for lin in (self.linear1, self.linear2):      # same reason as in TDNN
            nn.init.kaiming_normal_(lin.weight, nonlinearity="relu")
            nn.init.zeros_(lin.bias)
        self.act_fn = nn.ReLU()

    def forward(self, x):
        return self.act_fn(self.linear2(self.act_fn(self.linear1(x))))

    def inference(self, x):
        return self.act_fn(self.linear1(x))


class AMSoftmaxLoss(nn.Module):
    """Additive-margin softmax, s=30.0 m=0.4 (sv_voxceleb1/config.yaml)."""

    def __init__(self, hidden_dim, speaker_num, s=30.0, m=0.4):
        super().__init__()
        self.s, self.m, self.speaker_num = s, m, speaker_num
        self.W = nn.Parameter(torch.randn(hidden_dim, speaker_num))
        nn.init.xavier_normal_(self.W, gain=1)

    def forward(self, x, labels):            # (B,H), (B,)
        wf = F.normalize(x, dim=1) @ F.normalize(self.W, dim=0)
        num = self.s * (wf.gather(1, labels[:, None]).squeeze(1) - self.m)
        # logsumexp over the rest, excluding only the target logit. s3prl slices
        # it out with a Python loop, but masking is numerically identical and
        # much faster on a batch.
        excl = wf.scatter(1, labels[:, None], float("-inf"))
        den = torch.exp(num) + torch.exp(self.s * excl).sum(1)
        return -(num - torch.log(den)).mean()


class ASVProbe(WeightedSum):
    """SUPERB ASV: weighted sum -> Linear(D,512) -> XVector -> SP -> UttExtractor.
    Trained with AMSoftmax over dev speakers; scored by cosine EER on embed()."""

    def __init__(self, n_layers, dim, n_spk, hidden=512, agg_dim=1500):
        super().__init__(n_layers)
        self.connector = nn.Linear(dim, hidden)
        self.xvector = XVector(hidden, agg_dim)
        self.pool = StatsPooling()
        self.utt = UtteranceExtractor(2 * agg_dim, hidden)
        self.loss = AMSoftmaxLoss(hidden, n_spk)

    # frames consumed by XVector's 5 TDNN layers: (5-1)*1 + (3-1)*2 + (3-1)*3 = 14
    FRAMES_CONSUMED = 14

    def _trunk(self, hs, flens=None):
        x = self.xvector(self.connector(self.fuse(hs)))
        if flens is None:
            return self.pool(x)
        return self.pool(x, [n - self.FRAMES_CONSUMED for n in flens])

    def embed(self, hs, flens=None):         # embedding for evaluation (only up to linear1)
        return self.utt.inference(self._trunk(hs, flens))

    def forward(self, hs, labels, flens=None):   # for training: returns the loss directly
        return self.loss(self.utt(self._trunk(hs, flens)), labels)


class UttProbe(WeightedSum):
    """SUPERB utterance-level head for KS / IC / SID / ER:
    weighted sum -> Linear(D,256) -> mean-pool -> Linear(256,n_class).
    projector_dim=256 from speech_commands/fluent_commands/voxceleb1/emotion."""

    def __init__(self, n_layers, dim, n_class, proj=256):
        super().__init__(n_layers)
        self.projector = nn.Linear(dim, proj)
        self.head = nn.Linear(proj, n_class)

    def forward(self, hs):
        return self.head(self.projector(self.fuse(hs)).mean(1))
