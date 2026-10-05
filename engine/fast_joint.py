"""Fused TDT joint + argmax for NeMo's greedy label-looping decoder (parakeet-tdt), in Triton.

Hypothesis: the decoder (about 16% of a test-clean pass on GB10 after the encoder work) is latency-bound: every joint
evaluation is about seven small kernels (gather of the current encoder frame, add, ReLU, the 640 -> 8198 GEMM writing
[B, 8198] logits, max over the vocabulary, argmax over the durations, duration lookup), run hundreds of times per
batch inside NeMo's CUDA graph. Here one kernel gathers, adds, applies ReLU, multiplies and keeps a per-tile max/argmax
(logits rounded to bf16 exactly as torch's bf16 linear rounds them, ties to the lowest index as torch.max does); a
second tiny kernel reduces the tiles to (score, label) and the duration index. No logits tensor is written.
Same algorithm otherwise: install() replaces only the joint part of GreedyBatchedTDTLabelLoopingComputer's two graph
bodies. Not bitwise identical (fp32 summation order differs from cuBLAS), so it is checked by WER and by
hypothesis agreement. Ablate by not calling install(). Requires: no fusion models, no alignments or confidence.
FAST_JOINT_FP8=1 (an accuracy trade, checked by WER): the joint's output weights in FP8 e4m3 with one scale per output
column, dequantised to bf16 in registers; halves the 10.5 MB weight read per joint evaluation.
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

BN = 128
BM = 64


@triton.jit
def _joint_tiles(E, BIDX, SIDX, G, W, WS, BIAS, PV, PI, DL, B, T, N, NL, sEb, sEt, sGb,
                 K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, FP8: tl.constexpr):
    pn = tl.program_id(0)
    pm = tl.program_id(1)
    rows = pm * BM + tl.arange(0, BM)
    rmask = rows < B
    cols = pn * BN + tl.arange(0, BN)
    cmask = cols < N
    b = tl.load(BIDX + rows, mask=rmask, other=0)
    t = tl.load(SIDX + rows, mask=rmask, other=0)
    acc = tl.zeros([BM, BN], tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + tl.arange(0, BK)
        e = tl.load(E + b[:, None] * sEb + t[:, None] * sEt + kk[None, :], mask=rmask[:, None], other=0.0)
        g = tl.load(G + rows[:, None] * sGb + kk[None, :], mask=rmask[:, None], other=0.0)
        h = (e.to(tl.float32) + g.to(tl.float32)).to(tl.bfloat16)          # torch: bf16 add (rounded)
        h = tl.maximum(h.to(tl.float32), 0.0).to(tl.bfloat16)               # ReLU (exact in bf16)
        w = tl.load(W + cols[None, :] * K + kk[:, None], mask=cmask[None, :], other=0.0)   # [BK, BN] of W [N, K]
        if FP8:
            w = w.to(tl.bfloat16)                                            # dequantised (scale applied below)
        acc = tl.dot(h, w, acc)
    if FP8:
        acc = acc * tl.load(WS + cols, mask=cmask, other=1.0)[None, :]
    bias = tl.load(BIAS + cols, mask=cmask, other=0.0).to(tl.float32)
    logit = (acc + bias[None, :]).to(tl.bfloat16).to(tl.float32)            # torch: bf16 linear output
    lab = cols < NL
    v = tl.where(lab[None, :] & cmask[None, :], logit, float("-inf"))
    vmax = tl.max(v, 1)
    big = 2147483647
    idx = tl.min(tl.where(v == vmax[:, None], cols[None, :], big), 1)      # first index of the max
    tl.store(PV + pn * B + rows, vmax, mask=rmask)
    tl.store(PI + pn * B + rows, idx, mask=rmask)
    # duration logits (columns NL .. N-1) go to a small [B, N - NL] buffer
    dmask = (cols >= NL) & cmask
    tl.store(DL + rows[:, None] * (N - NL) + (cols - NL)[None, :], logit, mask=rmask[:, None] & dmask[None, :])


@triton.jit
def _joint_reduce(PV, PI, DL, SCORES, LABELS, DIDX, B, NP, ND: tl.constexpr, BR: tl.constexpr, BP: tl.constexpr):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    rmask = rows < B
    ps = tl.arange(0, BP)
    pm = ps < NP
    v = tl.load(PV + ps[None, :] * B + rows[:, None], mask=pm[None, :] & rmask[:, None], other=float("-inf"))
    i = tl.load(PI + ps[None, :] * B + rows[:, None], mask=pm[None, :] & rmask[:, None], other=2147483647)
    vmax = tl.max(v, 1)
    lab = tl.min(tl.where(v == vmax[:, None], i, 2147483647), 1)
    tl.store(SCORES + rows, vmax.to(tl.bfloat16), mask=rmask)
    tl.store(LABELS + rows, lab.to(tl.int64), mask=rmask)
    dd = tl.arange(0, 8)
    dmask = dd < ND
    d = tl.load(DL + rows[:, None] * ND + dd[None, :], mask=rmask[:, None] & dmask[None, :], other=float("-inf"))
    dmax = tl.max(d, 1)
    didx = tl.min(tl.where(d == dmax[:, None], dd[None, :], 64), 1)
    tl.store(DIDX + rows, didx.to(tl.int64), mask=rmask)


class FusedJoint:
    """Per decoding computer: preallocated partial buffers; call writes scores/labels/duration indices in place."""

    def __init__(self, linear: torch.nn.Linear, num_durations: int):
        self.fp8 = os.environ.get("FAST_JOINT_FP8", "0") == "1"
        w = linear.weight.detach()
        if self.fp8:                                                          # per-output-column scale
            self.WS = (w.float().abs().amax(1) / 448.0).clamp(min=1e-12).contiguous()
            self.W = (w.float() / self.WS[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn).contiguous()
        else:
            self.WS = torch.ones(w.shape[0], device=w.device)
            self.W = w.contiguous()                                           # [N, K] bf16
        self.bias = linear.bias.detach().contiguous()
        self.N, self.K = self.W.shape
        self.ND = num_durations
        self.NL = self.N - num_durations
        self.NP = triton.cdiv(self.N, BN)
        self.buf = {}

    def __call__(self, E, bidx, sidx, G, scores, labels, didx):
        B = bidx.shape[0]
        if B not in self.buf:
            dev = E.device
            self.buf[B] = (torch.empty(self.NP, B, device=dev, dtype=torch.float32),
                           torch.empty(self.NP, B, device=dev, dtype=torch.int32),
                           torch.empty(B, self.ND, device=dev, dtype=torch.float32))
        pv, pi, dl = self.buf[B]
        G2 = G.reshape(G.shape[0], -1)
        _joint_tiles[(self.NP, triton.cdiv(B, BM))](
            E, bidx, sidx, G2, self.W, self.WS, self.bias, pv, pi, dl, B, E.shape[1], self.N, self.NL,
            E.stride(0), E.stride(1), G2.stride(0), K=self.K, BM=BM, BN=BN, BK=64, FP8=self.fp8, num_warps=4,
            num_stages=2)
        _joint_reduce[(triton.cdiv(B, 64),)](pv, pi, dl, scores, labels, didx, B, self.NP, ND=self.ND, BR=64,
                                             BP=triton.next_power_of_2(self.NP), num_warps=4)


def install() -> None:
    """Patch NeMo's TDT label-looping graph bodies to use the fused joint."""
    from nemo.collections.asr.parts.submodules.transducer_decoding import tdt_label_looping as tl_mod
    C = tl_mod.GreedyBatchedTDTLabelLoopingComputer
    if getattr(C, "_fast_joint_installed", False):
        return

    def fj(self):
        if getattr(self, "_fj", None) is None:
            assert self.fusion_models is None and self.state.alignments is None
            self._fj = FusedJoint(self.joint.joint_net[-1], self.state.model_durations.shape[0])
        if getattr(self, "_fj_shape", None) != tuple(self.state.labels.shape):   # NeMo re-allocates its state
            self._fj_didx = torch.empty_like(self.state.labels)
            self._fj_scores = torch.empty_like(self.state.scores)
            self._fj_labels = torch.empty_like(self.state.labels)
            self._fj_shape = tuple(self.state.labels.shape)
        return self._fj

    def before_inner(self):
        st = self.state
        st.active_mask_prev.copy_(st.active_mask)
        fj(self)(st.encoder_output_projected, st.batch_indices, st.safe_time_indices, st.decoder_output,
                 st.scores, st.labels, self._fj_didx)
        st.durations.copy_(st.model_durations[self._fj_didx])
        torch.eq(st.labels, self._blank_index, out=st.blank_mask)
        st.time_indices_current_labels.copy_(st.time_indices)
        st.durations.masked_fill_(torch.logical_and(st.durations == 0, st.blank_mask), 1)
        st.time_indices.add_(st.durations * st.active_mask)
        torch.minimum(st.time_indices, st.last_timesteps, out=st.safe_time_indices)
        torch.less(st.time_indices, st.encoder_output_length, out=st.active_mask)
        torch.logical_and(st.active_mask, st.blank_mask, out=st.advance_mask)
        torch.any(st.advance_mask, out=st.advance_mask_any)

    def inner_step(self):
        st = self.state
        torch.where(st.advance_mask, st.time_indices, st.time_indices_current_labels, out=st.time_indices_current_labels)
        fj(self)(st.encoder_output_projected, st.batch_indices, st.safe_time_indices, st.decoder_output,
                 self._fj_scores, self._fj_labels, self._fj_didx)
        more_durations = st.model_durations[self._fj_didx]
        torch.where(st.advance_mask, self._fj_labels, st.labels, out=st.labels)
        torch.where(st.advance_mask, self._fj_scores, st.scores, out=st.scores)
        torch.eq(st.labels, self._blank_index, out=st.blank_mask)
        more_durations.masked_fill_(torch.logical_and(more_durations == 0, st.blank_mask), 1)
        torch.where(st.advance_mask, st.time_indices + more_durations, st.time_indices, out=st.time_indices)
        torch.where(st.advance_mask, more_durations, st.durations, out=st.durations)
        torch.minimum(st.time_indices, st.last_timesteps, out=st.safe_time_indices)
        torch.less(st.time_indices, st.encoder_output_length, out=st.active_mask)
        torch.logical_and(st.active_mask, st.blank_mask, out=st.advance_mask)
        torch.any(st.advance_mask, out=st.advance_mask_any)

    C._before_inner_loop_get_joint_output = before_inner
    C._inner_loop_step_find_next_non_blank = inner_step
    C._fast_joint_installed = True
