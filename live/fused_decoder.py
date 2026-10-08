# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Nigel Cannings. Part of fastconformer-trt (see LICENSE and NOTICE).
# Reimplements NVIDIA NeMo's (Apache-2.0) greedy label-looping RNN-T decoding semantics (NeMo 3.1.0).
"""Fused live RNN-T greedy decoder (custom arm, --decoder fused): our own fixed-shape CUDA-graph decoder, replacing
NeMo's label-looping computer underneath the lean decoder's slot-slab interface.

Hypothesis tested: NeMo's decoder caps live capacity in two ways that a decoder with fixed shapes avoids. (1) It
re-captures its CUDA graphs whenever a call's batch exceeds every earlier one; on an H100 that re-capture crashes with
"illegal memory access" at large batch (stock at 1,536 and 1,920 streams, ours at 3,200, pre-warm at 384 or more
slots), and on any GPU it stalls every live stream for 100 to 400 ms. (2) At large batch the decoder is a large share
of each step once the encoder is FP8 TensorRT. Here every graph is captured once at start-up, before any audio, for
fixed batch buckets up to the server's full slot count, and never again; occupancy varies through masks. Ablate:
--decoder lean (NeMo's computer, same slabs) or --decoder stock.

Semantics (NeMo 3.1.0 GreedyBatchedRNNTLabelLoopingComputer, greedy, max_symbols from the model's config, 10): per
stream, frame t starts at 0 in every call; joint(encoder frame t, prediction output) then argmax over the vocabulary
including blank; blank advances t; a non-blank label is emitted, the prediction network (embedding, 2-layer LSTM,
joint.pred projection) advances on it, and after max_symbols emissions at one frame t advances without another joint
evaluation. Carried per slot across chunks, exactly what NeMo carries: LSTM hidden and cell, the projected prediction
output. (NeMo also carries the last label and decoded length; neither changes a later decision.) A slot is reset to
NeMo's after-<SOS> state when a stream is admitted, which is what NeMo computes when it is given no previous state.

Loop form: NeMo nests a blank-skipping inner while loop inside an outer loop (CUDA conditional nodes). Here one flat
iteration does one joint evaluation for every row, then the masked prediction-network update for rows that emitted.
Each row needs T_row + U_row - forced iterations; rows that have finished are masked. A call's graph (gather from the
slabs, encoder-side projection in one GEMM, I1 iterations, scatter back) is followed by a host read of the number of
unfinished rows. Rows still unfinished (a burst of tokens, up to max_symbols per frame) continue as a new call of only
those rows, from their current frame and symbol count, on the smallest bucket that holds them; the rest of the batch
is not iterated again. Exact for any token count, no silent truncation. Iterations and continuations are logged
(stats()).

Graphs: one per batch bucket (powers of two from 8, and the full slot count), captured with capture_begin/capture_end
(not the torch.cuda.graph context, whose empty_cache() broke NeMo's graphs on 5 Oct) into one shared private pool;
graphs never run concurrently and every value that outlives a replay lives in buffers allocated outside capture. A
call's rows are gathered from the slot slabs by an index vector padded with a dummy slot (row N), so a bucket's cost
follows the call's size, not the slot count.

Joint (FUSED_JOINT): torch (the model's own float32 modules, TF32 as configured, logits materialised, torch.max; the
equivalence configuration) or triton (one kernel: frame gather, add, ReLU, bf16 GEMM against the 640 -> 13,088 output
layer, per-tile max and argmax with ties to the lowest index, then a tiny reduction; no logits tensor; adapted from
fastconformer-trt/engine/fast_joint.py, TDT durations removed). Prediction network (FUSED_LSTM): cudnn (the model's
own LSTM module) or triton (layer 1's input GEMM and both biases folded into a per-label table, so layer 1 is one
640-deep GEMM; one kernel per layer: bf16 GEMM with float32 accumulation, cell in float32, masked update; h double
buffered). The triton paths are not bitwise identical to float32 and are gated by WER.
Framework choice: plain PyTorch plus Triton.
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

if "TRITON_CACHE_DIR" not in os.environ and os.path.isdir(os.environ.get("LIVE_OUT", "/out")):
    # keep compiled kernels across fresh-process trials (the container's HOME is temporary)
    os.environ["TRITON_CACHE_DIR"] = os.path.join(os.environ.get("LIVE_OUT", "/out"), "triton_cache")

try:
    import triton
    import triton.language as tl
except ImportError:                                  # torch mode does not need it
    triton = None

BN, BM = 128, 64

if triton is not None:
    @triton.jit
    def _rnnt_joint_tiles(Fp, SIDX, G, W, BIAS, PV, PI, B, N, sFb, sFt, sGb,
                          K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pm = tl.program_id(0)                        # row tile fastest: concurrent programs share a weight tile
        pn = tl.program_id(1)
        rows = pm * BM + tl.arange(0, BM)
        rmask = rows < B
        cols = pn * BN + tl.arange(0, BN)
        cmask = cols < N
        t = tl.load(SIDX + rows, mask=rmask, other=0)
        acc = tl.zeros([BM, BN], tl.float32)
        for k0 in range(0, K, BK):
            kk = k0 + tl.arange(0, BK)
            e = tl.load(Fp + rows[:, None] * sFb + t[:, None] * sFt + kk[None, :], mask=rmask[:, None], other=0.0)
            g = tl.load(G + rows[:, None] * sGb + kk[None, :], mask=rmask[:, None], other=0.0)
            h = tl.maximum(e.to(tl.float32) + g.to(tl.float32), 0.0).to(tl.bfloat16)
            w = tl.load(W + cols[None, :] * K + kk[:, None], mask=cmask[None, :], other=0.0)   # [BK, BN] of W [N, K]
            acc = tl.dot(h, w, acc)
        bias = tl.load(BIAS + cols, mask=cmask, other=0.0).to(tl.float32)
        v = tl.where(cmask[None, :], acc + bias[None, :], float("-inf"))
        vmax = tl.max(v, 1)
        idx = tl.min(tl.where(v == vmax[:, None], cols[None, :], 2147483647), 1)   # first index of the max
        tl.store(PV + pn * B + rows, vmax, mask=rmask)
        tl.store(PI + pn * B + rows, idx, mask=rmask)

    @triton.jit
    def _rnnt_joint_reduce(PV, PI, LABELS, B, NP, BR: tl.constexpr, BP: tl.constexpr):
        rows = tl.program_id(0) * BR + tl.arange(0, BR)
        rmask = rows < B
        ps = tl.arange(0, BP)
        pm = ps < NP
        v = tl.load(PV + ps[None, :] * B + rows[:, None], mask=pm[None, :] & rmask[:, None], other=float("-inf"))
        i = tl.load(PI + ps[None, :] * B + rows[:, None], mask=pm[None, :] & rmask[:, None], other=2147483647)
        vmax = tl.max(v, 1)
        lab = tl.min(tl.where(v == vmax[:, None], i, 2147483647), 1)
        tl.store(LABELS + rows, lab.to(tl.int64), mask=rmask)


    @triton.jit
    def _lstm_cell(X1, X2, W, PRE, LAB, HOLD, C, HOUT, EMIT, B, H, K1: tl.constexpr, K2: tl.constexpr,
                   TABLE: tl.constexpr, BM: tl.constexpr, BJ: tl.constexpr, BK: tl.constexpr):
        """one LSTM layer step for rows [B] and hidden units [BJ]: gates = [X1 | X2] @ W^T + PRE (bf16 operands,
        float32 accumulation; PRE is the per-label table row for layer 1 (input GEMM and both biases folded) or the
        summed bias), cell in float32. Rows with EMIT write the new h (to HOUT) and c (in place); others keep HOLD/C."""
        pm = tl.program_id(0)
        pj = tl.program_id(1)
        rows = pm * BM + tl.arange(0, BM)
        rmask = rows < B
        j = pj * BJ + tl.arange(0, BJ)
        jmask = j < H
        KT: tl.constexpr = K1 + K2
        ai = tl.zeros([BM, BJ], tl.float32)
        af = tl.zeros([BM, BJ], tl.float32)
        ag = tl.zeros([BM, BJ], tl.float32)
        ao = tl.zeros([BM, BJ], tl.float32)
        for k0 in tl.static_range(0, KT, BK):
            kk = k0 + tl.arange(0, BK)
            if k0 < K1:
                x = tl.load(X1 + rows[:, None] * K1 + kk[None, :], mask=rmask[:, None], other=0.0)
            else:
                x = tl.load(X2 + rows[:, None] * K2 + (kk - K1)[None, :], mask=rmask[:, None], other=0.0)
            x = x.to(tl.bfloat16)
            wb = W + kk[:, None]
            ai = tl.dot(x, tl.load(wb + (0 * H + j)[None, :] * KT, mask=jmask[None, :], other=0.0), ai)
            af = tl.dot(x, tl.load(wb + (1 * H + j)[None, :] * KT, mask=jmask[None, :], other=0.0), af)
            ag = tl.dot(x, tl.load(wb + (2 * H + j)[None, :] * KT, mask=jmask[None, :], other=0.0), ag)
            ao = tl.dot(x, tl.load(wb + (3 * H + j)[None, :] * KT, mask=jmask[None, :], other=0.0), ao)
        m2 = rmask[:, None] & jmask[None, :]
        if TABLE:
            lab = tl.load(LAB + rows, mask=rmask, other=0)
            pb = PRE + lab[:, None] * (4 * H)
        else:
            pb = PRE + rows[:, None] * 0
        ai += tl.load(pb + (0 * H + j)[None, :], mask=m2, other=0.0)
        af += tl.load(pb + (1 * H + j)[None, :], mask=m2, other=0.0)
        ag += tl.load(pb + (2 * H + j)[None, :], mask=m2, other=0.0)
        ao += tl.load(pb + (3 * H + j)[None, :], mask=m2, other=0.0)
        c_old = tl.load(C + rows[:, None] * H + j[None, :], mask=m2, other=0.0)
        h_old = tl.load(HOLD + rows[:, None] * H + j[None, :], mask=m2, other=0.0)
        c_new = tl.sigmoid(af) * c_old + tl.sigmoid(ai) * (2.0 * tl.sigmoid(2.0 * ag) - 1.0)
        h_new = tl.sigmoid(ao) * (2.0 * tl.sigmoid(2.0 * c_new) - 1.0)
        e = tl.load(EMIT + rows, mask=rmask, other=0) != 0
        tl.store(C + rows[:, None] * H + j[None, :], tl.where(e[:, None], c_new, c_old), mask=m2)
        tl.store(HOUT + rows[:, None] * H + j[None, :], tl.where(e[:, None], h_new, h_old), mask=m2)


def default_iters(t_cap: int) -> int:
    """iterations per graph replay (env FUSED_I1 overrides): T frames plus room for about one token per frame."""
    return int(os.environ.get("FUSED_I1", "0")) or 2 * t_cap + 2


class FusedDecoder:
    def __init__(self, model, n_slots: int, max_tokens: int, t_cap: int, joint: str | None = None,
                 lstm: str | None = None):
        comp = getattr(model.decoding.decoding, "decoding_computer", None)
        if comp is not None:
            if comp.has_fusion_models() or getattr(comp, "per_stream_biasing_enabled", False):
                raise NotImplementedError("fusion models / biasing are not supported by the fused decoder")
        self.m = model
        self.joint_mode = joint or os.environ.get("FUSED_JOINT", "torch")
        if self.joint_mode not in ("torch", "triton"):
            raise ValueError(f"FUSED_JOINT={self.joint_mode}")
        if self.joint_mode == "triton" and triton is None:
            raise RuntimeError("triton joint requested but triton is not importable")
        self.lstm_mode = lstm or os.environ.get("FUSED_LSTM", "cudnn")
        if self.lstm_mode not in ("cudnn", "triton"):
            raise ValueError(f"FUSED_LSTM={self.lstm_mode}")
        if self.lstm_mode == "triton" and triton is None:
            raise RuntimeError("triton LSTM requested but triton is not importable")
        dec, jn = model.decoder, model.joint
        self.blank = int(model.decoding.blank_id)
        self.max_symbols = int(model.decoding.decoding.max_symbols)
        if not isinstance(jn.joint_net[0], torch.nn.ReLU) or not isinstance(jn.joint_net[-1], torch.nn.Linear):
            raise NotImplementedError(f"unexpected joint net {jn.joint_net}")
        if jn.log_softmax:
            raise NotImplementedError("joint log_softmax is not supported (argmax is unchanged by it, but check)")
        self.embed = dec.prediction["embed"]
        self.lstm = dec.prediction["dec_rnn"]
        self.out = jn.joint_net[-1]
        self.n_slots, self.cap, self.T = n_slots, max_tokens, t_cap
        self.i1 = default_iters(t_cap)
        if self.lstm_mode == "triton":            # h is double-buffered: an even count per graph ends in buffer 0
            self.i1 += self.i1 % 2
        N, dummy = n_slots, n_slots
        self.dummy = dummy
        self.buckets = []
        b = 8
        while b < N:
            self.buckets.append(b)
            b *= 2
        self.buckets.append(N)
        self.n_calls = 0
        self.n_continuations = 0
        self.rows_continued = 0
        self.iters_hist: dict[int, int] = {}
        dev = "cuda"
        with torch.inference_mode():
            # after-<SOS> state, exactly as NeMo builds it (predict on <SOS> = blank with no state, then project)
            sos = torch.full([1, 1], comp._SOS if comp is not None else self.blank,
                             dtype=torch.long, device=dev)
            g0, (h0, c0) = dec.predict(sos, None, add_sos=False, batch_size=1)[:2]
            g0 = jn.project_prednet(g0)[:, 0]                                       # [1, Hj]
            self.h0, self.c0, self.g0 = h0[:, 0].clone(), c0[:, 0].clone(), g0[0].clone()
            L, H = h0.shape[0], h0.shape[2]
            Hj, D = self.g0.shape[0], jn.enc.in_features
            self.L, self.H, self.Hj, self.D = L, H, Hj, D
            # slot slabs (N real slots + one dummy row for padding)
            self.sh = self.h0[:, None].repeat(1, N + 1, 1).contiguous()
            self.sc = self.c0[:, None].repeat(1, N + 1, 1).contiguous()
            self.sg = self.g0[None].repeat(N + 1, 1).contiguous()
            self.tok = torch.zeros(N + 1, max_tokens, dtype=torch.int32, device=dev)
            self.tok_len = torch.zeros(N + 1, dtype=torch.int64, device=dev)
            # call inputs (prefix [:Bb] of each is contiguous, shared by every bucket)
            self.idx = torch.full((N,), dummy, dtype=torch.long, device=dev)
            self.enc_in = torch.zeros(N, t_cap, D, dtype=torch.float32, device=dev)
            self.len_in = torch.zeros(N, dtype=torch.long, device=dev)
            # working state
            self.f = torch.zeros(N, t_cap, Hj, dtype=torch.float32, device=dev)
            self.t = torch.zeros(N, dtype=torch.long, device=dev)
            self.t0 = torch.zeros(N, dtype=torch.long, device=dev)                  # starting frame and symbol
            self.cnt0 = torch.zeros(N, dtype=torch.long, device=dev)                # count (continuations)
            self.cnt = torch.zeros(N, dtype=torch.long, device=dev)
            self.ntok = torch.zeros(N, dtype=torch.long, device=dev)
            self.g = torch.zeros(N, Hj, dtype=torch.float32, device=dev)
            self.wh = {bb: torch.zeros(L, bb, H, device=dev) for bb in self.buckets}
            self.wc = {bb: torch.zeros(L, bb, H, device=dev) for bb in self.buckets}
            if self.lstm_mode == "triton":
                lm = self.lstm.lstm
                if not isinstance(lm, torch.nn.LSTM) or lm.num_layers != 2 or lm.proj_size or lm.bidirectional:
                    raise NotImplementedError(f"triton LSTM expects a 2-layer unidirectional LSTM, got {lm}")
                # layer 1 input GEMM and both biases folded into a per-label table (blank = padding row = zeros)
                self.t1 = (self.embed.weight.float() @ lm.weight_ih_l0.float().t() + lm.bias_ih_l0.float()
                           + lm.bias_hh_l0.float()).contiguous()                        # [V, 4H]
                self.w1 = lm.weight_hh_l0.detach().to(torch.bfloat16).contiguous()          # [4H, H]
                self.w2 = torch.cat([lm.weight_ih_l1, lm.weight_hh_l1], 1).detach().to(torch.bfloat16).contiguous()
                self.b2 = (lm.bias_ih_l1.float() + lm.bias_hh_l1.float()).contiguous()
                # tile config (FUSED_LSTM_CFG="BM,BJ,BK,warps,stages"); BJ hidden units = 4 BJ gate columns
                self.lcfg = tuple(int(x) for x in os.environ.get("FUSED_LSTM_CFG", "64,64,64,4,2").split(","))
                self.wh2 = {bb: torch.zeros(L, bb, H, device=dev) for bb in self.buckets}
                self.emit_i = torch.zeros(N, dtype=torch.int32, device=dev)
                self.lab_e = torch.zeros(N, dtype=torch.long, device=dev)
            self._par = 0
            self.flag = torch.zeros(2, dtype=torch.long, device=dev)    # [rows unfinished, iterations with work]
            self.flag_host = torch.zeros(2, dtype=torch.long).pin_memory()
            if self.joint_mode == "triton":
                self.W = self.out.weight.detach().to(torch.bfloat16).contiguous()  # [V, Hj]
                self.bias = self.out.bias.detach().float().contiguous()
                self.V = self.W.shape[0]
                # tile config (FUSED_JOINT_CFG="BM,BN,BK,warps,stages"; default 64,128,64,4,2 as the offline kernel)
                self.jcfg = tuple(int(x) for x in os.environ.get("FUSED_JOINT_CFG", f"{BM},{BN},64,4,2").split(","))
                self.NP = triton.cdiv(self.V, self.jcfg[1])
                self.pv = torch.empty(self.NP * N, device=dev, dtype=torch.float32)
                self.pi = torch.empty(self.NP * N, device=dev, dtype=torch.int32)
                self.lab = torch.empty(N, device=dev, dtype=torch.int64)
        self.graphs = {}
        self._capture()

    # ---- graph bodies (all persistent values in buffers allocated above) ----
    def _joint_labels(self, bb: int, safe: torch.Tensor) -> torch.Tensor:
        f, g = self.f[:bb], self.g[:bb]
        if self.joint_mode == "torch":
            rows = torch.arange(bb, device="cuda")
            logits = self.m.joint.joint_after_projection(f[rows, safe].unsqueeze(1), g.unsqueeze(1)).squeeze(1).squeeze(1)
            return logits.max(-1)[1]
        lab = self.lab[:bb]
        bm, bn, bk, nw, ns = self.jcfg
        _rnnt_joint_tiles[(triton.cdiv(bb, bm), self.NP)](
            f, safe, g, self.W, self.bias, self.pv, self.pi, bb, self.V, f.stride(0), f.stride(1), g.stride(0),
            K=self.Hj, BM=bm, BN=bn, BK=bk, num_warps=nw, num_stages=ns)
        _rnnt_joint_reduce[(triton.cdiv(bb, 64),)](self.pv, self.pi, lab, bb, self.NP, BR=64,
                                                   BP=triton.next_power_of_2(self.NP), num_warps=4)
        return lab

    def _prologue(self, bb: int) -> None:
        idx = self.idx[:bb]
        self.wh[bb].copy_(self.sh.index_select(1, idx))
        self.wc[bb].copy_(self.sc.index_select(1, idx))
        self.g[:bb].copy_(self.sg.index_select(0, idx))
        self.ntok[:bb].copy_(self.tok_len.index_select(0, idx))
        self.f[:bb].copy_(self.m.joint.project_encoder(self.enc_in[:bb]))
        self.t[:bb].copy_(self.t0[:bb])
        self.cnt[:bb].copy_(self.cnt0[:bb])
        self.flag[1].zero_()

    def _iteration(self, bb: int) -> None:
        t, ln, cnt, ntok, g = self.t[:bb], self.len_in[:bb], self.cnt[:bb], self.ntok[:bb], self.g[:bb]
        h, c = self.wh[bb], self.wc[bb]
        active = t < ln
        self.flag[1].add_(active.any())
        safe = torch.minimum(t, (ln - 1).clamp(min=0))
        labels = self._joint_labels(bb, safe)
        blank = labels == self.blank
        emit = active & ~blank
        # append emitted tokens (rows that did not emit write to the dummy slot's row)
        row = torch.where(emit, self.idx[:bb], self.dummy)
        col = ntok.clamp(max=self.cap - 1)
        self.tok.index_put_((row, col), labels.to(torch.int32))
        ntok.add_(emit)
        # prediction network on the emitted labels, kept only for rows that emitted
        if self.lstm_mode == "cudnn":
            y = self.embed(torch.where(emit, labels, self.blank)).unsqueeze(0)       # [1, bb, H]
            out, (h2, c2) = self.lstm(y, (h, c))
            g2 = self.m.joint.project_prednet(out[0])
            e3 = emit[None, :, None]
            h.copy_(torch.where(e3, h2, h))
            c.copy_(torch.where(e3, c2, c))
        else:
            hin, hout = (self.wh[bb], self.wh2[bb]) if self._par == 0 else (self.wh2[bb], self.wh[bb])
            self._par ^= 1
            em, lab = self.emit_i[:bb], self.lab_e[:bb]
            em.copy_(emit)
            lab.copy_(torch.where(emit, labels, self.blank))
            H = self.H
            bm, bj, bk, nw, ns = self.lcfg
            grid = (triton.cdiv(bb, bm), triton.cdiv(H, bj))
            _lstm_cell[grid](hin[0], hin[0], self.w1, self.t1, lab, hin[0], c[0], hout[0], em, bb, H,
                             K1=H, K2=0, TABLE=True, BM=bm, BJ=bj, BK=bk, num_warps=nw, num_stages=ns)
            _lstm_cell[grid](hout[0], hin[1], self.w2, self.b2, lab, hin[1], c[1], hout[1], em, bb, H,
                             K1=H, K2=H, TABLE=False, BM=bm, BJ=bj, BK=bk, num_warps=nw, num_stages=ns)
            g2 = self.m.joint.project_prednet(hout[1])
        g.copy_(torch.where(emit[:, None], g2, g))
        # time: blank advances; max_symbols emissions at one frame force an advance
        cnt.add_(emit)
        adv = (active & blank) | (emit & (cnt >= self.max_symbols))
        t.add_(adv)
        cnt.masked_fill_(adv, 0)

    def _epilogue(self, bb: int) -> None:
        if self._par != 0:
            raise RuntimeError("h double buffer out of phase")
        idx = self.idx[:bb]
        self.sh.index_copy_(1, idx, self.wh[bb])
        self.sc.index_copy_(1, idx, self.wc[bb])
        self.sg.index_copy_(0, idx, self.g[:bb])
        self.tok_len.index_copy_(0, idx, self.ntok[:bb])
        self.flag[0].copy_((self.t[:bb] < self.len_in[:bb]).sum())
        self.flag_host.copy_(self.flag, non_blocking=True)

    def _first(self, bb: int) -> None:
        self._par = 0
        self._prologue(bb)
        for _ in range(self.i1):
            self._iteration(bb)
        self._epilogue(bb)

    def _capture(self) -> None:
        """capture every graph once, at start-up. Warm-up runs eagerly (all rows masked: len 0, dummy slot) so
        cuBLAS / cuDNN / Triton initialise outside capture; slabs are re-initialised afterwards."""
        pool = torch.cuda.graph_pool_handle()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.inference_mode():
            self.idx.fill_(self.dummy)
            self.len_in.zero_()
            self.t0.zero_()
            self.cnt0.zero_()
            for bb in self.buckets:
                for _ in range(2):
                    self._first(bb)
            torch.cuda.synchronize()
            for bb in self.buckets:
                gr = torch.cuda.CUDAGraph()
                gr.capture_begin(pool=pool)
                self._first(bb)
                gr.capture_end()
                self.graphs[bb] = gr
            for bb in self.buckets:                # replay once each (all rows masked)
                self.graphs[bb].replay()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        with torch.inference_mode():
            self.sh.copy_(self.h0[:, None].expand_as(self.sh))
            self.sc.copy_(self.c0[:, None].expand_as(self.sc))
            self.sg.copy_(self.g0[None].expand_as(self.sg))
            self.tok_len.zero_()
            self.tok.zero_()

    # ---- server interface (same as LeanDecoder) ----
    def reset(self, slot: int) -> None:
        with torch.inference_mode():
            self.sh[:, slot] = self.h0
            self.sc[:, slot] = self.c0
            self.sg[slot] = self.g0
            self.tok_len[slot] = 0

    def step(self, enc: torch.Tensor, enc_len: torch.Tensor, slots: torch.Tensor, first: bool) -> None:
        """enc [B, D, T] (after the language prompt), enc_len [B], slots [B]; `first` is not needed (slots are reset
        to the after-<SOS> state on admission)."""
        B, D, T = enc.shape
        if T > self.T:
            raise RuntimeError(f"encoder chunk has {T} frames, fused decoder captured for {self.T}")
        with torch.inference_mode():
            self.enc_in[:B, :T].copy_(enc.transpose(1, 2))
            self._launch(slots, enc_len, None, None)
            it = int(self.flag_host[1])
            # continuation: rows still unfinished after i1 iterations (a long burst of tokens, up to max_symbols per
            # frame) continue as a new, compacted call of only those rows, from their current frame and symbol
            # count; their state is already in the slabs. Exact, and the rest of the batch is not iterated again.
            while int(self.flag_host[0]):
                bb = self._bb
                rows = (self.t[:bb] < self.len_in[:bb]).nonzero()[:, 0]
                k = rows.numel()
                sub = (self.idx[rows].clone(), self.enc_in[rows].clone(), self.len_in[rows].clone(),
                       self.t[rows].clone(), self.cnt[rows].clone())
                self.enc_in[:k].copy_(sub[1])
                self._launch(sub[0], sub[2], sub[3], sub[4])
                self.n_continuations += 1
                self.rows_continued += k
                it += int(self.flag_host[1])
            self.n_calls += 1
            self.iters_hist[it] = self.iters_hist.get(it, 0) + 1

    def _launch(self, slots, lens, t0, cnt0) -> None:
        """replay the graph of the smallest bucket that holds these rows (inputs other than enc_in set here)."""
        B = slots.numel()
        bb = next((x for x in self.buckets if x >= B), None)
        if bb is None:
            raise RuntimeError(f"call of {B} rows exceeds the {self.n_slots} slots captured at start-up")
        self.idx[:B].copy_(slots)
        self.len_in[:B].copy_(lens)
        if bb > B:
            self.idx[B:bb].fill_(self.dummy)
            self.len_in[B:bb].zero_()
        if t0 is None:
            self.t0[:bb].zero_()
            self.cnt0[:bb].zero_()
        else:
            self.t0[:B].copy_(t0)
            self.cnt0[:B].copy_(cnt0)
            self.t0[B:bb].zero_()
            self.cnt0[B:bb].zero_()
        self._bb = bb
        self.graphs[bb].replay()
        torch.cuda.current_stream().synchronize()

    def text(self, slot: int) -> tuple[str, list[int]]:
        n = int(self.tok_len[slot])
        if n > self.cap:
            raise RuntimeError(f"token slab overflow: stream in slot {slot} has {n} tokens, capacity {self.cap}")
        ids = [t for t in self.tok[slot, :n].tolist() if t != self.blank]
        return self.m.decoding.decode_tokens_to_str_with_strip_punctuation(ids), ids

    def stats(self) -> dict:
        return {"joint": self.joint_mode, "lstm": self.lstm_mode, "buckets": self.buckets, "i1": self.i1, "t_cap": self.T,
                "max_symbols": self.max_symbols, "calls": self.n_calls, "continuations": self.n_continuations,
                "rows_continued": self.rows_continued,
                "graphs_captured_at_startup": len(self.graphs),
                "iterations_needed_hist": dict(sorted(self.iters_hist.items()))}
