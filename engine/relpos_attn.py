"""Fused relative-position attention for FastConformer (Transformer-XL style), in Triton.

Hypothesis: on GB10 the attention block's memory traffic, not its arithmetic, dominates. TensorRT materialises the
position scores q_v.p^T as [H, B, T, 2T-1], shifts them into [H, B, T, T], copies q/v into its attention kernel's
buffers and permutes the output back (about 33 ms per 32 x 16 s batch, against about 10 ms of attention proper).
This kernel computes, per (head, utterance, 64-query tile) and 64-key tile:
    s[i, j] = (q_u[i].k[j] + q_u[i].p[T-1-i+j] + c[T-1-i+j]) / sqrt(dk),    c = (pos_bias_v - pos_bias_u).p
with the q.p band (64 x 128) from one tensor-core dot, the relative shift done through a small per-program scratch
tile, online softmax, keys beyond the utterance's length skipped, and the output written straight to [B, T, H*dk].
Same maths as lean_encoder's heads-first path (q_v.p = q_u.p + c). Ablate by not using it (TensorRT's own path).

Run as a script: accuracy and speed against the PyTorch reference at the encoder's shapes.
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _relpos_attn(Q, K, V, P, C, LEN, O, S, T, R, B, H, scale,
                 BM: tl.constexpr, BN: tl.constexpr, DK: tl.constexpr, GATHER: tl.constexpr):
    pid_m = tl.program_id(0)
    hb = tl.program_id(1)
    h = hb // B
    b = hb % B
    W: tl.constexpr = BM + BN                                   # band width (rows of p a tile pair needs, padded)
    i0 = pid_m * BM
    ii = tl.arange(0, BM)
    jj = tl.arange(0, BN)
    ww = tl.arange(0, W)
    dd = tl.arange(0, DK)
    base = (h.to(tl.int64) * B + b) * T * DK
    q = tl.load(Q + base + (i0 + ii)[:, None] * DK + dd[None, :], mask=(i0 + ii)[:, None] < T, other=0.0)
    L = tl.load(LEN + b)
    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, DK], tl.float32)
    pbase = h.to(tl.int64) * R * DK
    sbase = S + (hb.to(tl.int64) * tl.num_programs(0) + pid_m) * BM * W
    for j0 in range(0, L, BN):
        jm = (j0 + jj) < L
        k = tl.load(K + base + (j0 + jj)[:, None] * DK + dd[None, :], mask=jm[:, None], other=0.0)
        v = tl.load(V + base + (j0 + jj)[:, None] * DK + dd[None, :], mask=jm[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k))                                # [BM, BN]
        r0 = T - 1 - (i0 + BM - 1) + j0                           # band row of (i0 + BM - 1, j0)
        rr = r0 + ww
        rm = (rr >= 0) & (rr < R)
        pb = tl.load(P + pbase + rr[:, None] * DK + dd[None, :], mask=rm[:, None], other=0.0)
        band = tl.dot(q, tl.trans(pb)) + tl.load(C + h * R + rr, mask=rm, other=0.0)[None, :]   # [BM, W]
        if GATHER:                                                # rel shift in registers
            s2 = tl.gather(band, ((BM - 1) - ii)[:, None] + jj[None, :], 1)
        else:                                                     # rel shift through a scratch tile
            tl.store(sbase + ii[:, None] * W + ww[None, :], band)
            tl.debug_barrier()
            s2 = tl.load(sbase + ii[:, None] * W + ((BM - 1) - ii)[:, None] + jj[None, :])
            tl.debug_barrier()
        s = (s + s2) * scale
        s = tl.where(jm[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
        m_i = m_new
    out = acc / l_i[:, None]
    obase = O + (b.to(tl.int64) * T) * H * DK + h * DK           # O is [B, T, H, DK]
    tl.store(obase + (i0 + ii)[:, None] * H * DK + dd[None, :], out.to(tl.float16), mask=(i0 + ii)[:, None] < T)


def relpos_attention(q_u, k, v, p, c, lengths, BM=64, BN=64, num_warps=4, num_stages=2, gather=True):
    """q_u, k, v [H, B, T, dk] fp16; p [H, 2T-1, dk] fp16; c [H, 2T-1] fp32; lengths [B] int32 -> [B, T, H*dk] fp16."""
    H, B, T, DK = q_u.shape
    R = p.shape[1]
    o = torch.empty(B, T, H * DK, device=q_u.device, dtype=torch.float16)
    grid = (triton.cdiv(T, BM), H * B)
    scratch = torch.empty(grid[0] * grid[1] * BM * (BM + BN), device=q_u.device, dtype=torch.float32)
    _relpos_attn[grid](q_u, k, v, p, c, lengths, o, scratch, T, R, B, H, 1.0 / math.sqrt(DK),
                       BM=BM, BN=BN, DK=DK, GATHER=gather, num_warps=num_warps, num_stages=num_stages)
    return o


def reference(q_u, k, v, p, c, lengths):
    """lean_encoder heads-first maths: bd = q_u.p + c, gather the relative shift, key mask, softmax."""
    H, B, T, DK = q_u.shape
    bd = torch.matmul(q_u.float().reshape(H, B * T, DK), p.float().transpose(1, 2)) + c[:, None, :]
    bd = bd.view(H, B, T, 2 * T - 1)
    ar = torch.arange(T, device=q_u.device)
    rel = (T - 1) - ar[:, None] + ar[None, :]
    bd = torch.gather(bd, 3, rel[None, None].expand(H, B, T, T))
    s = (torch.matmul(q_u.float(), k.float().transpose(-1, -2)) + bd) / math.sqrt(DK)
    s = s.masked_fill(~(ar[None, :] < lengths[:, None])[None, :, None, :], float("-inf"))
    a = torch.matmul(torch.softmax(s, -1), v.float())                       # [H, B, T, dk]
    return a.permute(1, 2, 0, 3).reshape(B, T, H * DK)


if __name__ == "__main__":
    import time
    torch.manual_seed(0)
    for B, T in ((32, 200), (32, 400), (8, 750)):
        H, DK = 8, 128
        q = torch.randn(H, B, T, DK, device="cuda").half()
        k = torch.randn(H, B, T, DK, device="cuda").half()
        v = torch.randn(H, B, T, DK, device="cuda").half()
        p = (torch.randn(H, 2 * T - 1, DK, device="cuda") * 0.5).half()
        c = torch.randn(H, 2 * T - 1, device="cuda")
        lengths = torch.randint(T // 3, T + 1, (B,), device="cuda", dtype=torch.int32)
        lengths[0] = T
        o = relpos_attention(q, k, v, p, c, lengths)
        ref = reference(q, k, v, p, c, lengths)
        valid = (torch.arange(T, device="cuda")[None] < lengths[:, None])[..., None]
        err = float(((o.float() - ref) * valid).norm() / (ref * valid).norm())

        def bench(f, n=30):
            for _ in range(3): f()
            torch.cuda.synchronize(); t = time.time()
            for _ in range(n): f()
            torch.cuda.synchronize(); return (time.time() - t) / n * 1e3
        tr = bench(lambda: reference(q, k, v, p, c, lengths))
        print(f"B{B} T{T}: torch fp32 reference {tr:.3f} ms", flush=True)
        for g in (True, False):
            for bm, bn in ((64, 64), (32, 32), (128, 128)):
                for nw in (4, 8):
                    for ns in (1, 2, 3):
                        try:
                            o = relpos_attention(q, k, v, p, c, lengths, bm, bn, nw, ns, g)
                            e = float(((o.float() - ref) * valid).norm() / (ref * valid).norm())
                            t = bench(lambda: relpos_attention(q, k, v, p, c, lengths, bm, bn, nw, ns, g))
                            print(f"   T{T} gather={g} BM{bm} BN{bn} w{nw} s{ns}: {t:.3f} ms err {e:.1e}", flush=True)
                        except Exception as ex:
                            print(f"   gather={g} BM{bm} BN{bn} w{nw} s{ns}: FAIL {str(ex)[:80]}", flush=True)
