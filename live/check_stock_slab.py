# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Nigel Cannings. Part of fastconformer-trt (see LICENSE and NOTICE).
"""Check of StockStep.slab_step (8 Oct: NeMo's stock encoder step piece by piece on the cache slab) against the stock
full-batch path (gather all rows, one NeMo call, scatter): float32 slabs (as shipped) of N slots with random caches,
B random slots in random order, random features, two chained steps (first chunk then steady). Compares the encoder
output, lengths and the whole slabs bit for bit, for several piece sizes, and reports the peak memory of a step.

usage: python check_stock_slab.py OUT_JSON --r 13 --slots 1500 --b 300,1024 --pieces 512,128
"""
from __future__ import annotations

import argparse
import json

import torch

import harness


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--r", type=int, default=13)
    ap.add_argument("--slots", type=int, default=1500)
    ap.add_argument("--b", default="300,1024")
    ap.add_argument("--pieces", default="512,128")
    ap.add_argument("--seed", type=int, default=20261005)
    a = ap.parse_args()
    m = harness.load_model("nvidia/nemotron-3.5-asr-streaming-0.6b", a.r, "en-US", a.seed)
    st = harness.make_encoder_step(m, "stock", None, a.r)
    cfg = m.encoder.streaming_cfg
    pick = lambda v, i: v[i] if isinstance(v, list) else v
    N, res = a.slots, []
    with torch.inference_mode():
        g = torch.Generator(device="cuda").manual_seed(a.seed)
        c0, t0, l0 = m.encoder.get_initial_cache_state(batch_size=N)
        base = (torch.randn(c0.shape, device="cuda", generator=g) * 0.5, torch.randn(t0.shape, device="cuda", generator=g) * 0.5,
                torch.randint(0, 57, (N,), device="cuda", generator=g))
        for B in [int(x) for x in a.b.split(",")]:
            slots = torch.randperm(N, device="cuda", generator=g)[:B]
            for mode in ["full"] + [f"piece{p}" for p in a.pieces.split(",")]:
                cc, ct, cl = (x.clone() for x in base)
                outs, pk = [], []
                if mode != "full":
                    st.piece = int(mode[5:])
                for k, first in enumerate((True, False)):
                    W = pick(cfg.chunk_size, 0 if first else 1) + pick(cfg.pre_encode_cache_size, 0 if first else 1)
                    drop = 0 if first else cfg.drop_extra_pre_encoded
                    sig = torch.randn(B, 128, W, device="cuda", generator=torch.Generator(device="cuda").manual_seed(a.seed + B + k))
                    lens = torch.full((B,), W, dtype=torch.int64, device="cuda")
                    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); b0 = torch.cuda.memory_allocated()
                    if mode == "full":
                        e, el, c2, t2, l2 = st(sig, lens, cc.index_select(1, slots), ct.index_select(1, slots),
                                               cl.index_select(0, slots), drop)
                        cc.index_copy_(1, slots, c2); ct.index_copy_(1, slots, t2); cl.index_copy_(0, slots, l2)
                    else:
                        e, el = st.slab_step(sig, lens, cc, ct, cl, slots, drop)
                    torch.cuda.synchronize()
                    pk.append(round((torch.cuda.max_memory_allocated() - b0) / 1e9, 2))
                    outs.append((e.clone(), el.clone()))
                if mode == "full":                      # keep only the reference (memory: one extra slab set)
                    f = (outs, cc, ct, cl, pk)
                    continue
                r_ = (outs, cc, ct, cl, pk)
                same = (all(torch.equal(x[0], y[0]) and torch.equal(x[1], y[1]) for x, y in zip(f[0], r_[0]))
                        and torch.equal(f[1], r_[1]) and torch.equal(f[2], r_[2]) and torch.equal(f[3], r_[3]))
                diff = max(float((x[0] - y[0]).abs().max()) for x, y in zip(f[0], r_[0]))
                rec = {"r": a.r, "slots": N, "B": B, "mode": mode, "bit_identical": bool(same), "max_abs_enc_diff": diff,
                       "max_abs_cache_diff": max(float((f[1] - r_[1]).abs().max()), float((f[2] - r_[2]).abs().max())),
                       "max_abs_enc": max(float(x[0].abs().max()) for x in f[0]), "max_abs_cache": float(f[1].abs().max()),
                       "peak_step_gb_full": f[4], "peak_step_gb_piece": r_[4]}
                print(json.dumps(rec), flush=True)
                res.append(rec)
                del r_, outs, cc, ct, cl
                torch.cuda.empty_cache()
            del f
            torch.cuda.empty_cache()
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
