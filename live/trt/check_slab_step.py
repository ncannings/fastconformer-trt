# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Nigel Cannings. Part of fastconformer-trt (see LICENSE and NOTICE).
"""Check of TrtStep.slab_step (8 Oct, piece-wise gather, run and scatter on the cache slab) against the full-batch path
(gather all rows, __call__, scatter all rows): random float16 slabs of N slots, a random subset of B slots in random
order, random features; two chained steps (first-chunk then steady engine, or steady twice). Compares the encoder
output, lengths and the whole slabs bit for bit, and the peak memory of one step for each path.

usage: python trt/check_slab_step.py OUT_JSON --r 13 --engine /out/trt/ml_r13/steady_fp8_b1024.plan --slots 3000
       --b 200,1024,2048 (with TRT_MAX_CALL_B set, e.g. 512, so large B run in pieces)
"""
from __future__ import annotations

import argparse
import json

import torch

import harness
import trt_step


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--r", type=int, default=13)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--slots", type=int, default=3000)
    ap.add_argument("--b", default="200,1024,2048")
    ap.add_argument("--seed", type=int, default=20261005)
    a = ap.parse_args()
    m = harness.load_model("nvidia/nemotron-3.5-asr-streaming-0.6b", a.r, "en-US", a.seed)
    st = trt_step.build(m, a.engine, a.r)
    N = a.slots
    res = []
    with torch.inference_mode():
        g = torch.Generator(device="cuda").manual_seed(a.seed)
        cc0, ct0, cl0 = m.encoder.get_initial_cache_state(batch_size=N)
        base_cc = (torch.randn(cc0.shape, device="cuda", generator=g) * 0.5).half()
        base_ct = (torch.randn(ct0.shape, device="cuda", generator=g) * 0.5).half()
        base_cl = torch.randint(0, 57, (N,), device="cuda", generator=g)
        for B in [int(x) for x in a.b.split(",")]:
            slots = torch.randperm(N, device="cuda", generator=g)[:B]
            outs, peak = {}, {}
            for mode in ("full", "slab"):
                cc, ct, cl = base_cc.clone(), base_ct.clone(), base_cl.clone()
                es = []
                pk = []
                for k, first in enumerate((True, False)):
                    W = st.w_first if first else st.w_steady
                    drop = 0 if first else st.drop
                    sig = torch.randn(B, 128, W, device="cuda", generator=torch.Generator(device="cuda").manual_seed(a.seed + B + k))
                    lens = torch.full((B,), W, dtype=torch.int64, device="cuda")
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    b0 = torch.cuda.memory_allocated()
                    if mode == "full":
                        e, el, c2, t2, l2 = st(sig, lens, cc.index_select(1, slots), ct.index_select(1, slots),
                                               cl.index_select(0, slots), drop)
                        cc.index_copy_(1, slots, c2)
                        ct.index_copy_(1, slots, t2)
                        cl.index_copy_(0, slots, l2)
                    else:
                        e, el = st.slab_step(sig, lens, cc, ct, cl, slots, drop)
                    torch.cuda.synchronize()
                    pk.append(round((torch.cuda.max_memory_allocated() - b0) / 1e9, 2))
                    es.append((e.clone(), el.clone()))
                outs[mode] = (es, cc, ct, cl)
                peak[mode] = pk
            f, s_ = outs["full"], outs["slab"]
            same = (all(torch.equal(x[0], y[0]) and torch.equal(x[1], y[1]) for x, y in zip(f[0], s_[0]))
                    and torch.equal(f[1], s_[1]) and torch.equal(f[2], s_[2]) and torch.equal(f[3], s_[3]))
            rec = {"slots": N, "B": B, "bit_identical": bool(same), "peak_step_gb_full": peak["full"],
                   "peak_step_gb_slab": peak["slab"], "fallbacks": st.fallbacks}
            print(json.dumps(rec), flush=True)
            res.append(rec)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
