# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Nigel Cannings. Part of fastconformer-trt (see LICENSE and NOTICE).
"""Check of the persistent-buffer TensorRT call path (7 Oct) against the legacy path (per-call outputs, torch.cat of
the pieces): same inputs, both paths, outputs compared bit for bit, for the first-chunk and steady engines, with one
call (B <= max B) and with pieces (B > TRT_MAX_CALL_B). Inputs: two chained steps from zero caches with random features
(seeded), so the second step sees non-trivial caches. Also the peak memory of one call at each B, both paths.

usage: python trt/check_call_path.py OUT_JSON --r 13 --engine /out/trt/ml_r13/steady_fp8_b1024.plan --b 64,200,1024,2048
       (run with TRT_MAX_CALL_B set, e.g. 512, so the larger B run in pieces)
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
    ap.add_argument("--b", default="64,200,1024,2048")
    ap.add_argument("--seed", type=int, default=20261005)
    a = ap.parse_args()
    m = harness.load_model("nvidia/nemotron-3.5-asr-streaming-0.6b", a.r, "en-US", a.seed)
    st = trt_step.build(m, a.engine, a.r)
    cfg = m.encoder.streaming_cfg
    res = []
    with torch.inference_mode():
        for B in [int(x) for x in a.b.split(",")]:
            for first in (True, False):
                W = (st.w_first if first else st.w_steady)
                drop = 0 if first else st.drop
                g = torch.Generator(device="cuda").manual_seed(a.seed + B)
                sig = torch.randn(B, 128, W, device="cuda", generator=g)
                lens = torch.full((B,), W, dtype=torch.int64, device="cuda")
                cc0, ct0, cl0 = m.encoder.get_initial_cache_state(batch_size=B)
                cc0, ct0 = cc0.half(), ct0.half()
                outs, peaks = {}, {}
                for legacy in (True, False):
                    trt_step.LEGACY = legacy
                    torch.cuda.synchronize()
                    cc, ct, cl = cc0.clone(), ct0.clone(), cl0.clone()
                    seq = []
                    for k in range(2):                       # two chained steps; caches cloned like the harness slab
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        base = torch.cuda.memory_allocated()
                        e, el, cc2, ct2, cl2 = st(sig, lens, cc, ct, cl, drop)
                        torch.cuda.synchronize()
                        peaks.setdefault(legacy, []).append((torch.cuda.max_memory_allocated() - base) / 1e9)
                        seq.append([x.clone() for x in (e, el, cc2, ct2, cl2)])
                        cc, ct, cl = cc2.clone(), ct2.clone(), cl2.clone()
                    outs[legacy] = seq
                same = all(torch.equal(x.nan_to_num(), y.nan_to_num()) and torch.equal(x.isnan(), y.isnan())
                           for s0, s1 in zip(outs[True], outs[False]) for x, y in zip(s0, s1))
                rec = {"B": B, "first": first, "max_call_b": (st.first if first else st.steady).max_b,
                       "bit_identical": same,
                       "peak_gb_legacy": [round(x, 2) for x in peaks[True]],
                       "peak_gb_new": [round(x, 2) for x in peaks[False]],
                       "max_abs_diff": max(float((x.float() - y.float()).abs().nan_to_num().max())
                                           for s0, s1 in zip(outs[True], outs[False]) for x, y in zip(s0, s1))}
                print(json.dumps(rec), flush=True)
                res.append(rec)
                del outs
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
