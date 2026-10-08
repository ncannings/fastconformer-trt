# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Nigel Cannings. Part of fastconformer-trt (see LICENSE and NOTICE).
"""Microbenchmark: encoder streaming step time per chunk versus number of streams N, stock PyTorch (float32, TF32, as
the stock script and the harness) versus TensorRT engines. Steady chunks only (full width, drop 2), caches from a
short real warm-up are not needed for timing, so inputs are seeded random features and random caches with full
cache length (56), which is the steady state of a live stream.

ON A SHARED GPU THESE TIMINGS ARE NOT REPORTABLE (vLLM co-tenant at about 96 % utilisation); they show direction
only. Rerun unchanged in a quiet window for numbers.

Per N: median and min of --iters CUDA-event-timed calls after --warmup calls. TensorRT is timed with float16 caches
in and out (a float16 slab, harness --cache-dtype float16) and, separately, with float32 caches (casts included).
Stock at large N needs float32 caches of 24 x N x 56 x 1024 x 4 bytes in and out (N=1024: 5.6 GB each way), so N
above --stock-max is skipped and recorded as skipped.

usage: python trt/bench_step.py --r 13 --engines /out/trt/ml_r13/steady_fp16_b512.plan,/out/trt/ml_r13/steady_fp8_b512.plan
       [--ns 1,16,64,256,1024] [--stock-max 256] [--iters 20] [--warmup 5]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import trt_step  # noqa: E402


def timeit(fn, warmup: int, iters: int) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
    ts.sort()
    return {"median_ms": ts[len(ts) // 2], "min_ms": ts[0], "max_ms": ts[-1]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=int, required=True)
    ap.add_argument("--model", default="ml", choices=sorted(C.MODELS))
    ap.add_argument("--engines", default="")
    ap.add_argument("--ns", default="1,16,64,256,1024")
    ap.add_argument("--stock-max", type=int, default=256)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    import gpu_reserve
    gpu_reserve.reserve(float(os.environ.get("RESERVE_HOST_GB", "6")), float(os.environ.get("RESERVE_CUDA_GB", "4")))
    C.seed_all()
    m = C.load_model(C.MODELS[a.model], a.r)
    enc = m.encoder
    g = C.geometry(enc)
    W, drop, L = g["steady_w"], g["drop"], g["layers"]
    steps = {"stock": None}
    for p in [e for e in a.engines.split(",") if e]:
        steps[os.path.basename(p)] = trt_step.build(m, p, a.r)
    rows = []
    for N in [int(x) for x in a.ns.split(",")]:
        gen = torch.Generator(device="cuda").manual_seed(C.SEED + N)
        sig = torch.randn(N, g["feat_in"], W, device="cuda", generator=gen)
        lens = torch.full((N,), W, dtype=torch.int64, device="cuda")
        cl = torch.full((N,), g["cache_len"], dtype=torch.int64, device="cuda")
        cc16 = (0.3 * torch.randn(L, N, g["cache_len"], g["d_model"], device="cuda", generator=gen)).half()
        ct16 = (10 * torch.randn(L, N, g["d_model"], g["conv_cache"], device="cuda", generator=gen)).half()
        row = {"N": N, "r": a.r, "chunk_ms": C.CHUNK_MS[a.r], "not_reportable": "shared GPU"}
        for name, st in steps.items():
            if st is None:
                if N > a.stock_max:
                    row["stock"] = "skipped (memory)"
                    continue
                cc, ct = cc16.float(), ct16.float()
                fn = lambda: enc.cache_aware_stream_step(processed_signal=sig, processed_signal_length=lens,
                                                         cache_last_channel=cc, cache_last_time=ct,
                                                         cache_last_channel_len=cl, keep_all_outputs=False,
                                                         drop_extra_pre_encoded=drop)
                row["stock"] = timeit(fn, a.warmup, a.iters)
                del cc, ct
            elif N > st.steady.max_b:
                row[name] = "skipped (N above engine max batch; memory)"
                continue
            else:
                fn = lambda: st(sig, lens, cc16, ct16, cl, drop)
                row[name + ":cache_fp16"] = timeit(fn, a.warmup, a.iters)
                if N <= a.stock_max:
                    cc, ct = cc16.float(), ct16.float()
                    fn = lambda: st(sig, lens, cc, ct, cl, drop)
                    row[name + ":cache_fp32"] = timeit(fn, a.warmup, a.iters)
                    del cc, ct
            torch.cuda.empty_cache()
        row["peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        print("BENCH", json.dumps(row), flush=True)
        rows.append(row)
        if a.out:
            with open(a.out, "a") as f:
                f.write(json.dumps(row) + "\n")
        del sig, cc16, ct16
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
