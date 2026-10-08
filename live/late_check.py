# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Nigel Cannings. Part of fastconformer-trt (see LICENSE and NOTICE).
"""Test the end-of-stream hypothesis for the late-chunk rule (6 October): with every stream exactly DUR_S long, each
stream ends on the same short final chunk, so its second-to-last chunk counts as late whenever its step starts more
than that remainder after it was ready. Runs the harness (virtual clock, continuous batching, warm-up on) on the
sweep's load twice per arm: (a) streams of exactly --dur-s (as the sweep), (b) streams cut to end on a full chunk
(the final chunk is a whole chunk). Reports late chunks in total, those that are a stream's second-to-last chunk, and
the alternative count "step started more than one chunk duration after the chunk was ready". Shared GPU: the step
times are not the quiet-window ones, so only the split of the late count matters.

usage: python late_check.py OUT_JSON --r 13 --arms stock:200,stockgc:200,fp8lean:600 [--dur-s 120]
"""
from __future__ import annotations

import argparse
import json
import os

import harness
import loadgen


def aligned_samples(m, dur_s: float) -> int:
    cfg = m.encoder.streaming_cfg
    first, steady = cfg.chunk_size if isinstance(cfg.chunk_size, list) else (cfg.chunk_size, cfg.chunk_size)
    frames = int(dur_s * 100) + 1
    k = (frames - first) // steady
    n_frames = first + k * steady                     # whole chunks only
    return (n_frames - 1) * 160


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--r", type=int, default=13)
    ap.add_argument("--arms", default="stock:200,stockgc:200,fp8lean:600")
    ap.add_argument("--dur-s", type=float, default=120.0)
    ap.add_argument("--seed", type=int, default=20261005)
    ap.add_argument("--loads", default="exact,spread", help="exact (equal lengths), spread (len_spread 0.5), aligned")
    ap.add_argument("--clock", default="virtual", choices=["virtual", "real"])
    a = ap.parse_args()
    m = harness.load_model("nvidia/nemotron-3.5-asr-streaming-0.6b", a.r, "en-US", a.seed)
    eng = f"/out/trt/ml_r{a.r}/steady_fp8_b1024.plan"
    res = []
    for spec in a.arms.split(","):
        arm, n = spec.split(":")
        n = int(n)
        if arm.startswith("fp8"):
            enc, kw = harness.make_encoder_step(m, "/w/trt/trt_step.py:build", eng, a.r), \
                {"cache_dtype": "float16", "decoder": "lean"}
        else:
            enc, kw = harness.make_encoder_step(m, "stock", None, a.r), {}
        os.environ["HARNESS_GC"] = "freeze" if arm == "stockgc" else "default"
        for load in a.loads.split(","):
            sources, streams = loadgen.make_load("earnings22_full", n, a.seed + n, 10.0, a.dur_s,
                                                 len_spread=0.5 if load == "spread" else 0.0)
            if load == "aligned":
                ns = aligned_samples(m, a.dur_s)
                for s in streams:
                    s.n_samples = ns
            s = harness.run(None, m, sources, streams, ["en-US"] * n, "chunk", a.clock, a.r, a.seed, None, 0.0, enc,
                            **kw)
            rec = {"arm": arm, "n": n, "load": load, "stream_s": [min(x.n_samples for x in streams) / 16000, max(x.n_samples for x in streams) / 16000],
                   "clock": a.clock, "late_pct": round(100 * s["deadline_misses"] / max(1, s["n_chunks"]), 3),
                   "p95_ms": s["final_token_latency_ms_excl_algo"]["p95"],
                   "chunks": s["n_chunks"], "late": s["deadline_misses"],
                   "late_second_to_last": s["late_second_to_last_chunk"],
                   "late_wait_over_one_chunk": s["late_by_wait_over_one_chunk"],
                   "step_p50_ms": s["step_wall_ms"]["p50"], "gc_gen2": s["gc_gen2_count"],
                   "note": "shared GPU, not reportable"}
            print(json.dumps(rec), flush=True)
            res.append(rec)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
