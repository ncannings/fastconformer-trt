# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Nigel Cannings. Part of fastconformer-trt (see LICENSE and NOTICE).
"""Profile one harness step: where the time goes at N streams, host (Python, launches) versus GPU kernels.

All N streams start together (stagger 0), so after the first chunk every step carries all N streams (B = N). A short
warm-up run first, then a profiled run (torch.profiler, CPU + CUDA) over --dur-s of audio. For each stage of the step
(live.mel, live.slab_gather, live.encoder, live.slab_scatter, live.prompt, live.decoder; labels in harness.py) it
reports per step: host time (CPU wall inside the stage's range), GPU kernel time (sum of the durations of the CUDA
kernels launched from inside it) and kernel count, plus the measured step wall time and all kernel time in the step. Host time well above kernel time means the stage is bound by Python or
launch overhead; kernel time close to wall means it is GPU bound. On a shared GPU the absolute numbers are not
reportable (kernels queue behind other work), but the host-side numbers and the ratios are indicative.

usage: python profile_step.py OUT_JSON [--r 13,3] [--n 64,256] [--dur-s 8] [--encoder stock|FILE.py:build --engine P]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

import harness
import loadgen

STAGES = ["live.mel", "live.slab_gather", "live.encoder", "live.slab_scatter", "live.prompt", "live.decoder"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--r", default="13,3")
    ap.add_argument("--n", default="64,256")
    ap.add_argument("--dur-s", type=float, default=8.0)
    ap.add_argument("--encoder", default="stock")
    ap.add_argument("--engine", default=None)
    ap.add_argument("--decoder", default="stock", choices=["stock", "lean", "fused"])
    ap.add_argument("--mel-graph", action="store_true")
    ap.add_argument("--cache-dtype", default="float32")
    ap.add_argument("--seed", type=int, default=20261005)
    a = ap.parse_args()
    rs = [int(x) for x in a.r.split(",")]
    m = harness.load_model("nvidia/nemotron-3.5-asr-streaming-0.6b", rs[0], "en-US", a.seed)
    results = []
    for r in rs:
        harness.set_chunk(m, r)
        enc = harness.make_encoder_step(m, a.encoder, a.engine, r)
        for n in [int(x) for x in a.n.split(",")]:
            sources, streams = loadgen.make_load("earnings22_full", n, a.seed, 0.0, a.dur_s)
            harness.run(None, m, sources, streams, ["en-US"] * n, "chunk", "virtual", r, a.seed, None, 0.0, enc,
                        cache_dtype=a.cache_dtype, decoder=a.decoder, mel_graph=a.mel_graph)                                # warm-up
            srv = harness.Server(m, sources, streams, ["en-US"] * n, "chunk", "virtual", 0.0, enc,
                                 cache_dtype=a.cache_dtype, decoder=a.decoder, mel_graph=a.mel_graph)
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                srv.run()
            steady = [x for x in srv.steps if x["B"] == n and x["n_first"] == 0]
            ns = len(srv.steps)
            stages = {k: {"host_ms": 0.0, "kernel_ms": 0.0, "n_kernels": 0} for k in STAGES}

            def walk(e):
                ms, nk = sum(k.duration for k in e.kernels) / 1000, len(e.kernels)
                for c in e.cpu_children:
                    a_, b_ = walk(c)
                    ms, nk = ms + a_, nk + b_
                return ms, nk
            evs = prof.events()
            for e in evs:
                if e.name in stages and e.device_type == torch.autograd.DeviceType.CPU:
                    st = stages[e.name]
                    st["host_ms"] += e.time_range.elapsed_us() / 1000
                    km, nk = walk(e)
                    st["kernel_ms"] += km
                    st["n_kernels"] += nk
            for st in stages.values():
                for k in list(st):
                    st[k] = st[k] / ns
            kern_total = sum(sum(k.duration for k in e.kernels) for e in evs) / 1000 / ns
            rec = {"r": r, "chunk_ms": harness.CHUNK_MS[r], "n": n, "n_steps": ns, "n_steady_steps": len(steady),
                   "step_wall_ms_mean": float(np.mean([x["wall_s"] for x in srv.steps])) * 1000,
                   "steady_step_wall_ms_median": float(np.median([x["wall_s"] for x in steady])) * 1000 if steady else None,
                   "gpu_kernel_ms_per_step_total": kern_total, "stages": stages,
                   "realtime_budget_ms": harness.CHUNK_MS[r],
                   "note": "shared GPU (vLLM busy): not reportable; host times and ratios indicative only"}
            print(json.dumps(rec), flush=True)
            results.append(rec)
            del srv, prof
    json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
