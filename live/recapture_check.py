# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Nigel Cannings. Part of fastconformer-trt (see LICENSE and NOTICE).
"""Zero-re-capture check for --decoder fused: run the live server at each N (slots), counting every CUDA-graph capture
in the process (torch.cuda.CUDAGraph.capture_begin, patched) before the run starts (start-up: model, fused decoder,
warm-up) and during the run, plus NeMo decoder re-initialisations (the harness trace). Pass: every stream finishes,
no capture during any run, NeMo's decoder never re-initialised. On the H100, NeMo's decoder pre-warm crashed at 384
or more slots (6 Oct); this is the matching check for the fused decoder.

usage: python recapture_check.py OUT_JSON --r 13 --n 384,2048 [--dur-s 30] [--stagger-s 5]
       [--encoder FILE.py:build --engine P --cache-dtype float16] [--mel-graph]
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import torch

import harness
import loadgen

CAPTURES: list[float] = []
_orig = torch.cuda.CUDAGraph.capture_begin


def _counted(self, *a, **k):
    CAPTURES.append(time.perf_counter())
    return _orig(self, *a, **k)


torch.cuda.CUDAGraph.capture_begin = _counted


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--r", type=int, default=13)
    ap.add_argument("--n", default="384,2048")
    ap.add_argument("--dur-s", type=float, default=30.0)
    ap.add_argument("--stagger-s", type=float, default=5.0)
    ap.add_argument("--encoder", default="stock")
    ap.add_argument("--engine", default=None)
    ap.add_argument("--cache-dtype", default="float32", choices=["float32", "float16"])
    ap.add_argument("--mel-graph", action="store_true")
    ap.add_argument("--seed", type=int, default=20261005)
    a = ap.parse_args()
    print(f"seed {a.seed}", flush=True)
    m = harness.load_model("nvidia/nemotron-3.5-asr-streaming-0.6b", a.r, "en-US", a.seed)
    enc = harness.make_encoder_step(m, a.encoder, a.engine, a.r)
    res, ok_all = [], True
    for n in [int(x) for x in a.n.split(",")]:
        sources, streams = loadgen.make_load("earnings22_full", n, a.seed + n, a.stagger_s, a.dur_s)
        c0 = len(CAPTURES)
        srv = harness.Server(m, sources, streams, ["en-US"] * n, "chunk", "virtual", 0.0, enc,
                             cache_dtype=a.cache_dtype, decoder="fused", mel_graph=a.mel_graph)
        srv.run()                                  # run() warms up first, then sets t0 and serves
        torch.cuda.synchronize()
        t0_abs = srv.t0                            # perf_counter at the start of serving
        at_start = sum(1 for t in CAPTURES[c0:] if t < t0_abs)
        in_run = sum(1 for t in CAPTURES[c0:] if t >= t0_abs)
        s = srv.summary(harness.CHUNK_MS[a.r] / 1000)
        ok = (in_run == 0 and not s["decoder_graph_reinits"] and s["n_finished"] == n and s["aborted"] is None)
        ok_all &= ok
        rec = {"r": a.r, "n": n, "pass": ok, "captures_at_startup": at_start, "captures_during_run": in_run,
               "nemo_decoder_reinits": s["decoder_graph_reinits"], "n_finished": s["n_finished"],
               "batch_max": s["batch_size"]["max"], "fused": s["fused_decoder"],
               "peak_gpu_mem_allocated_gb": s["peak_gpu_mem_allocated_gb"], "step_split_ms_mean": s["step_split_ms_mean"]}
        print(json.dumps(rec), flush=True)
        res.append(rec)
        del srv
    json.dump({"pass": ok_all, "runs": res}, open(a.out, "w"), indent=1)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
