# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Nigel Cannings. Part of fastconformer-trt (see LICENSE and NOTICE).
"""Gate (c) for --decoder fused: decoder time per call, fused against lean (NeMo's label-looping computer), on identical
inputs, without the encoder in the process (the encoder's memory at N = 1,024 does not fit beside the shared GPU's
other tenants).

Two stages.
  record: run the harness (all N0 streams start together, so steady calls carry all N0) with the given encoder and the
          fused decoder, and save every decoder call's input (encoder output after the language prompt, lengths, slot
          ids, first-chunk flag) to OUT.pt.
  bench:  load the model only; for each N (a multiple of N0) build N slots, replay the recorded calls with the rows
          tiled N / N0 times (copy k of slot s is slot s + k N0), through each decoder arm, twice: the first pass warms
          up (NeMo captures its graphs at the new batch size there), the second pass is timed, every call alone
          (synchronise before and after, wall clock). The fused decoder's done-flag check and continuations are inside
          its time; for lean, NeMo's computer call, state gather and scatter and token append. Also checks that every
          arm emits the same tokens per slot as lean (torch joint: expected identical; triton: near-ties may differ).
On a shared GPU the times are NOT reportable; the ratio is indicative.

usage: python bench_decoder.py record OUT.pt --r 13 --n0 256 [--dur-s 12] [--encoder FILE.py:build --engine P
       --cache-dtype float16]
       python bench_decoder.py bench OUT_JSON --rec OUT.pt --r 13 --n 256,1024 [--arms lean,fused:torch,fused:triton,fused:triton:triton]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

import harness
import loadgen


def record(a) -> None:
    m = harness.load_model("nvidia/nemotron-3.5-asr-streaming-0.6b", a.r, "en-US", a.seed)
    enc = harness.make_encoder_step(m, a.encoder, a.engine, a.r)
    sources, streams = loadgen.make_load("earnings22_full", a.n0, a.seed, 0.0, a.dur_s)
    srv = harness.Server(m, sources, streams, ["en-US"] * a.n0, "chunk", "virtual", 0.0, enc,
                         cache_dtype=a.cache_dtype, decoder="fused")
    calls = []
    orig = srv.lean.step

    def step(e, el, slots, first):
        calls.append((e.detach().float().cpu(), el.cpu(), slots.cpu(), bool(first)))
        orig(e, el, slots, first)
    srv.lean.step = step
    srv.run()
    torch.save({"r": a.r, "n0": a.n0, "calls": calls, "encoder": a.encoder, "engine": a.engine,
                "max_tokens": srv.lean.cap, "seed": a.seed}, a.out)
    print(json.dumps({"recorded_calls": len(calls), "n0": a.n0, "r": a.r}), flush=True)


def bench(a) -> None:
    from fused_decoder import FusedDecoder
    from lean_decoder import LeanDecoder
    rec = torch.load(a.rec, weights_only=True)
    n0, calls = rec["n0"], rec["calls"]
    m = harness.load_model("nvidia/nemotron-3.5-asr-streaming-0.6b", a.r, "en-US", a.seed)
    t_cap = int(m.encoder.streaming_cfg.valid_out_len)
    res = []
    for n in [int(x) for x in a.n.split(",")]:
        k = n // n0
        if k * n0 != n:
            raise ValueError(f"N={n} is not a multiple of the recorded N0={n0}")
        gpu_calls = []
        for e, el, sl, first in calls:
            gpu_calls.append((e.cuda().repeat(k, 1, 1).contiguous(), el.cuda().repeat(k),
                              torch.cat([sl + j * n0 for j in range(k)]).cuda(), first))
        toks = {}
        for arm in a.arms.split(","):
            dec, joint, lstm = (arm.split(":") + ["", ""])[:3]
            t0 = time.perf_counter()
            if dec == "lean":
                d = LeanDecoder(m, n, rec["max_tokens"])
            else:
                d = FusedDecoder(m, n, rec["max_tokens"], t_cap, joint=joint or None, lstm=lstm or None)
            init_s = time.perf_counter() - t0
            times = []
            for p in range(2):
                for s in range(n):
                    d.reset(s)
                for e, el, sl, first in gpu_calls:
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    d.step(e, el, sl, first)
                    torch.cuda.synchronize()
                    if p == 1:
                        times.append((sl.numel(), first, (time.perf_counter() - t1) * 1000))
            tl = d.tok_len[:n].cpu()
            toks[arm] = [d.tok[s, :int(tl[s])].cpu() for s in range(n)]
            ref = toks.get("lean")
            same = sum(torch.equal(x, y) for x, y in zip(ref, toks[arm])) if ref is not None else None
            steady = [ms for b, f, ms in times if b == n and not f]
            r = {"r": a.r, "n": n, "arm": arm, "init_s": round(init_s, 2), "calls": len(times),
                 "steady_calls": len(steady),
                 "steady_ms_median": float(np.median(steady)) if steady else None,
                 "steady_ms_p90": float(np.percentile(steady, 90)) if steady else None,
                 "all_ms_sum": float(sum(x[2] for x in times)),
                 "slots_identical_to_lean": f"{same}/{n}" if same is not None else None,
                 "fused": d.stats() if hasattr(d, "stats") else None,
                 "max_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
                 "note": "shared GPU: not reportable"}
            print(json.dumps(r), flush=True)
            res.append(r)
            del d
        del gpu_calls
    json.dump(res, open(a.out, "w"), indent=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["record", "bench"])
    ap.add_argument("out")
    ap.add_argument("--r", type=int, default=13)
    ap.add_argument("--n0", type=int, default=256)
    ap.add_argument("--n", default="256,1024")
    ap.add_argument("--dur-s", type=float, default=12.0)
    ap.add_argument("--rec", default=None)
    ap.add_argument("--arms", default="lean,fused:torch,fused:triton")
    ap.add_argument("--encoder", default="stock")
    ap.add_argument("--engine", default=None)
    ap.add_argument("--cache-dtype", default="float32", choices=["float32", "float16"])
    ap.add_argument("--seed", type=int, default=20261005)
    a = ap.parse_args()
    print(f"seed {a.seed}", flush=True)
    record(a) if a.stage == "record" else bench(a)


if __name__ == "__main__":
    main()
