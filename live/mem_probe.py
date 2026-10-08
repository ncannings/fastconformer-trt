# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Nigel Cannings. Part of fastconformer-trt (see LICENSE and NOTICE).
"""Memory probe for the live server (7 Oct, after the H100 warm-up OOMs): one server run at N slots, all streams
starting within --stagger-s, with the given encoder and decoder; reports the process peaks over everything (start-up,
warm-up and serving) of allocated and reserved CUDA memory, the reserved-but-unallocated gap at the end, and the
transcripts (sha256) so two code versions can be compared for identical output.

usage: python mem_probe.py OUT_JSON --r 13 --n 1024 [--dur-s 20] [--encoder FILE.py:build --engine P
       --cache-dtype float16] [--decoder fused]
"""
from __future__ import annotations

import argparse
import hashlib
import json

import torch

import harness
import loadgen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--r", type=int, default=13)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--dur-s", type=float, default=20.0)
    ap.add_argument("--stagger-s", type=float, default=5.0)
    ap.add_argument("--encoder", default="stock")
    ap.add_argument("--engine", default=None)
    ap.add_argument("--cache-dtype", default="float32", choices=["float32", "float16"])
    ap.add_argument("--decoder", default="fused")
    ap.add_argument("--mel-graph", action="store_true")
    ap.add_argument("--seed", type=int, default=20261005)
    a = ap.parse_args()
    m = harness.load_model("nvidia/nemotron-3.5-asr-streaming-0.6b", a.r, "en-US", a.seed)
    enc = harness.make_encoder_step(m, a.encoder, a.engine, a.r)
    sources, streams = loadgen.make_load("earnings22_full", a.n, a.seed + a.n, a.stagger_s, a.dur_s)
    torch.cuda.synchronize()
    base_alloc = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    srv = harness.Server(m, sources, streams, ["en-US"] * a.n, "chunk", "virtual", 0.0, enc,
                         cache_dtype=a.cache_dtype, decoder=a.decoder, mel_graph=a.mel_graph)
    pk = []
    orig_run = srv.run

    def run():                                  # harness.run resets the peak after warm-up: read it just before
        orig = torch.cuda.reset_peak_memory_stats

        def rp(*x, **k):
            pk.append((torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()))
            torch.cuda.reset_peak_memory_stats = orig
            return orig(*x, **k)
        torch.cuda.reset_peak_memory_stats = rp
        return orig_run()
    run()
    torch.cuda.synchronize()
    serve = (torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved())
    txt = "\n".join(s.hyp.text for s in sorted(srv.finished, key=lambda s: s.ls.sid))
    rec = {"r": a.r, "n": a.n, "encoder": a.encoder, "decoder": a.decoder,
           "base_allocated_gb": base_alloc / 1e9,
           "startup_warmup_peak_allocated_gb": pk[0][0] / 1e9 if pk else None,
           "startup_warmup_peak_reserved_gb": pk[0][1] / 1e9 if pk else None,
           "serving_peak_allocated_gb": serve[0] / 1e9, "serving_peak_reserved_gb": serve[1] / 1e9,
           "end_reserved_minus_allocated_gb": (torch.cuda.memory_reserved() - torch.cuda.memory_allocated()) / 1e9,
           "n_finished": len(srv.finished), "transcripts_sha256": hashlib.sha256(txt.encode()).hexdigest()[:16]}
    print(json.dumps(rec), flush=True)
    json.dump(rec, open(a.out, "w"), indent=1)
    open(a.out + ".txt", "w").write(txt + "\n")      # transcripts, one stream per line in stream order (diffable)


if __name__ == "__main__":
    main()
