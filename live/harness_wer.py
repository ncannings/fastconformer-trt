# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Nigel Cannings. Part of fastconformer-trt (see LICENSE and NOTICE).
"""WER of the live harness on a whole evaluation set: every utterance is one stream, at most --concurrency streams live at
once (closed loop: a stream is admitted, and its audio starts arriving, when a slot frees), so they join and leave
throughout; virtual clock, continuous batching. Scored
with the offline scorer (finetune_stride.normaliser_for, jiwer corpus WER), so it is directly comparable with the P0
stock-script numbers (p0_stock.py) for the same set and chunk size. Use it to gate a custom arm (--encoder, --decoder,
--cache-dtype) on accuracy.

usage: python harness_wer.py OUT_JSON --set test_clean --r 13 [--decoder lean] [--encoder FILE.py:build --engine P]
       [--concurrency 64] [--n 0 (all)]
"""
from __future__ import annotations

import argparse
import json

import jiwer
import numpy as np

import finetune_stride as fs
import harness
import loadgen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--set", default="test_clean")
    ap.add_argument("--r", type=int, default=13)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--lang", default="en-US")
    ap.add_argument("--encoder", default="stock")
    ap.add_argument("--engine", default=None)
    ap.add_argument("--decoder", default="stock", choices=["stock", "lean", "fused"])
    ap.add_argument("--mel-graph", action="store_true")
    ap.add_argument("--cache-dtype", default="float32", choices=["float32", "float16"])
    ap.add_argument("--mel", default="chunk", choices=["chunk", "full"])
    ap.add_argument("--seed", type=int, default=20261005)
    a = ap.parse_args()
    print(f"seed {a.seed}", flush=True)
    rows = fs.rows_of(a.set)
    if a.n:
        rows = rows[:a.n]
    auds = [fs.load_audio(r) for r in rows]
    mean_s = float(np.mean([len(x) for x in auds])) / 16000
    gap = mean_s / a.concurrency
    sources = auds
    streams = [loadgen.LoadStream(sid=i, src=i, start=0, n_samples=len(x), arrival_s=i * gap,
                                  text=fs.ref_text(r)) for i, (x, r) in enumerate(zip(auds, rows))]
    m = harness.load_model("nvidia/nemotron-3.5-asr-streaming-0.6b", a.r, a.lang, a.seed)
    enc = harness.make_encoder_step(m, a.encoder, a.engine, a.r)
    srv = harness.Server(m, sources, streams, [a.lang] * len(streams), a.mel, "virtual", 0.0, enc,
                         cache_dtype=a.cache_dtype, decoder=a.decoder, mel_graph=a.mel_graph,
                         max_slots=a.concurrency, admit_limit=a.concurrency)
    srv.run()
    norm = fs.normaliser_for(a.set)
    pairs = [(norm(s.ls.text), norm(s.hyp.text) or "<empty>") for s in sorted(srv.finished, key=lambda s: s.ls.sid)]
    pairs = [(r, h) for r, h in pairs if r]
    res = {"set": a.set, "r": a.r, "chunk_ms": harness.CHUNK_MS[a.r], "n": len(pairs), "encoder": a.encoder,
           "engine": a.engine, "decoder": a.decoder, "cache_dtype": a.cache_dtype, "mel": a.mel,
           "concurrency": a.concurrency, "wer": 100 * jiwer.wer([r for r, _ in pairs], [h for _, h in pairs]),
           "mean_batch": float(np.mean([x["B"] for x in srv.steps])), "seed": a.seed,
           "fused_decoder": srv.lean.stats() if hasattr(srv.lean, "stats") else None,
           "nemo_decoder_reinits": len(getattr(srv, "reinit_events", []))}
    print(json.dumps(res), flush=True)
    with open(a.out, "w") as f:
        json.dump({"result": res, "hyps": [s.hyp.text for s in sorted(srv.finished, key=lambda s: s.ls.sid)]}, f,
                  ensure_ascii=False)


if __name__ == "__main__":
    main()
