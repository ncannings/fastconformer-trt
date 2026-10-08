# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Nigel Cannings. Part of fastconformer-trt (see LICENSE and NOTICE).
"""Decoder equivalence (gate a for --decoder fused): NeMo's label-looping decoder (through the lean decoder's slabs) and
the fused decoder are fed the SAME encoder outputs, call by call, inside one harness run, each carrying its own
per-slot state. Every call's emitted tokens are compared per stream; the first divergence of each stream is recorded
with its call index, and final transcripts are compared as strings. This isolates the decoder: no encoder, batching
or scheduling difference can enter.

usage: python dec_equiv.py OUT_JSON --r 13 --n 8 [--max-s 120] [--stagger-s 30] [--tick eager|grid]
       [--joint torch|triton] [--source earnings22_full] [--set test_clean --concurrency 64 --nutt 200]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

import finetune_stride as fs
import harness
import loadgen
from fused_decoder import FusedDecoder
from lean_decoder import LeanDecoder


class Tee:
    def __init__(self, lean: LeanDecoder, fused: FusedDecoder):
        self.a, self.b = lean, fused
        self.calls = 0
        self.first_div: dict[int, dict] = {}

    def reset(self, slot: int) -> None:
        self.a.reset(slot)
        self.b.reset(slot)
        self.first_div.pop(slot, None)

    def step(self, enc, enc_len, slots, first) -> None:
        self.a.step(enc, enc_len, slots, first)
        self.b.step(enc, enc_len, slots, first)
        self.calls += 1
        la = self.a.tok_len.index_select(0, slots)
        lb = self.b.tok_len.index_select(0, slots)
        for j, s in enumerate(slots.tolist()):
            if s in self.first_div:
                continue
            na, nb = int(la[j]), int(lb[j])
            n = min(na, nb, self.a.cap)
            ta = self.a.tok[s, :n]
            tb = self.b.tok[s, :n]
            if na != nb or not torch.equal(ta, tb):
                self.first_div[s] = {"call": self.calls, "first": bool(first), "len_lean": na, "len_fused": nb}

    def text(self, slot: int):
        ta, ia = self.a.text(slot)
        tb, ib = self.b.text(slot)
        self.last = (ta, tb, ia == ib, self.first_div.get(slot))
        return tb, ib


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--r", type=int, default=13)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-s", type=float, default=120)
    ap.add_argument("--stagger-s", type=float, default=30)
    ap.add_argument("--tick", default="eager", choices=["eager", "grid"])
    ap.add_argument("--joint", default="torch", choices=["torch", "triton"])
    ap.add_argument("--lstm", default="cudnn", choices=["cudnn", "triton"])
    ap.add_argument("--source", default="earnings22_full")
    ap.add_argument("--set", default=None, help="evaluation set instead of long-form streams (closed loop)")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--nutt", type=int, default=0)
    ap.add_argument("--encoder", default="stock")
    ap.add_argument("--engine", default=None)
    ap.add_argument("--cache-dtype", default="float32", choices=["float32", "float16"])
    ap.add_argument("--seed", type=int, default=20261005)
    a = ap.parse_args()
    print(f"seed {a.seed}", flush=True)
    m = harness.load_model("nvidia/nemotron-3.5-asr-streaming-0.6b", a.r, "en-US", a.seed)
    enc = harness.make_encoder_step(m, a.encoder, a.engine, a.r)
    if a.set:
        rows = fs.rows_of(a.set)
        rows = rows[:a.nutt] if a.nutt else rows
        sources = [fs.load_audio(r) for r in rows]
        gap = float(np.mean([len(x) for x in sources])) / 16000 / a.concurrency
        streams = [loadgen.LoadStream(sid=i, src=i, start=0, n_samples=len(x), arrival_s=i * gap, text=fs.ref_text(r))
                   for i, (x, r) in enumerate(zip(sources, rows))]
        kw = {"max_slots": a.concurrency, "admit_limit": a.concurrency}
        tick = 0.0
    else:
        sources, streams = loadgen.from_specs(loadgen.make_streams(a.source, a.n, a.seed, a.stagger_s, a.max_s))
        kw = {}
        tick = 0.0 if a.tick == "eager" else harness.CHUNK_MS[a.r] / 1000
    srv = harness.Server(m, sources, streams, ["en-US"] * len(streams), "chunk", "virtual", tick, enc,
                         cache_dtype=a.cache_dtype, decoder="lean", **kw)
    fused = FusedDecoder(m, srv.n_slots, srv.lean.cap, int(srv.cfg.valid_out_len), joint=a.joint,
                         lstm=a.lstm)
    tee = Tee(srv.lean, fused)
    srv.lean = tee
    per = []
    orig_text = tee.text

    def text(slot):
        r = orig_text(slot)
        per.append(tee.last)
        return r
    tee.text = text
    srv.run()
    same_tok = sum(1 for x in per if x[2])
    same_str = sum(1 for x in per if x[0] == x[1])
    divs = [x[3] for x in per if x[3] is not None]
    res = {"r": a.r, "n_streams": len(per), "joint": a.joint, "lstm": a.lstm, "tick": a.tick, "set": a.set, "calls": tee.calls,
           "identical_token_sequences": f"{same_tok}/{len(per)}",
           "identical_transcripts": f"{same_str}/{len(per)}", "first_divergences": divs[:50],
           "fused_stats": fused.stats(), "nemo_reinits": srv.reinit_events[:20] if hasattr(srv, "reinit_events") else None,
           "seed": a.seed}
    print(json.dumps(res), flush=True)
    diffs = [{"lean": x[0], "fused": x[1], "div": x[3]} for x in per if not x[2]]
    json.dump({"result": res, "differences": diffs[:100]}, open(a.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
