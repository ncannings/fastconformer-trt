# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Nigel Cannings. Part of fastconformer-trt (see LICENSE and NOTICE).
"""Which FP8 linear layers hurt the encoder output most? Per group of quantizers kept in FP16, the error of the
fake-quantised streaming step against the float step, teacher-forced on the float step's own inputs and caches.

Hypothesis tested: the FP8 error is concentrated in a few layers or a few linear types, so keeping those in FP16
(recipe skip=...) buys most of the accuracy back for a small share of the FP8 speed. Ablation: group "none".

Metric per group: relative squared error of the encoder output over valid frames (sum (q - f)^2 / sum f^2) and its
reduction against all-FP8, over --n dev-clean utterances (never a test set), at most --max-steps steps.

usage: python trt/sensitivity.py --r 13 [--recipe base] [--n 8] [--max-steps 40] [--out FILE.jsonl]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import fq_step  # noqa: E402
import recipe as RC  # noqa: E402

TYPES = ["feed_forward1.linear1", "feed_forward1.linear2", "feed_forward2.linear1", "feed_forward2.linear2",
         "linear_q", "linear_k", "linear_v", "linear_pos", "linear_out"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=int, required=True)
    ap.add_argument("--recipe", default="base")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    import gpu_reserve
    gpu_reserve.reserve(float(os.environ.get("RESERVE_HOST_GB", "6")), float(os.environ.get("RESERVE_CUDA_GB", "4")))
    C.seed_all()
    m = C.load_model(C.MODELS["ml"], a.r)
    enc = m.encoder
    auds = C.dev_audios("dev_clean", a.n, C.SEED + 3)
    recs = []
    with torch.no_grad():
        cc, ct, cl = enc.get_initial_cache_state(batch_size=len(auds))
        for k, ch, ln, last in C.stock_chunks(m, auds):
            if k >= a.max_steps:
                break
            drop = 0 if k == 0 else enc.streaming_cfg.drop_extra_pre_encoded
            out = enc.cache_aware_stream_step(processed_signal=ch, processed_signal_length=ln, cache_last_channel=cc,
                                              cache_last_time=ct, cache_last_channel_len=cl, keep_all_outputs=True,
                                              drop_extra_pre_encoded=drop)
            recs.append((ch, ln, cc.half(), ct.half(), cl, drop, out[0], out[1]))
            cc, ct, cl = out[2], out[3], out[4]
        RC.quantize(m, a.recipe, fq_step._run_stream, C.SEED, log=lambda t: print(t, flush=True))
    from modelopt.torch.quantization.nn import TensorQuantizer
    qs = [(n, q) for n, q in enc.named_modules() if isinstance(q, TensorQuantizer) and q.is_enabled]

    def err() -> float:
        num = den = 0.0
        with torch.no_grad():
            for ch, ln, cc, ct, cl, drop, e0, el0 in recs:
                e = enc.cache_aware_stream_step(processed_signal=ch, processed_signal_length=ln,
                                                cache_last_channel=cc.float(), cache_last_time=ct.float(),
                                                cache_last_channel_len=cl, keep_all_outputs=True,
                                                drop_extra_pre_encoded=drop)[0]
                v = (torch.arange(e0.shape[-1], device=e0.device)[None] < el0[:, None])[:, None, :]
                num += float((((e - e0) ** 2) * v).sum())
                den += float(((e0 ** 2) * v).sum())
        return num / den

    groups = [("none", [])] + [(f"L{i}", [f"layers.{i}."]) for i in range(len(enc.layers))] + [(t, [t]) for t in TYPES]
    base = None
    rows = []
    for name, pats in groups:
        sel = [q for n, q in qs if any(p in n for p in pats)]
        for q in sel:
            q.disable()
        e = err()
        for q in sel:
            q.enable()
        base = e if base is None else base
        row = {"r": a.r, "recipe": a.recipe, "group": name, "n_quantizers_fp16": len(sel), "rel_sq_err": e,
               "reduction": 1.0 - e / base, "steps": len(recs), "n_utts": a.n}
        rows.append(row)
        print("SENS", json.dumps(row), flush=True)
        if a.out:
            with open(a.out, "a") as f:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
