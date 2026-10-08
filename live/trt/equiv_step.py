# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Nigel Cannings. Part of fastconformer-trt (see LICENSE and NOTICE).
"""Step-level equivalence of a TensorRT streaming encoder step against NeMo's PyTorch step (cache_aware_stream_step).

Two checks on --n utterances of a training-free set streamed as NeMo's stock script does (one batch):
  teacher-forced: every step's engine output from the stock step's own inputs (outputs and the three new caches);
  free-running: the engine carries its own caches through the whole utterance; encoder outputs compared per step.
Reported per tensor: max abs diff, max abs reference value, mean abs diff (over valid output frames), plus whether
lengths agree exactly, and the stock-fallback count (must be zero for engines covering both chunk kinds).

usage: python trt/equiv_step.py ENGINE(steady plan) --r 13 [--n 8] [--set dev_clean] [--batch 8]
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


class Acc:
    def __init__(self):
        self.d = {}

    def add(self, k, ref, got, valid=None):
        diff = (ref.float() - got.float()).abs()
        if valid is not None:
            diff = diff * valid
            n = float(valid.expand_as(diff).sum())
        else:
            n = diff.numel()
        x = self.d.setdefault(k, {"max_abs": 0.0, "ref_max": 0.0, "sum": 0.0, "n": 0.0})
        x["max_abs"] = max(x["max_abs"], float(diff.max()) if diff.numel() else 0.0)
        x["ref_max"] = max(x["ref_max"], float(ref.abs().max()) if ref.numel() else 0.0)
        x["sum"] += float(diff.sum())
        x["n"] += n

    def out(self):
        return {k: {"max_abs": v["max_abs"], "ref_max": v["ref_max"], "mean_abs": v["sum"] / max(v["n"], 1)}
                for k, v in self.d.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("engine")
    ap.add_argument("--r", type=int, required=True)
    ap.add_argument("--model", default="ml", choices=sorted(C.MODELS))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--set", default="dev_clean")
    ap.add_argument("--seed", type=int, default=C.SEED + 2)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    import gpu_reserve
    gpu_reserve.reserve(float(os.environ.get("RESERVE_HOST_GB", "6")), float(os.environ.get("RESERVE_CUDA_GB", "4")))
    C.seed_all(a.seed)
    m = C.load_model(C.MODELS[a.model], a.r)
    enc = m.encoder
    st = trt_step.build(m, a.engine, a.r)
    auds = C.dev_audios(a.set, a.n, a.seed)
    tf, fr = Acc(), Acc()
    len_ok, len_ok_fr = True, True
    cc, ct, cl = enc.get_initial_cache_state(batch_size=len(auds))
    fcc, fct, fcl = cc.clone(), ct.clone(), cl.clone()
    steps = 0
    for k, ch, ln, last in C.stock_chunks(m, auds):
        drop = 0 if k == 0 else enc.streaming_cfg.drop_extra_pre_encoded
        ref = enc.cache_aware_stream_step(processed_signal=ch, processed_signal_length=ln, cache_last_channel=cc,
                                          cache_last_time=ct, cache_last_channel_len=cl, keep_all_outputs=last,
                                          drop_extra_pre_encoded=drop)
        got = st(ch, ln, cc, ct, cl, drop, keep_all_outputs=last)
        fgot = st(ch, ln, fcc, fct, fcl, drop, keep_all_outputs=last)
        e0, el0 = ref[0], ref[1]
        valid = (torch.arange(e0.shape[-1], device=e0.device)[None] < el0[:, None])[:, None, :]
        if got[0].shape != e0.shape:
            raise SystemExit(f"shape mismatch at step {k}: {tuple(got[0].shape)} vs {tuple(e0.shape)}")
        tf.add("encoded", e0, got[0], valid)
        tf.add("cache_last_channel", ref[2], got[2])
        tf.add("cache_last_time", ref[3], got[3])
        fr.add("encoded", e0, fgot[0], valid)
        len_ok &= bool(torch.equal(el0, got[1]) and torch.equal(ref[4], got[4]))
        len_ok_fr &= bool(torch.equal(el0, fgot[1]) and torch.equal(ref[4], fgot[4]))
        cc, ct, cl = ref[2], ref[3], ref[4]
        fcc, fct, fcl = fgot[2], fgot[3], fgot[4]
        steps += 1
    torch.cuda.synchronize()
    res = {"engine": a.engine, "r": a.r, "set": a.set, "n_utts": a.n, "steps": steps, "seed": a.seed,
           "teacher_forced": tf.out(), "free_running": fr.out(), "lengths_equal_tf": len_ok,
           "lengths_equal_free": len_ok_fr, "fallbacks": st.fallbacks, "calls": st.calls}
    print("EQUIV", json.dumps(res), flush=True)
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
