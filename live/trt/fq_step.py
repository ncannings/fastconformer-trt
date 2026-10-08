# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Nigel Cannings. Part of fastconformer-trt (see LICENSE and NOTICE).
"""Fake-quantised FP8 encoder step for the harness: screens a calibration recipe (trt/recipe.py) by WER without an
ONNX export or a TensorRT build. The model's own encoder is quantised in place with ModelOpt (fake quantisation:
FP8 rounding of the linear-layer inputs and weights, float32 arithmetic elsewhere), calibrated with the recipe at
the model's chunk size, then the stock NeMo step is returned. A real TensorRT engine computes the non-quantised parts
in FP16, so the screen ranks recipes; the chosen recipe is then confirmed with an engine (trt_step.py).

Hypothesis and ablation as in recipe.py. Harness usage:
  --encoder /w/trt/fq_step.py:build --engine RECIPE        (RECIPE: a recipe.py name or spec string)
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import recipe as RC  # noqa: E402


def _run_stream(m, audios):
    enc = m.encoder
    cc, ct, cl = enc.get_initial_cache_state(batch_size=len(audios))
    for k, ch, ln, last in C.stock_chunks(m, audios):
        drop = 0 if k == 0 else enc.streaming_cfg.drop_extra_pre_encoded
        _, _, cc, ct, cl = enc.cache_aware_stream_step(
            processed_signal=ch, processed_signal_length=ln, cache_last_channel=cc, cache_last_time=ct,
            cache_last_channel_len=cl, keep_all_outputs=True, drop_extra_pre_encoded=drop)


def build(model, engine: str, r: int):
    """live/harness.py ENCODER INTERFACE entry point; engine is the recipe."""
    with torch.no_grad():
        rec = RC.quantize(model, engine, _run_stream, C.SEED, log=lambda t: print("fq_step:", t, flush=True))
    print(f"fq_step: r={r}, recipe {rec['spec']!r}", flush=True)

    def enc_step(sig, lens, cc, ct, cl, drop):
        dt = cc.dtype
        out = model.encoder.cache_aware_stream_step(
            processed_signal=sig, processed_signal_length=lens, cache_last_channel=cc.float(),
            cache_last_time=ct.float(), cache_last_channel_len=cl, keep_all_outputs=False,
            drop_extra_pre_encoded=drop)
        return out[0], out[1], out[2].to(dt), out[3].to(dt), out[4]
    return enc_step
