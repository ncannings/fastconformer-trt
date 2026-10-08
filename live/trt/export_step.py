# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Nigel Cannings. Part of fastconformer-trt (see LICENSE and NOTICE).
# Exports NVIDIA NeMo's (Apache-2.0) cache-aware encoder step; FP8 calibration with NVIDIA TensorRT Model Optimizer.
"""P4 step 1: export the cache-aware encoder streaming step to ONNX (two graphs per chunk size: first chunk and steady
chunks), plain and with ModelOpt FP8 PTQ.

Hypothesis tested: the streaming step is a static-shape-per-chunk-size graph that TensorRT can compile with a dynamic
stream count, and FP8 PTQ calibrated on streaming chunks (with live caches) from training-free data keeps stock WER.
Ablate by selecting the stock PyTorch step in the harness (--encoder stock).

Steps: (1) the export wrapper (common.StepExport) is checked against NeMo's encoder.cache_aware_stream_step on real
dev-clean chunks with live caches (max abs diff of outputs and new caches); (2) ONNX export, opset 17, legacy
exporter as the offline project, dynamic B (axis 0 of the signal, axis 1 of the caches) and T; (3) with --quant fp8,
ModelOpt FP8_DEFAULT_CFG on the linear layers (convolutions and the subsampling module not quantised, as offline),
calibrated by running stock streaming over --calib-n utterances of --calib-set (never a test set), then the
fake-quantised step is compared with the float step and exported the same way.

usage: python trt/export_step.py --r 13 [--model ml|en] [--quant none|fp8] [--calib-set dev_clean] [--calib-n 64]
       [--recipe NAME_OR_SPEC --tag _name]   (recipes: trt/recipe.py)
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402


def run_stream(m, step_fn, audios, record=None):
    """Stock streaming over a batch with step_fn(chunk, lens, cc, ct, cl, drop) -> (e, el, cc, ct, cl)."""
    enc = m.encoder
    cc, ct, cl = enc.get_initial_cache_state(batch_size=len(audios))
    for k, ch, ln, last in C.stock_chunks(m, audios):
        drop = 0 if k == 0 else enc.streaming_cfg.drop_extra_pre_encoded
        out = step_fn(ch, ln, cc, ct, cl, drop)
        if record is not None:
            record.append((k, ch, ln, cc, ct, cl, drop, out))
        _, _, cc, ct, cl = out


def stock_step(m):
    def f(ch, ln, cc, ct, cl, drop):
        return m.encoder.cache_aware_stream_step(processed_signal=ch, processed_signal_length=ln, cache_last_channel=cc,
                                                 cache_last_time=ct, cache_last_channel_len=cl, keep_all_outputs=True,
                                                 drop_extra_pre_encoded=drop)
    return f


def wrapper_check(m, recs, tag):
    """Compare StepExport (float16 cache boundary) with the recorded stock steps, from the same inputs."""
    worst = {"enc": 0.0, "cc": 0.0, "ct": 0.0, "len": True, "cl": True}
    ref_max = 0.0
    for k, ch, ln, cc, ct, cl, drop, (e0, el0, cc0, ct0, cl0) in recs:
        w = C.StepExport(m.encoder, drop)
        e1, el1, cc1, ct1, cl1 = w(ch, ln, cc.half(), ct.half(), cl)
        valid = (torch.arange(e0.shape[-1], device=e0.device)[None] < el0[:, None])[:, None, :]
        if e0.shape == e1.shape:
            worst["enc"] = max(worst["enc"], float(((e0 - e1).abs() * valid).max()))
        else:
            worst["enc"] = float("inf")
        worst["cc"] = max(worst["cc"], float((cc0 - cc1.float()).abs().max()))
        worst["ct"] = max(worst["ct"], float((ct0 - ct1.float()).abs().max()))
        worst["len"] &= bool(torch.equal(el0, el1))
        worst["cl"] &= bool(torch.equal(cl0, cl1))
        ref_max = max(ref_max, float(e0.abs().max()))
        worst["cc_ref_max"] = max(worst.get("cc_ref_max", 0.0), float(cc0.abs().max()))
        worst["ct_ref_max"] = max(worst.get("ct_ref_max", 0.0), float(ct0.abs().max()))
    worst["enc_ref_max"] = ref_max
    print(f"WRAPPER CHECK [{tag}] (float16 cache boundary vs stock float32 step, inputs from stock):", json.dumps(worst),
          flush=True)
    return worst


def export(m, drop, width, path, g):
    """export into a scratch directory, then consolidate the weights into one external data file next to path"""
    import shutil
    import onnx
    tmp = path + ".tmpdir"
    os.makedirs(tmp, exist_ok=True)
    _export(m, drop, width, os.path.join(tmp, "model.onnx"), g)
    mo = onnx.load(os.path.join(tmp, "model.onnx"))
    data = os.path.basename(path) + ".data"
    if os.path.exists(path + ".data"):
        os.remove(path + ".data")
    onnx.save_model(mo, path, save_as_external_data=True, all_tensors_to_one_file=True, location=data,
                    size_threshold=1024)
    shutil.rmtree(tmp)
    print("exported", path, flush=True)


def _export(m, drop, width, path, g):
    w = C.StepExport(m.encoder, drop).eval()
    B = 4
    sig = torch.randn(B, g["feat_in"], width, device="cuda")
    ln = torch.full((B,), width, dtype=torch.int64, device="cuda")
    cc = torch.zeros(g["layers"], B, g["cache_len"], g["d_model"], device="cuda", dtype=torch.float16)
    ct = torch.zeros(g["layers"], B, g["d_model"], g["conv_cache"], device="cuda", dtype=torch.float16)
    cl = torch.zeros(B, dtype=torch.int64, device="cuda")
    torch.onnx.export(w, (sig, ln, cc, ct, cl), path, opset_version=17, dynamo=False,
                      input_names=["audio_signal", "length", "cache_last_channel", "cache_last_time",
                                   "cache_last_channel_len"],
                      output_names=["outputs", "encoded_lengths", "cache_last_channel_next", "cache_last_time_next",
                                    "cache_last_channel_next_len"],
                      dynamic_axes={"audio_signal": {0: "B", 2: "T"}, "length": {0: "B"},
                                    "cache_last_channel": {1: "B"}, "cache_last_time": {1: "B"},
                                    "cache_last_channel_len": {0: "B"}, "outputs": {0: "B", 2: "T2"},
                                    "encoded_lengths": {0: "B"}, "cache_last_channel_next": {1: "B"},
                                    "cache_last_time_next": {1: "B"}, "cache_last_channel_next_len": {0: "B"}})
    print("exported", path, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=int, required=True)
    ap.add_argument("--model", default="ml", choices=sorted(C.MODELS))
    ap.add_argument("--quant", default="none", choices=["none", "fp8"])
    ap.add_argument("--calib-set", default="dev_clean")
    ap.add_argument("--calib-n", type=int, default=64)
    ap.add_argument("--calib-batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--recipe", default="", help="trt/recipe.py name or spec; empty = part 1 recipe (calib flags)")
    ap.add_argument("--tag", default="", help="suffix for the FP8 file names, e.g. _p99999")
    a = ap.parse_args()
    assert "test" not in a.calib_set and (a.calib_set.startswith(("dev_", "fleurs_dev_"))), "training-free dev sets only"
    import gpu_reserve
    gpu_reserve.reserve(float(os.environ.get("RESERVE_HOST_GB", "6")), float(os.environ.get("RESERVE_CUDA_GB", "4")))
    C.seed_all(a.seed)
    m = C.load_model(C.MODELS[a.model], a.r)
    g = C.geometry(m.encoder)
    print("geometry", json.dumps(g, default=str), flush=True)
    od = os.path.join(C.OUT, f"{a.model}_r{a.r}")
    os.makedirs(od, exist_ok=True)
    json.dump(g, open(os.path.join(od, "geometry.json"), "w"), default=str, indent=1)

    chk = C.dev_audios("dev_clean", 4, a.seed + 1)            # check utterances: disjoint seed from calibration
    recs = []
    run_stream(m, stock_step(m), chk, recs)
    print("check: 4 dev-clean utterances,", len(recs), "steps", flush=True)
    if a.quant == "none":
        wrapper_check(m, recs, "fp32")
        export(m, 0, g["first_w"], os.path.join(od, "first_fp32.onnx"), g)
        export(m, g["drop"], g["steady_w"], os.path.join(od, "steady_fp32.onnx"), g)
        return

    import recipe as RC
    spec = a.recipe or f"data={a.calib_set}:{a.calib_n};act=max;batch={a.calib_batch}"   # default: part 1 recipe
    rec = RC.quantize(m, spec, lambda mm, auds: run_stream(mm, stock_step(mm), auds), a.seed,
                      log=lambda t: print(t, flush=True))
    json.dump({"recipe": rec["spec"], "seed": a.seed}, open(os.path.join(od, f"fp8{a.tag}_recipe.json"), "w"))
    wrapper_check(m, recs, "fp8 fake-quant vs float stock")
    export(m, 0, g["first_w"], os.path.join(od, f"first_fp8{a.tag}.onnx"), g)
    export(m, g["drop"], g["steady_w"], os.path.join(od, f"steady_fp8{a.tag}.onnx"), g)
    import modelopt.torch.opt as mto
    torch.save(mto.modelopt_state(m.encoder), os.path.join(od, f"fp8{a.tag}_modelopt_state.pt"))
    torch.save({k: v for k, v in m.encoder.state_dict().items() if "quantizer" in k},
               os.path.join(od, f"fp8{a.tag}_amax.pt"))


if __name__ == "__main__":
    main()
