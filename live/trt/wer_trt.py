# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Nigel Cannings. Part of fastconformer-trt (see LICENSE and NOTICE).
"""WER of a TensorRT encoder step through NeMo's own cache-aware streaming script, scored as P0 stock is.

Runs /opt/NeMo/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py unmodified in
this process, with ConformerEncoder.cache_aware_stream_step routed through the engine (trt_step.patch_class), with the
same arguments as live/p0_stock.py (batch 32, seed 20261005, att_context_size [56, r], target_lang, strip_lang_tags),
then scores with live/score.py. Everything outside the encoder step (buffer, prompt, RNN-T decoding) is stock.

usage: python trt/wer_trt.py ENGINE(steady plan) --r 13 --set test_clean OUT_DIR [--lang en-US] [--batch 32]
"""
from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SCRIPT = "/opt/NeMo/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py"
SEED = 20261005


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("engine")
    ap.add_argument("out")
    ap.add_argument("--r", type=int, required=True)
    ap.add_argument("--set", default="test_clean")
    ap.add_argument("--lang", default="en-US")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--model", default="nvidia/nemotron-3.5-asr-streaming-0.6b")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    import gpu_reserve
    gpu_reserve.reserve(float(os.environ.get("RESERVE_HOST_GB", "6")), float(os.environ.get("RESERVE_CUDA_GB", "4")))
    import score
    import trt_step
    trt_step.patch_class(a.engine)
    mani = os.path.join(os.environ.get("LIVE_OUT", "/out"), "manifests", f"{a.set}.json")
    run = f"{a.set}_r{a.r}_{a.lang}_{a.tag or os.path.basename(a.engine)[:-5]}"
    odir = os.path.join(a.out, run)
    args = [f"pretrained_name={a.model}", f"dataset_manifest={mani}", f"batch_size={a.batch}",
            f"att_context_size=[56,{a.r}]", f"output_path={odir}", f"random_seed={SEED}", "hydra.run.dir=/tmp/hydra",
            "hydra.output_subdir=null"]
    if "nemotron-3.5" in a.model:
        args += [f"target_lang={a.lang}", "strip_lang_tags=true"]
    print("seed", SEED, "run", run, "args", args, flush=True)
    t0 = time.time()
    sys.argv = [SCRIPT] + args
    try:
        runpy.run_path(SCRIPT, run_name="__main__")
    except SystemExit as e:
        if e.code not in (None, 0):
            raise
    wall = time.time() - t0
    mname = os.path.splitext(os.path.basename(a.model))[0]
    ojson = os.path.join(odir, f"streaming_out_{mname}_{a.set}.json")
    head, _, tail = a.set.rpartition("_s")
    sname = head if head and tail.isdigit() else a.set
    res = score.score(sname, ojson)
    rep = [s.report() for s in trt_step.STEPS]
    falls = sum(s.fallbacks for s in trt_step.STEPS)
    res.update({"manifest": a.set, "r": a.r, "engine": a.engine, "target_lang": a.lang, "batch": a.batch,
                "seed": SEED, "steps": rep, "fallbacks": falls, "wall_s_shared_gpu_not_reportable": round(wall, 1)})
    print("WER", json.dumps(res), flush=True)
    with open(os.path.join(a.out, "results.jsonl"), "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
