# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Nigel Cannings. Part of fastconformer-trt (see LICENSE and NOTICE).
"""P1 correctness test: the live harness must produce the same transcripts as NeMo's own cache-aware streaming
script for the same audio and chunk size.

Builds the harness's long-form streams (loadgen.py), writes them as wav files plus a manifest, runs the stock script
(unmodified, subprocess) on them twice (all streams in one batch, and one stream per batch), then runs the harness
in-process (virtual clock, staggered joins and leaves) three ways: --mel full and --mel chunk on the chunk-period grid
(streams batched together), and --mel chunk eager (each stream mostly stepped alone). Reports, for every pair of
outputs, how many stream transcripts are identical strings and the word-level difference (WER of one output against
the other, normalised), plus each output's WER against the reference text.

usage: python equiv.py OUT_DIR --r 3 --n 8 [--source earnings22_full] [--max-s 120] [--stagger-s 30] [--lang en-US]
       [--encoder FILE.py:build --engine PATH] [--skip-stock]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess

import jiwer
import soundfile as sf

import finetune_stride as fs
import gpu_reserve
import harness
import loadgen

SCRIPT = "/opt/NeMo/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py"


def wait_mem(gb: float = 24.0) -> None:
    """shared box: wait for headroom before each GPU stage."""
    import time
    while gpu_reserve.mem_available_gb() < gb:
        time.sleep(30)


def stock(model: str, manifest: str, out: str, r: int, batch: int, lang: str, seed: int) -> list[str]:
    cmd = ["python", "/w/gpu_reserve.py", SCRIPT, f"pretrained_name={model}", f"dataset_manifest={manifest}", f"batch_size={batch}",
           f"att_context_size=[56,{r}]", f"output_path={out}", f"random_seed={seed}", "hydra.run.dir=/tmp/hydra",
           "hydra.output_subdir=null", f"target_lang={lang}", "strip_lang_tags=true"]
    wait_mem()
    p = subprocess.run(cmd, capture_output=True, text=True)
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "log.txt"), "w").write(p.stdout[-20000:] + "\n" + p.stderr[-50000:])
    js = [f for f in os.listdir(out) if f.startswith("streaming_out_")]
    if p.returncode != 0 or not js:
        raise RuntimeError(f"stock run failed rc={p.returncode}: {p.stderr[-3000:]}")
    return [json.loads(l)["pred_text"] for l in open(os.path.join(out, js[0]))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--r", type=int, default=3)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--source", default="earnings22_full")
    ap.add_argument("--max-s", type=float, default=120)
    ap.add_argument("--stagger-s", type=float, default=30)
    ap.add_argument("--lang", default="en-US")
    ap.add_argument("--model", default="nvidia/nemotron-3.5-asr-streaming-0.6b")
    ap.add_argument("--seed", type=int, default=20261005)
    ap.add_argument("--encoder", default="stock", help="harness encoder step: stock or FILE.py:build")
    ap.add_argument("--engine", default=None)
    ap.add_argument("--decoder", default="stock", choices=["stock", "lean", "fused"])
    ap.add_argument("--mel-graph", action="store_true")
    ap.add_argument("--cache-dtype", default="float32", choices=["float32", "float16"])
    ap.add_argument("--skip-stock", action="store_true", help="reuse stock outputs already in OUT_DIR")
    a = ap.parse_args()
    print(f"seed {a.seed}", flush=True)
    os.makedirs(f"{a.out}/wav", exist_ok=True)
    specs = loadgen.make_streams(a.source, a.n, a.seed, a.stagger_s, a.max_s)
    man = f"{a.out}/manifest.json"
    with open(man, "w") as f:
        for s in specs:
            p = f"{a.out}/wav/{s.sid:04d}.wav"
            sf.write(p, s.audio, 16000, subtype="FLOAT")
            f.write(json.dumps({"audio_filepath": p, "text": s.text, "duration": s.duration_s}, ensure_ascii=False) + "\n")
    outs = {}
    for key, b in (("stock_batch_all", a.n), ("stock_batch_1", 1)):
        d = f"{a.out}/stock_b{b}"
        js = [f for f in os.listdir(d) if f.startswith("streaming_out_")] if os.path.isdir(d) else []
        if a.skip_stock and js:
            outs[key] = [json.loads(l)["pred_text"] for l in open(os.path.join(d, js[0]))]
        else:
            outs[key] = stock(a.model, man, d, a.r, b, a.lang, a.seed)
    wait_mem()
    m = harness.load_model(a.model, a.r, a.lang, a.seed)
    enc = harness.make_encoder_step(m, a.encoder, a.engine, a.r)
    sources, streams = loadgen.from_specs(specs)
    tick = harness.CHUNK_MS[a.r]
    for mel, t in (("full", tick), ("chunk", tick), ("chunk", 0)):
        name = f"harness_mel_{mel}_" + ("grid" if t else "eager")
        d = f"{a.out}/{name}"
        harness.run(d, m, sources, streams, [a.lang] * len(streams), mel, "virtual", a.r, a.seed,
                    {"source": a.source, "encoder": a.encoder}, t, enc, cache_dtype=a.cache_dtype, decoder=a.decoder, mel_graph=a.mel_graph)
        rec = sorted((json.loads(l) for l in open(f"{d}/transcripts.jsonl")), key=lambda x: x["sid"])
        outs[name] = [x["pred_text"] for x in rec]
    norm = fs.normaliser_for(a.source)
    refs = [norm(s.text) for s in specs]
    res = {"r": a.r, "n": a.n, "source": a.source, "max_s": a.max_s, "stagger_s": a.stagger_s, "seed": a.seed,
           "audio_s": sum(s.duration_s for s in specs), "wer_vs_reference": {}, "pairs": {}}
    for k, v in outs.items():
        res["wer_vs_reference"][k] = 100 * jiwer.wer(refs, [norm(h) or "<empty>" for h in v])
    keys = list(outs)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            x, y = outs[keys[i]], outs[keys[j]]
            same = sum(p == q for p, q in zip(x, y))
            diff = 100 * jiwer.wer([norm(p) or "<empty>" for p in x], [norm(q) or "<empty>" for q in y])
            res["pairs"][f"{keys[i]} vs {keys[j]}"] = {"identical_streams": f"{same}/{len(x)}",
                                                       "word_diff_pct": diff}
    json.dump({"result": res, "transcripts": outs}, open(f"{a.out}/equiv.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
