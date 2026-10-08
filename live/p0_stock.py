# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Nigel Cannings. Part of fastconformer-trt (see LICENSE and NOTICE).
"""P0: stock streaming WER. Runs NeMo's own cache-aware streaming script, unmodified,
(/opt/NeMo/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py, NeMo 3.1.0 in the
26.06 container) once per (set, chunk size) on the manifests from make_manifests.py, then scores its output with the
offline scorer (score.py). Script settings are its defaults (float32 compute, no AMP, TF32 matmuls ("high"), greedy
batched RNN-T decoding), plus target_lang, att_context_size, strip_lang_tags=true, batch_size and random_seed.

Results append to OUT/results.jsonl, one line per run, with the wall time (shared GPU: not reportable as speed).
A run whose output file already exists is skipped, so the job can be stopped and resumed.

usage: python p0_stock.py OUT --sets test_clean,fleurs_de_de --r 0,1,3,6,13 [--lang auto] [--batch 32] [--tag rep2]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

import gpu_reserve
import score

SCRIPT = "/opt/NeMo/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py"
MANI = os.path.join(os.environ.get("LIVE_OUT", "/out"), "manifests")
SEED = 20261005
CHUNK_MS = {0: 80, 1: 160, 3: 320, 6: 560, 13: 1120}
LANG = {"test_clean": "en-US", "earnings22_full": "en-US", "en_us": "en-US", "es_419": "es-US", "fr_fr": "fr-FR",
        "de_de": "de-DE", "it_it": "it-IT", "pt_br": "pt-BR", "nl_nl": "nl-NL", "ru_ru": "ru-RU", "uk_ua": "uk-UA",
        "pl_pl": "pl-PL", "sv_se": "sv-SE", "cs_cz": "cs-CZ", "da_dk": "da-DK", "bg_bg": "bg-BG", "fi_fi": "fi-FI",
        "hr_hr": "hr-HR", "sk_sk": "sk-SK", "hu_hu": "hu-HU", "ro_ro": "ro-RO", "et_ee": "et-EE"}


def set_of(manifest: str) -> str:
    """scorer set name (decides the normaliser): strip a _s<N> subset suffix."""
    head, _, tail = manifest.rpartition("_s")
    return head if head and tail.isdigit() else manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--sets", required=True)
    ap.add_argument("--r", default="0,1,3,6,13")
    ap.add_argument("--lang", default=None, help="override target_lang for every set (e.g. auto)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--tag", default="")
    ap.add_argument("--model", default="nvidia/nemotron-3.5-asr-streaming-0.6b")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    print(f"seed {SEED}", flush=True)
    for r in [int(x) for x in a.r.split(",")]:
        for m in a.sets.split(","):
            s = set_of(m)
            expect = LANG.get(s[7:] if s.startswith("fleurs_") else s)
            lang = a.lang or expect
            run = f"{m}_r{r}_{lang}{('_' + a.tag) if a.tag else ''}"
            odir = os.path.join(a.out, run)
            # the stock script names its output streaming_out_<splitext(basename(model))>_<manifest>.json
            mname = os.path.splitext(os.path.basename(a.model))[0]
            ojson = os.path.join(odir, f"streaming_out_{mname}_{m}.json")
            if os.path.exists(ojson):
                print("skip", run, flush=True)
                continue
            cmd = ["python", "/w/gpu_reserve.py", SCRIPT, f"pretrained_name={a.model}", f"dataset_manifest={MANI}/{m}.json",
                   f"batch_size={a.batch}", f"att_context_size=[56,{r}]", f"output_path={odir}",
                   f"random_seed={SEED}", "hydra.run.dir=/tmp/hydra", "hydra.output_subdir=null"]
            if "nemotron-3.5" in a.model:
                cmd += [f"target_lang={lang}", f"strip_lang_tags={'false' if lang == 'auto' else 'true'}"]
            while gpu_reserve.mem_available_gb() < 24:       # shared box: wait for headroom before a run
                print(f"waiting: MemAvailable {gpu_reserve.mem_available_gb():.1f} GB", flush=True)
                time.sleep(60)
            t0 = time.time()
            p = subprocess.run(cmd, capture_output=True, text=True)
            wall = time.time() - t0
            os.makedirs(odir, exist_ok=True)
            open(os.path.join(odir, "log.txt"), "w").write(p.stdout[-20000:] + "\n" + p.stderr[-50000:])
            if p.returncode != 0 or not os.path.exists(ojson):     # logged, left for the next pass of the queue
                print(f"FAILED {run} rc={p.returncode}: {p.stderr[-600:]}", flush=True)
                continue
            res = score.score(s, ojson, expect if lang == "auto" else None)
            res.update({"manifest": m, "r": r, "chunk_ms": CHUNK_MS[r], "target_lang": lang, "batch": a.batch,
                        "model": a.model, "tag": a.tag, "seed": SEED,
                        "wall_s_shared_gpu_not_reportable": round(wall, 1)})
            print(json.dumps(res), flush=True)
            with open(os.path.join(a.out, "results.jsonl"), "a") as f:
                f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
