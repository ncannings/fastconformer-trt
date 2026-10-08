# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Nigel Cannings. Part of fastconformer-trt (see LICENSE and NOTICE).
"""P0 data: write each evaluation set as 16 kHz mono wav files plus a NeMo manifest, so that NeMo's own cache-aware
streaming script (examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py) can run on
it unmodified. Audio goes through the offline scorer's load_audio (finetune_stride.py), so the samples are the ones
the offline numbers were scored on. 16-bit mono 16 kHz sources are written back as PCM_16 (lossless); anything else (stereo, resampled) as float.

Manifests are sorted by duration (shorter first) so batches waste little padding; each line keeps `row`, the index
in the source parquet, for scoring. A seeded subset manifest (<set>_s<N>.json, seed 20261005, the offline shuffle)
is written too.

usage: python make_manifests.py test_clean earnings22_full fleurs_de_de ... [--subset 200]
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random

import numpy as np
import soundfile as sf

import finetune_stride as fs

OUT = os.environ.get("LIVE_OUT", os.path.expanduser("~/asr_data/live"))
SEED = 20261005


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="+")
    ap.add_argument("--subset", type=int, default=200)
    a = ap.parse_args()
    print(f"seed {SEED}", flush=True)
    os.makedirs(f"{OUT}/manifests", exist_ok=True)
    for s in a.sets:
        rows = fs.rows_of(s)
        wdir = f"{OUT}/wav/{s}"
        os.makedirs(wdir, exist_ok=True)
        items = []
        for i, r in enumerate(rows):
            path = f"{wdir}/{i:05d}.wav"
            info = sf.info(io.BytesIO(r["audio"]["bytes"]))
            if not os.path.exists(path):
                w = fs.load_audio(r)
                sub = "PCM_16" if (info.subtype == "PCM_16" and info.samplerate == 16000 and info.channels == 1) else "FLOAT"
                if sub == "PCM_16":
                    assert np.allclose(np.round(w * 32768), w * 32768, atol=1e-3), f"{s} row {i} not 16-bit"
                sf.write(path, w, 16000, subtype=sub)
            dur = sf.info(path).duration
            items.append({"audio_filepath": path, "text": fs.ref_text(r), "duration": dur, "row": i})
        idx = list(range(len(items)))
        random.Random(SEED).shuffle(idx)
        for name, sel in ((s, items), (f"{s}_s{a.subset}", [items[k] for k in idx[:a.subset]])):
            with open(f"{OUT}/manifests/{name}.json", "w") as f:
                for it in sorted(sel, key=lambda x: (x["duration"], x["row"])):
                    f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(s, len(items), "utts", round(sum(x["duration"] for x in items) / 3600, 2), "h", flush=True)


if __name__ == "__main__":
    main()
