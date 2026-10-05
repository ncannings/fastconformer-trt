"""FLEURS test split (as downloaded by fetch_data.sh) to the parquet layout the runners read: one
fleurs_<code>.parquet per language with audio {bytes, path} and text (the raw transcription; the runners normalise).
FLEURS_SPLIT=dev: the dev split (downloaded by getdev.sh to <data>/fleurs_dev) -> fleurs_dev_<code>.parquet, used
only for FP8 calibration, never scored.
usage: [FLEURS_SPLIT=dev] fleurs_to_parquet.py [CODE ...]   (default: every downloaded language)"""
import csv
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq

D = os.environ.get("ASR_DATA", "/data")
SPLIT = os.environ.get("FLEURS_SPLIT", "test")
SRC = f"{D}/fleurs" if SPLIT == "test" else f"{D}/fleurs_{SPLIT}"
codes = sys.argv[1:] or sorted(c for c in os.listdir(SRC) if os.path.exists(f"{SRC}/{c}/done"))
for c in codes:
    out = f"{D}/fleurs_{c}.parquet" if SPLIT == "test" else f"{D}/fleurs_{SPLIT}_{c}.parquet"
    if os.path.exists(out):
        continue
    rows = list(csv.reader(open(f"{SRC}/{c}/{SPLIT}.tsv", encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE))
    audio, text = [], []
    for r in rows:
        fn = f"{SRC}/{c}/{SPLIT}/{r[1]}"
        audio.append({"bytes": open(fn, "rb").read(), "path": r[1]})
        text.append(r[2])
    pq.write_table(pa.table({"audio": audio, "text": text}), out)
    print(c, len(rows), "utterances ->", out, flush=True)
