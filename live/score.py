# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Nigel Cannings. Part of fastconformer-trt (see LICENSE and NOTICE).
"""Score a NeMo cache-aware streaming output file (streaming_out_*.json, one line per manifest entry, same order) with
the offline project's scorer: Whisper English normaliser for English sets, Whisper basic normaliser elsewhere
(finetune_stride.normaliser_for), corpus WER with jiwer, empty hypotheses scored as "<empty>", empty references
dropped. Language tags of the form <xx-XX> are always removed from the hypothesis before scoring; with --expect-lang
the tag the model emitted is checked against the expected one (auto-detect runs).

usage: python score.py SET OUT_JSON [--expect-lang de-DE] [--result result.json]
"""
from __future__ import annotations

import argparse
import json
import re

import jiwer

import finetune_stride as fs

TAG = re.compile(r"\s*<([a-z]{2}-[A-Z]{2})>")


def score(set_name: str, path: str, expect_lang: str | None = None) -> dict:
    norm = fs.normaliser_for(set_name)
    recs = [json.loads(l) for l in open(path)]
    refs, hyps, tags = [], [], []
    for r in recs:
        h = r["pred_text"] or ""
        tags.append(TAG.findall(h))
        h = TAG.sub(" ", h)
        rn, hn = norm(r["text"]), norm(h) or "<empty>"
        if rn:
            refs.append(rn)
            hyps.append(hn)
    out = {"set": set_name, "n": len(recs), "n_scored": len(refs), "wer": 100 * jiwer.wer(refs, hyps),
           "n_empty_hyp": sum(h == "<empty>" for h in hyps)}
    if expect_lang:
        out["lang_tag_correct"] = sum(1 for t in tags if t and t[-1] == expect_lang) / len(tags)
        out["lang_tag_missing"] = sum(1 for t in tags if not t) / len(tags)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("set")
    ap.add_argument("out_json")
    ap.add_argument("--expect-lang")
    ap.add_argument("--result")
    a = ap.parse_args()
    res = score(a.set, a.out_json, a.expect_lang)
    print(json.dumps(res), flush=True)
    if a.result:
        json.dump(res, open(a.result, "w"), indent=1)


if __name__ == "__main__":
    main()
