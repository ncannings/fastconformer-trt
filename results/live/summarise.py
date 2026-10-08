# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Nigel Cannings. Part of fastconformer-trt (see LICENSE and NOTICE).
"""Rebuild the live-streaming summary tables (c120_spark.json, c120_h100.json, nvidia_method_h100.json) from the raw
per-trial lines in raw/ (each line is one trial written by live/sweep.py, with the harness summary).

Rules (docs/09-live-streaming.md, section 3; frozen 6 October 2026 and amended once, before the final runs):
  a trial passes when it was not aborted, at most 0.1% of its chunks were late (a chunk is late when its step started
  more than one chunk duration after the chunk was ready) and its p95 final-token latency, excluding the chunk delay,
  is at most 1,000 ms (sweep.py writes this as "pass");
  C120 = the largest N at which every trial passed and at least one trial was 120 s long;
  A = the largest N passing a 60 s trial with zero late chunks (and the latency bound); B = the same for 120 s.
The C120 computed here is checked against the sweep's own "confirmed_max_streams" and the script stops on a mismatch.

usage: python results/live/summarise.py        (writes the three JSON files next to this script)
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")


def lines(path: str) -> list[dict]:
    return [json.loads(x) for x in open(path) if x.strip()]


def arm_summary(path: str) -> dict:
    rows = lines(path)
    start = next(r for r in rows if r["kind"] == "start")
    trials = [r for r in rows if r["kind"] == "trial"]
    crashes = [r for r in rows if r["kind"] == "crash"]
    result = [r for r in rows if r["kind"] == "result"][-1]
    by_n = defaultdict(list)
    for t in trials:
        by_n[t["n"]].append(t)
    for c in crashes:                                   # a crashed trial counts as a failed trial (--crash-as-fail)
        by_n[c["n"]].append({"n": c["n"], "dur_s": c["dur_s"], "pass": False, "pass_zero_miss": False, "crash": True})
    c120 = [n for n, ts in by_n.items() if all(t["pass"] for t in ts) and any(t["dur_s"] >= 120 for t in ts)]
    c120 = max(c120) if c120 else None
    a_ = [t["n"] for t in trials if t["dur_s"] < 120 and t["pass_zero_miss"]]
    b_ = [t["n"] for t in trials if t["dur_s"] >= 120 and t["pass_zero_miss"]]
    conf = result.get("confirmed_max_streams")
    conf = int(conf) if conf is not None else None
    if conf != c120:
        raise SystemExit(f"{path}: computed C120 {c120} differs from the sweep's confirmed_max_streams {conf}")
    s0 = trials[0]["summary"]
    out = {
        "file": os.path.relpath(path, HERE),
        "gpu": start["env"]["gpu"],
        "chunk_ms": trials[0]["chunk_ms"],
        "encoder": "NeMo (stock)" if start["args"]["encoder"] == "stock" else start["args"]["engine"].split("/")[-1],
        "decoder": start["args"]["decoder"],
        "mel_graph": start["args"]["mel_graph"],
        "cache_dtype": start["args"]["cache_dtype"],
        "gc_mode": s0.get("gc_mode"),
        "seed": start["seed"],
        "stagger_s": start["args"]["stagger_s"],
        "fresh_process": start["args"]["fresh_process"],
        "late_rule": start["args"]["late_rule"],
        "C120": c120,
        "A_60s_zero_late": max(a_) if a_ else None,
        "B_120s_zero_late": max(b_) if b_ else None,
        "search_stopped": result.get("stopped"),
        "trials": [],
    }
    for t in trials:
        s = t["summary"]
        ft = s.get("final_token_latency_ms_excl_algo") or {}
        out["trials"].append({
            "n": t["n"], "dur_s": t["dur_s"], "pass": t["pass"], "chunks": s["n_chunks"],
            "late_chunks": s["late_by_wait_over_one_chunk"], "late_pct": round(100 * t["late_frac"], 4),
            "final_token_ms_p50": None if ft.get("p50") is None else round(ft["p50"], 1),
            "final_token_ms_p95": None if ft.get("p95") is None else round(ft["p95"], 1),
            "aborted": s.get("aborted"),
            "step_ms_mean_split": {k: round(v, 1) for k, v in (s.get("step_split_ms_mean") or {}).items()},
            "gpu_util_mean_pct": None if s.get("gpu_util_mean_pct") is None else round(s["gpu_util_mean_pct"], 1),
            "peak_gpu_mem_allocated_gb": round(s["peak_gpu_mem_allocated_gb"], 2),
            "decoder_graph_recaptures": len(s.get("decoder_graph_reinits") or []),
        })
    for c in crashes:
        out["trials"].append({"n": c["n"], "dur_s": c["dur_s"], "pass": False, "crash": True, "rc": c.get("rc")})
    if c120 is not None:
        t120 = [t for t in out["trials"] if t["n"] == c120 and t["dur_s"] >= 120][-1]
        out["at_C120_120s"] = {k: t120[k] for k in ("chunks", "late_chunks", "final_token_ms_p50", "final_token_ms_p95")}
    return out


def curve(path: str) -> list[dict]:
    rows = []
    for t in lines(path):
        if t["kind"] != "trial":
            continue
        s = t["summary"]
        ft = s.get("final_token_latency_ms_excl_algo") or {}
        rows.append({"n": t["n"], "dur_s": t["dur_s"], "stagger_s": s.get("stagger_s"),
                     "encoder": "NeMo (stock)" if s["encoder"] == "stock" else (s.get("engine") or "").split("/")[-1],
                     "decoder": s["decoder"], "gc_mode": s.get("gc_mode"), "mel_graph": s.get("mel_graph"),
                     "keeps_up": s.get("aborted") is None and s["late_by_wait_over_one_chunk"] == 0,
                     "late_chunks": s["late_by_wait_over_one_chunk"], "chunks": s["n_chunks"],
                     "median_final_token_ms": None if ft.get("p50") is None else round(ft["p50"], 1),
                     "step_ms_p50": round(s["step_wall_ms"]["p50"], 1), "step_ms_max": round(s["step_wall_ms"]["max"], 1),
                     "aborted": s.get("aborted")})
    return rows


def main() -> None:
    spark = {}
    for arm in ("stock", "fp16", "fp8"):
        for r in (0, 3, 13):
            spark[f"{arm}_r{r}"] = arm_summary(os.path.join(RAW, "spark_20261007", f"{arm}_r{r}.jsonl"))
    h100 = {}
    for name in ("sweep_stockgc_r0", "sweep_stockgc_r13"):
        h100[name.replace("sweep_", "")] = arm_summary(os.path.join(RAW, "h100_20261007_v5", f"{name}.jsonl"))
    for name in ("sweep_fp16fusedmg_r0", "sweep_fp16fused_r13", "sweep_fp8fused_r13"):
        h100[name.replace("sweep_", "")] = arm_summary(os.path.join(RAW, "h100_20261007_v4", f"{name}.jsonl"))
    v6 = os.path.join(RAW, "h100_20261008_v6", "sweep_stockgc_pw_r0.jsonl")
    if os.path.exists(v6):
        h100["stockgc_pw_r0"] = arm_summary(v6)
    v7 = os.path.join(RAW, "h100_20261008_v7", "sweep_fp8fusedmg_r0.jsonl")
    if os.path.exists(v7):
        h100["fp8fusedmg_r0"] = arm_summary(v7)
    nv = {}
    d = os.path.join(RAW, "h100_nvidia_method")
    for f in sorted(os.listdir(d)):
        nv[f.replace(".jsonl", "")] = curve(os.path.join(d, f))
    for name, obj in (("c120_spark.json", spark), ("c120_h100.json", h100), ("nvidia_method_h100.json", nv)):
        with open(os.path.join(HERE, name), "w") as f:
            json.dump(obj, f, indent=1)
            f.write("\n")
    for k, v in {**spark, **h100}.items():
        print(f"{v['gpu'][:12]:12s} {k:18s} {v['chunk_ms']:5d} ms  C120 {v['C120']}  A {v['A_60s_zero_late']}  "
              f"B {v['B_120s_zero_late']}  at C120 {v.get('at_C120_120s')}")


if __name__ == "__main__":
    main()
