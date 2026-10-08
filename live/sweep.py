# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Nigel Cannings. Part of fastconformer-trt (see LICENSE and NOTICE).
"""Concurrency sweep (pass rule: the frozen C120 rule in docs/09-live-streaming.md, frozen 6 Oct: at most 0.1% of chunks late, p95 final-token
latency within the bound; zero-miss pass is recorded alongside as pass_zero_miss).  for a quiet GPU window: for each chunk size, the largest number N of simulated real-time streams
that one GPU serves with no stream falling behind real time and p95 final-token latency within a bound.

Runs unattended in one process: load the model once; for each chunk size r (default 13, 3, 0 = 1.12 s, 320 ms,
80 ms) build the encoder step (stock NeMo or a custom engine, see harness.py ENCODER INTERFACE), warm up, then search
N: grow geometrically from --n-start until a trial fails, then bisect between the last pass and the first fail down
to --resolution. Optionally confirm the result with one longer trial (--confirm-dur-s) and step down until it holds.

A trial is harness.Server on a real-time clock: N long-form Earnings-22 streams (loadgen.make_load, shared source
audio, seeded random start segments, arrivals uniform over --stagger-s), each --dur-s long, mel computed per chunk on
the GPU, continuous batching by default (a step starts as soon as any chunk is complete and takes every complete chunk;
under load steps run back to back and batches grow by themselves). The chunk-period grid (--tick-ms -1) is available
but adds up to one chunk period of waiting to every chunk, including the last one, which inflates final-token latency. A trial PASSES when no chunk's step started after the stream's next chunk
was already complete (deadline_misses == 0, i.e. no stream fell behind real time), the run was not aborted for
backlog (--abort-backlog-s, which ends a hopeless trial early to save window time), and the p95 final-token latency
excluding the algorithmic chunk delay is at most --bound-ms. Every trial is one line in OUT/results.jsonl (with the
full harness summary: latency percentiles including and excluding the algorithmic delay, step split, memory, GPU
utilisation); each chunk size ends with a "result" line. Before each trial the script checks that MemAvailable
stays above 16 GB with the trial's cache slab added; if not, it stops that chunk size and says so.

--dry-run: tiny N (2 to 8), 20 s streams, to test the whole path on a shared GPU. Its numbers are not reportable.

usage: python sweep.py OUT [--r 13,3,0] [--encoder stock|FILE.py:build --engine PATH (may contain {r})] [--bound-ms 1000]
       [--dur-s 120] [--stagger-s 10] [--n-start 16] [--n-max 4096] [--growth 2] [--resolution 0.05]
       [--confirm-dur-s 300] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time

import numpy as np
import torch

import gpu_reserve
import harness
import loadgen

CACHE_MB_PER_STREAM = 6.3          # fp32 attention + conv caches, measured (harness summary cache_bytes_per_stream_mb)


def env_info() -> dict:
    def sh(c):
        try:
            return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception as e:
            return repr(e)
    return {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__, "mem_available_gb": round(gpu_reserve.mem_available_gb(), 1),
            "nvidia_smi_apps": sh("nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader"),
            "nvidia_smi_clocks": sh("nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,utilization.gpu,temperature.gpu "
                                    "--format=csv,noheader")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--r", default="13,3,0")
    ap.add_argument("--encoder", default="stock")
    ap.add_argument("--engine", default=None)
    ap.add_argument("--decoder", default="stock", choices=["stock", "lean", "fused"])
    ap.add_argument("--mel-graph", action="store_true")
    ap.add_argument("--cache-dtype", default="float32", choices=["float32", "float16"])
    ap.add_argument("--bound-ms", type=float, default=1000.0, help="p95 final-token latency, excl. algorithmic delay")
    ap.add_argument("--dur-s", type=float, default=120.0)
    ap.add_argument("--stagger-s", type=float, default=10.0)
    ap.add_argument("--len-spread", type=float, default=0.0,
                    help="stream lengths uniform in dur * [1-f, 1+f] (0 = all equal, as on 5 and 6 Oct; 0.5 gives "
                         "60 to 180 s for 120 s confirmations)")
    ap.add_argument("--n-start", type=int, default=16)
    ap.add_argument("--n-max", type=int, default=4096)
    ap.add_argument("--n-lo", type=int, default=0, help="resume: a known passing N (search bisects from here)")
    ap.add_argument("--n-hi", type=int, default=0, help="resume: a known failing N")
    ap.add_argument("--growth", type=float, default=2.0)
    ap.add_argument("--resolution", type=float, default=0.05, help="stop when (first fail - last pass) <= this x pass")
    ap.add_argument("--abort-backlog-s", type=float, default=3.0)
    ap.add_argument("--confirm-dur-s", type=float, default=0.0)
    ap.add_argument("--max-late", type=float, default=0.001, help="section 7: allowed fraction of late chunks")
    ap.add_argument("--late-rule", default="wait", choices=["wait", "original"],
                    help="wait (Amendment 1, default): late = waited more than one chunk duration; original: the "
                         "step started after the stream's next chunk was complete")
    ap.add_argument("--confirm-max-steps", type=int, default=5, help="steps down of the confirmation run before giving up")
    ap.add_argument("--source", default="earnings22_full")
    ap.add_argument("--lang", default="en-US")
    ap.add_argument("--mel", default="chunk", choices=["chunk", "full"])
    ap.add_argument("--clock", default="real", choices=["real", "virtual"])
    ap.add_argument("--model", default="nvidia/nemotron-3.5-asr-streaming-0.6b")
    ap.add_argument("--seed", type=int, default=20261005)
    ap.add_argument("--tick-ms", type=float, default=0.0,
                    help="0 = continuous batching (a step whenever chunks are ready, all ready chunks batched); "
                         "-1 = a grid of one chunk period (adds up to one chunk of waiting to every chunk)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fresh-process", action="store_true",
                    help="run every trial in its own process (model load and warm-up each time): found 6 Oct that "
                         "trials late in one long process run slower than the same trial in a fresh process")
    ap.add_argument("--crash-as-fail", action="store_true",
                    help="fresh process: a trial whose process crashes is logged (kind crash) and counted as a fail, "
                         "and the search goes on (default: the sweep stops)")
    ap.add_argument("--deadline-epoch", type=float, default=0.0,
                    help="unix time after which no new trial starts; the arm ends with what it has (resumable runs)")
    ap.add_argument("--one-trial", default=None, help=argparse.SUPPRESS)      # internal: "n,dur,r"
    a = ap.parse_args()
    if a.dry_run:
        a.n_start, a.n_max, a.dur_s, a.stagger_s, a.resolution = 2, 8, 20.0, 3.0, 0.5
    random.seed(a.seed)
    np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    res_path = os.path.join(a.out, "results.jsonl")

    def log(rec: dict) -> None:
        print(json.dumps(rec), flush=True)
        with open(res_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    rs = [int(x) for x in a.r.split(",")]
    one = [float(x) for x in a.one_trial.split(",")] if a.one_trial else None
    if one:
        rs = [int(one[2])]
    T0 = time.time()
    m = None if (a.fresh_process and not one) else harness.load_model(a.model, rs[0], a.lang, a.seed)
    print(f"[timing] model loaded {time.time() - T0:.1f} s", file=sys.stderr, flush=True)
    sources, _ = loadgen.make_load(a.source, 1, a.seed, 0, 1)
    pool_gb = sum(len(x) for x in sources) * 4 / 1e9
    if not one:
        log({"kind": "start", "args": vars(a), "seed": a.seed, "n_sources": len(sources), "source_audio_h":
             round(sum(len(x) for x in sources) / 16000 / 3600, 2), "env": env_info(),
             "note": "dry run on a shared GPU: not reportable" if a.dry_run else "reportable only if the window was quiet"})

    for r in rs:
        engine = a.engine.replace("{r}", str(r)) if a.engine else None      # per-chunk-size engines: ml_r{r}/...
        if m is not None:
            harness.set_chunk(m, r)
            enc = harness.make_encoder_step(m, a.encoder, engine, r)
        tick = harness.CHUNK_MS[r] if a.tick_ms < 0 else a.tick_ms
        chunk_ms = harness.CHUNK_MS[r]

        def trial(n: int, dur: float, record: bool = True, clock: str | None = None) -> bool:
            if a.deadline_epoch and record and not one and time.time() + dur + 60 > a.deadline_epoch:
                raise TimeoutError(f"deadline: no time for N={n} ({dur:.0f} s) before {a.deadline_epoch:.0f}")
            per = CACHE_MB_PER_STREAM / (2 if a.cache_dtype == "float16" else 1)
            need = n * per / 1e3 + pool_gb + 2.0 + (6.0 if m is None else 0.0)    # fresh process: + model, pool
            avail = gpu_reserve.mem_available_gb()
            if avail - need < 16.0:
                raise MemoryError(f"N={n} needs about {need:.1f} GB; MemAvailable {avail:.1f} GB would go below 16 GB")
            if m is None:                                     # fresh process: this script, one trial, same args
                if not record:
                    return True                               # the child warms up itself
                argv = [x for x in sys.argv[1:] if x != "--fresh-process"]
                p = subprocess.run([sys.executable, os.path.abspath(__file__)] + argv +
                                   ["--one-trial", f"{n},{dur},{r}"], capture_output=True, text=True)
                with open(os.path.join(a.out, "trial_logs.txt"), "a") as f:
                    f.write(f"=== N={n} dur={dur} r={r} rc={p.returncode}\n" + p.stderr[-4000:] + "\n")
                last = [json.loads(l) for l in open(res_path) if '"kind": "trial"' in l][-1:]
                if p.returncode != 0 or not last or last[0]["n"] != n or last[0]["dur_s"] != dur:
                    if a.crash_as_fail:
                        log({"kind": "crash", "r": r, "n": n, "dur_s": dur, "rc": p.returncode,
                             "stderr_tail": p.stderr[-1500:]})
                        return False
                    raise RuntimeError(f"fresh-process trial N={n} failed rc={p.returncode}: {p.stderr[-1500:]}")
                return last[0]["pass"]
            _, streams = loadgen.make_load(a.source, n, a.seed + n, a.stagger_s, dur, len_spread=a.len_spread)
            out = os.path.join(a.out, f"trials/r{r}_n{n}_d{int(dur)}") if record else None
            t0 = time.time()
            s = harness.run(out, m, sources, streams, [a.lang] * n, a.mel, clock or a.clock, r, a.seed,
                            {"encoder": a.encoder, "engine": engine, "dur_s": dur, "stagger_s": a.stagger_s},
                            tick, enc, a.abort_backlog_s, write_steps=True, cache_dtype=a.cache_dtype, decoder=a.decoder, mel_graph=a.mel_graph)
            p95 = s["final_token_latency_ms_excl_algo"]["p95"]
            late_orig = s["deadline_misses"] / max(1, s["n_chunks"])
            # Amendment 1 (6 Oct): late = step started more than one chunk duration after the chunk was ready
            late = (s["late_by_wait_over_one_chunk"] / max(1, s["n_chunks"])) if a.late_rule == "wait" else late_orig
            n_late = s["late_by_wait_over_one_chunk"] if a.late_rule == "wait" else s["deadline_misses"]
            zero_ok = s["aborted"] is None and n_late == 0 and p95 is not None and p95 <= a.bound_ms
            # section 7 (frozen 6 Oct): pass = at most 0.1% late chunks, p95 within the bound, not aborted
            ok = s["aborted"] is None and late <= a.max_late and p95 is not None and p95 <= a.bound_ms
            if record:
                log({"kind": "trial", "r": r, "chunk_ms": chunk_ms, "tick_ms": tick, "n": n, "dur_s": dur, "pass": ok, "pass_zero_miss": zero_ok,
                     "late_frac": late, "late_rule": a.late_rule, "late_frac_original_rule": late_orig,
                     "wall_s": round(time.time() - t0, 1), "mem_available_gb": round(gpu_reserve.mem_available_gb(), 1),
                     "summary": s})
            return ok

        if one:                                               # child: warm up, one recorded trial, exit
            print(f"[timing] encoder built {time.time() - T0:.1f} s", file=sys.stderr, flush=True)
            trial(min(a.n_start, 4), 5.0, record=False, clock="virtual")
            print(f"[timing] warm-up done {time.time() - T0:.1f} s", file=sys.stderr, flush=True)
            trial(int(one[0]), one[1])
            return
        lo, hi = 0, None
        try:
            trial(min(a.n_start, 4), 10.0, record=False, clock="virtual")     # warm-up: kernels, graphs, allocator
            lo, hi, n = 0, None, a.n_start
            if a.n_lo and a.n_hi:                     # resume a search: bisect between a known pass and fail
                lo, hi, n = a.n_lo, a.n_hi, (a.n_lo + a.n_hi) // 2
            while True:
                if trial(n, a.dur_s):
                    lo = n
                    if hi is None:
                        if n >= a.n_max:
                            break
                        n = min(a.n_max, max(n + 1, int(n * a.growth)))
                        continue
                else:
                    hi = n
                if hi is not None and (hi - lo <= max(1, int(lo * a.resolution)) or hi - lo <= 1):
                    break
                n = (lo + hi) // 2 if lo else max(1, hi // 2)
                if n == lo or n == hi:
                    break
            confirmed, confirm_note = None, None
            if a.confirm_dur_s > 0 and lo > 0:
                c, steps = lo, 0
                while True:
                    if trial(c, a.confirm_dur_s):
                        confirmed = c
                        break
                    steps += 1
                    if steps > a.confirm_max_steps:
                        confirm_note = f"no confirmation within {a.confirm_max_steps} steps down; last failed N={c}"
                        break
                    c = max(1, int(c * 0.9))            # section 7: step down 10%
            log({"kind": "result", "r": r, "chunk_ms": chunk_ms, "encoder": a.encoder, "max_streams": lo,
                 "first_fail": hi, "confirmed_max_streams": confirmed, "confirm_note": confirm_note, "bound_ms": a.bound_ms, "dur_s": a.dur_s,
                 "hit_n_max": hi is None})
        except (MemoryError, TimeoutError) as e:
            # keep what the search established: lo passed, the memory guard or the deadline stopped the next trial
            log({"kind": "result", "r": r, "chunk_ms": chunk_ms, "encoder": a.encoder, "max_streams": lo,
                 "first_fail": hi, "confirmed_max_streams": None,
                 "stopped": f"{'memory' if isinstance(e, MemoryError) else 'deadline'}: {e}"})
    log({"kind": "end", "env": env_info()})


if __name__ == "__main__":
    main()
