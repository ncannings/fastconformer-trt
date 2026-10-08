# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Nigel Cannings. Part of fastconformer-trt (see LICENSE and NOTICE).
"""Live demo for video: N Earnings-22 calls stream in real time through the live harness, with a terminal dashboard
(rich) showing the clock, live streams, step time, late chunks, how far the server is behind real time, chunk and
final-token latency, and the transcripts of four streams scrolling as tokens come out.

Acts (--acts, comma list of ARM:R:N, run one after the other in one process, same load per chunk size):
  stock  NeMo as shipped with gc.freeze (the strongest fair stock), NeMo's own decoder
  ours   TensorRT encoder engine (--engines; default FP8) + float16 caches + mel graph + fused decoder (--decoder fused, FUSED_JOINT and
         FUSED_LSTM from the environment; default triton)
  R = 0, 3 or 13 (80 ms, 320 ms, 1.12 s chunks), N = streams.
Before each act's clock starts the server is fully prepared (the "warming up" panel): the fused decoder captures all
its CUDA graphs in the Server constructor, and Server.run warms the encoder step and mel graph at every reachable
batch size before the first sample arrives. An act ends at --act-s (stock acts that fall behind keep running to that
time, showing "behind real time" grow). The display runs between steps, at most --fps times a second; for ours it
detokenises only the four displayed streams. NeMo, TensorRT and Python log output go to --log (file descriptors 1 and
2 are redirected); only the dashboard reaches the terminal. Numbers on screen from a shared GPU are not reportable;
the concurrency claims come from the sweeps (docs/09-live-streaming.md).

usage: python demo_live.py [--acts stock:0:232,ours:0:232,...] [--act-s 45] [--show 4] [--log /out/demo_live.log]
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from collections import deque

import numpy as np
import torch
from rich.console import Console, Group
from rich.live import Live as RLive
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

import harness
import loadgen

ENG = "/out/trt/ml_r{r}/steady_fp8_b1024.plan"


class Dash:
    def __init__(self, title: str, srv: harness.Server, n_show: int, fps: float, chunk_ms: int, n: int,
                 caption: str = ""):
        self.title, self.srv, self.n_show, self.dt = title, srv, n_show, 1.0 / fps
        self.caption = caption or None
        self.chunk_ms, self.n = chunk_ms, n
        self.last = 0.0
        self.shown: list = []
        self.walls = deque(maxlen=200)
        self.n_steps_seen = 0
        self.live = None
        self.peak_lag = 0.0

    def _text(self, s) -> str:
        srv = self.srv
        if srv.lean is not None:
            return srv.lean.text(s.slot)[0] if s.step else ""
        return s.hyp.text if s.hyp is not None else ""

    def _pick(self):
        live = [s for s in self.srv.live if not s.done]
        keep = [s for s in self.shown if s in live]
        for s in live:
            if len(keep) >= self.n_show:
                break
            if s not in keep:
                keep.append(s)
        self.shown = keep

    def render(self):
        srv = self.srv
        now = srv.now()
        for x in srv.steps[self.n_steps_seen:]:
            self.walls.append(x["wall_s"] * 1000)
        self.n_steps_seen = len(srv.steps)
        rts = [srv._ready_time(s) for s in srv.live]
        waiting = [t for t in rts if t <= now]
        lag = (now - min(waiting)) if waiting else 0.0
        self.peak_lag = max(self.peak_lag, lag)
        late = total = 0
        recent = []
        for s in srv.live + srv.finished[-200:]:
            ch = s.chunks
            for k in range(max(0, len(ch) - 40), len(ch)):
                rt, fn, em, _, _, st = ch[k]
                total += 1                     # Amendment 1: late = waited more than one chunk duration
                late += st - rt > self.chunk_ms / 1000 + 1e-9
                if now - em < 5.0:
                    recent.append((em - rt) * 1000)
        w = list(self.walls)[-50:]
        m = Table.grid(padding=(0, 3))
        for _ in range(4):
            m.add_column(justify="left")
        behind = Text(f"{lag:5.2f} s", style="bold red" if lag > self.chunk_ms / 1000 else "bold green")
        latepct = 100 * late / max(1, total)
        m.add_row(Text(f"clock {int(now // 60):02d}:{now % 60:05.2f}", style="bold"),
                  Text(f"streams live {len(srv.live):4d} / {self.n}"),
                  Text(f"chunk {self.chunk_ms} ms"),
                  Text(f"steps {len(srv.steps):6d}"))
        m.add_row(Text(f"step p50 {np.median(w) if w else 0:6.1f} ms"),
                  Text(f"batch {srv.steps[-1]['B'] if srv.steps else 0:4d}"),
                  Text.assemble("behind real time ", behind),
                  Text(f"late chunks {latepct:6.2f}%", style="bold red" if latepct > 0.1 else "bold green"))
        m.add_row(Text(f"chunk latency p50 {np.percentile(recent, 50) if recent else 0:6.0f} ms"),
                  Text(f"p95 {np.percentile(recent, 95) if recent else 0:6.0f} ms"),
                  Text(f"peak lag {self.peak_lag:5.2f} s"), Text(""))
        self._pick()
        panels = []
        width = max(40, (self.live.console.width if self.live is not None else 120) - 6)
        for s in self.shown:
            t = self._text(s)
            heard = s.idx * srv.hop / 16000
            arrived = max(0.0, now - s.ls.arrival_s)
            tail = t[-(2 * width):] if len(t) > 2 * width else t
            body = Text(tail)
            if len(tail) > 30:
                body.stylize("bold", len(tail) - 30)
            panels.append(Panel(body, height=4, title=f"stream {s.ls.sid:3d}  audio in {arrived:5.1f} s  "
                                f"transcribed to {heard:5.1f} s", title_align="left"))
        return Panel(Group(m, *panels), title=self.title, subtitle=self.caption, subtitle_align="left",
                     border_style="cyan")

    def __call__(self, srv):
        now = time.perf_counter()
        if now - self.last < self.dt:
            return
        self.last = now
        self.live.update(self.render())


ACT_TITLE = {0: "80 ms chunks: voice agents", 1: "160 ms chunks", 3: "320 ms chunks: live captions",
             6: "560 ms chunks", 13: "1.12 s chunks: transcription"}
ARM_TITLE = {"stock": "STOCK NeMo (gc.freeze)", "ours": "OUR ENGINE (FP8 + mel graph + fused decoder)"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default="stock:0:232,ours:0:232,stock:3:680,ours:3:680,stock:13:1240,ours:13:1240")
    ap.add_argument("--act-s", type=float, default=45.0)
    ap.add_argument("--stagger-s", type=float, default=10.0)
    ap.add_argument("--show", type=int, default=4)
    ap.add_argument("--fps", type=float, default=6.0)
    ap.add_argument("--pause-s", type=float, default=4.0, help="pause between acts (shows the final frame)")
    ap.add_argument("--log", default=os.path.join(os.environ.get("LIVE_OUT", "/out"), "demo_live.log"))
    ap.add_argument("--seed", type=int, default=20261005)
    # 8 Oct (H100 video): GPU label in the title, engines per chunk size, the label of our arm, and which chunk sizes
    # use the mel graph (defaults reproduce the Spark demo)
    ap.add_argument("--title-prefix", default="")
    ap.add_argument("--engines", default="", help="per chunk size, e.g. 0=/out/trt/ml_r0/steady_fp16_b2048.plan,13=...")
    ap.add_argument("--ours-label", default=ARM_TITLE["ours"])
    ap.add_argument("--mel-graph-rs", default="0,1,3,6,13")
    # 8 Oct (Spark retake): one-line caption on the panel's bottom border, e.g. the share of measured capacity shown
    ap.add_argument("--caption", default="")
    a = ap.parse_args()
    ARM_TITLE["ours"] = a.ours_label
    engines = {int(k): v for k, v in (x.split("=", 1) for x in a.engines.split(",") if x)}
    mg_rs = {int(x) for x in a.mel_graph_rs.split(",") if x}
    acts = []
    for spec in a.acts.split(","):
        arm, r, n = spec.split(":")
        if arm not in ARM_TITLE:
            raise ValueError(f"unknown arm {arm} (stock or ours)")
        acts.append((arm, int(r), int(n)))
    os.environ.setdefault("FUSED_JOINT", "triton")
    os.environ.setdefault("FUSED_LSTM", "triton")
    os.environ.setdefault("TRT_MAX_CALL_B", "512")
    # terminal = the dashboard only: keep a handle on the real stdout, send fds 1 and 2 (NeMo, TensorRT, Python
    # warnings and prints, C libraries) to the log file
    term = os.fdopen(os.dup(1), "w", buffering=1)
    logf = os.open(a.log, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(logf, 1)
    os.dup2(logf, 2)
    import sys
    sys.stdout = os.fdopen(1, "w", buffering=1)
    sys.stderr = os.fdopen(2, "w", buffering=1)
    con = Console(file=term, force_terminal=True)
    con.clear()                                    # wipe the container's start-up banner from the screen
    con.print("[bold]Loading Nemotron 3.5 ASR streaming (0.6B) on the GPU ...[/bold]")
    m = harness.load_model("nvidia/nemotron-3.5-asr-streaming-0.6b", acts[0][1], "en-US", a.seed)
    sources, _ = loadgen.make_load("earnings22_full", 1, a.seed, 0, 1)
    for arm, r, n in acts:
        harness.set_chunk(m, r)
        chunk_ms = harness.CHUNK_MS[r]
        _, streams = loadgen.make_load("earnings22_full", n, a.seed, a.stagger_s, a.act_s + 30)
        title = f"{ACT_TITLE.get(r, f'{chunk_ms} ms chunks')}  |  {ARM_TITLE[arm]}  |  {n} live streams"
        if a.title_prefix:
            title = f"{a.title_prefix}  |  {title}"
        print(f"=== act {arm} r={r} N={n}", flush=True)
        with RLive(console=con, refresh_per_second=a.fps, transient=False, redirect_stdout=False,
                   redirect_stderr=False) as lv:
            lv.update(Panel(Text("warming up (graphs captured, encoder warmed at every batch size) ..."),
                            title=title, subtitle=a.caption or None, subtitle_align="left", border_style="cyan"))
            if arm == "stock":
                enc = harness.make_encoder_step(m, "stock", None, r)
                kw = {}
                os.environ["HARNESS_GC"] = "freeze"
            else:
                enc = harness.make_encoder_step(m, "/w/trt/trt_step.py:build", engines.get(r, ENG.format(r=r)), r)
                kw = {"cache_dtype": "float16", "decoder": "fused", "mel_graph": r in mg_rs}
                os.environ["HARNESS_GC"] = "default"
            srv = harness.Server(m, sources, streams, ["en-US"] * n, "chunk", "real", 0.0, enc, **kw)
            srv.stop_at = a.act_s
            dash = Dash(title, srv, a.show, a.fps, chunk_ms, n, a.caption)
            dash.live = lv
            srv.on_step = dash
            srv.run()
            lv.update(dash.render())
            print(json.dumps({"act": [arm, r, n], **srv.summary(chunk_ms / 1000)}, default=str), flush=True)
        if arm == "stock":
            mode = harness.dec_graph_mode(m)
            if mode is not None and "full" not in mode.lower():
                con.print(f"[bold red]WARNING: NeMo decoder fell back to CUDA graph mode {mode}; this act is not "
                          f"representative. Re-run.[/bold red]")
        time.sleep(a.pause_s)
        del srv, dash, enc
        gc.collect()
    con.print("[bold]End of demo.[/bold]")


if __name__ == "__main__":
    main()
