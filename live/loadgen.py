# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Nigel Cannings. Part of fastconformer-trt (see LICENSE and NOTICE).
"""Seeded load generator for the live harness: N simulated real-time streams of long-form audio with staggered starts.

Long-form audio is built by concatenating, in order, the segments of one Earnings-22 call (the `id` column is
<call>/<segment>.wav) or the utterances of one LibriSpeech test-clean speaker (chapter then utterance order). Stream i
uses source (i mod number of sources); when N exceeds the number of sources, later rounds start at a seeded random
segment so that streams sharing a source are not identical. Each stream gets an arrival time drawn uniformly from
[0, stagger_s) and is cut to at most max_s seconds (at a segment boundary, never mid-segment, so its reference text
stays exact). Everything is a pure function of (source set, N, seed, stagger_s, max_s).
"""
from __future__ import annotations

import glob
import os
import random
from dataclasses import dataclass, field

import numpy as np
import pyarrow.parquet as pq

import finetune_stride as fs

DATA = os.environ.get("ASR_DATA", os.path.expanduser("~/asr_data"))
SR = 16000


@dataclass
class StreamSpec:
    sid: int
    source: str            # call id or speaker id
    arrival_s: float       # when its first sample arrives, relative to the start of the run
    audio: np.ndarray = field(repr=False)
    text: str = field(repr=False)
    segments: list = field(default_factory=list, repr=False)

    @property
    def duration_s(self) -> float:
        return len(self.audio) / SR


def _sources(source_set: str, n: int) -> dict[str, list[dict]]:
    """source key -> rows in order, each with audio bytes and text; only the first n sources (sorted by key) are read,
    to keep host memory small on the shared box."""
    if source_set == "earnings22_full":
        files = sorted(glob.glob(os.path.join(DATA, "earnings22_full", "*.parquet")))
        ids = []
        for f in files:
            ids += pq.read_table(f, columns=["id"]).column("id").to_pylist()
        keep = set(sorted({i.split("/")[0] for i in ids})[:n])
        rows = []
        for f in files:
            t = pq.read_table(f, columns=["id"])
            sel = [k for k, i in enumerate(t.column("id").to_pylist()) if i.split("/")[0] in keep]
            if sel:
                rows += pq.read_table(f, columns=["id", "audio", "text"]).take(sel).to_pylist()
        groups: dict[str, list[dict]] = {}
        for r in rows:
            call, seg = r["id"].split("/")
            r["_order"] = int(seg.split(".")[0])
            groups.setdefault(call, []).append(r)
    elif source_set == "test_clean":
        rows = pq.read_table(os.path.join(DATA, "test_clean.parquet"), columns=["id", "audio", "text", "speaker_id"]).to_pylist()
        groups = {}
        for r in rows:
            r["_order"] = tuple(int(x) for x in r["id"].split("-")[1:])
            groups.setdefault(str(r["speaker_id"]), []).append(r)
    else:
        raise ValueError(f"unknown source set {source_set}")
    for g in groups.values():
        g.sort(key=lambda r: r["_order"])
    return dict(sorted(groups.items()))


def make_streams(source_set: str, n: int, seed: int, stagger_s: float, max_s: float) -> list[StreamSpec]:
    rng = random.Random(seed)
    src = _sources(source_set, n)
    keys = list(src)
    out = []
    for i in range(n):
        key = keys[i % len(keys)]
        segs = src[key]
        start = 0 if i < len(keys) else rng.randrange(len(segs))
        auds, texts, used = [], [], []
        total = 0
        for r in segs[start:] + segs[:start]:
            w = fs.load_audio(r)
            if auds and (total + len(w)) / SR > max_s:
                break
            auds.append(w)
            texts.append(fs.ref_text(r))
            used.append(r["id"])
            total += len(w)
        out.append(StreamSpec(sid=i, source=key, arrival_s=rng.uniform(0, stagger_s),
                              audio=np.concatenate(auds), text=" ".join(texts), segments=used))
    return out


@dataclass
class LoadStream:
    """A stream defined by reference into a shared source: sample t is source[(start + t) mod len(source)]."""
    sid: int
    src: int               # index into the sources list
    start: int             # first sample in the source (a segment boundary)
    n_samples: int
    arrival_s: float
    text: str | None = None


def from_specs(specs: list[StreamSpec]) -> tuple[list[np.ndarray], list[LoadStream]]:
    """the explicit-audio streams of make_streams as (sources, streams): one source per stream, no wrap-around."""
    return ([s.audio for s in specs],
            [LoadStream(sid=s.sid, src=i, start=0, n_samples=len(s.audio), arrival_s=s.arrival_s, text=s.text)
             for i, s in enumerate(specs)])


_LOAD_CACHE: dict = {}


def make_load(source_set: str, n: int, seed: int, stagger_s: float, dur_s: float,
              max_sources: int = 1000, len_spread: float = 0.0) -> tuple[list[np.ndarray], list[LoadStream]]:
    """N streams of exactly dur_s seconds for a concurrency sweep, sharing source audio (no per-stream copies, so N
    can be in the thousands). Source k is one whole Earnings-22 call (or test-clean speaker), its segments
    concatenated in order. Stream i reads source (i mod number of sources) from a seeded random segment boundary and
    wraps round to the start of the call if it runs past the end. Arrival times are uniform in [0, stagger_s).
    len_spread f > 0 draws each stream's length from dur_s * [1 - f, 1 + f] (seeded).
    Reference text is not kept (sweeps measure time; correctness is equiv.py's job)."""
    rng = random.Random(seed)
    key = (source_set, max_sources)
    if key not in _LOAD_CACHE:                     # decode the source audio once per process
        disk = os.path.join(os.environ.get("LIVE_OUT", "/out"), f"loadgen_cache_{source_set}_{max_sources}.npz")
        if os.path.exists(disk):                   # decoded pool saved by an earlier process (identical audio)
            z = np.load(disk, allow_pickle=False)
            pool, lens, nseg = z["pool"], z["lens"], z["nseg"]
            seglens = z["seglens"]
            sources, bounds, o, q = [], [], 0, 0
            for L, k in zip(lens, nseg):
                sources.append(pool[o:o + L])
                sl = seglens[q:q + k]
                bounds.append(np.cumsum(np.concatenate([[0], sl[:-1]])))
                o += L
                q += k
        else:
            src = _sources(source_set, max_sources)
            sources, bounds, seglens = [], [], []
            for segs in src.values():
                auds = [fs.load_audio(r) for r in segs]
                sources.append(np.concatenate(auds))
                bounds.append(np.cumsum([0] + [len(a) for a in auds[:-1]]))
                seglens += [len(a) for a in auds]
            try:
                tmp = disk + ".tmp.npz"
                np.savez(tmp, pool=np.concatenate(sources), lens=np.array([len(x) for x in sources]),
                         nseg=np.array([len(b) for b in bounds]), seglens=np.array(seglens))
                os.replace(tmp, disk)
            except OSError:
                pass
        _LOAD_CACHE[key] = (sources, bounds)
    sources, bounds = _LOAD_CACHE[key]
    out = []
    for i in range(n):
        k = i % len(sources)
        start = int(bounds[k][rng.randrange(len(bounds[k]))])
        arrival = rng.uniform(0, stagger_s)
        # len_spread > 0 (load fix of 6 Oct): seeded per-stream length uniform in dur_s * [1 - f, 1 + f], so final
        # partial chunks vary; with equal lengths every stream ends on the same short remainder (late-rule artefact)
        d = dur_s if len_spread <= 0 else rng.uniform(dur_s * (1 - len_spread), dur_s * (1 + len_spread))
        out.append(LoadStream(sid=i, src=k, start=start, n_samples=int(round(d * SR)), arrival_s=arrival))
    return sources, out
