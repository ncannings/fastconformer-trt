# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Nigel Cannings. Part of fastconformer-trt (see LICENSE and NOTICE).
# Follows NVIDIA NeMo (Apache-2.0) semantics: conformer_stream_step split at its seams and the chunk geometry of
# CacheAwareStreamingAudioBuffer; NeMo's own modules do the work in the stock arm.
"""P1 live harness: N simulated real-time streams served by one loop of batched cache-aware steps, with a pluggable
encoder step (stock NeMo, or a custom engine such as live/trt/).

What it does. A seeded load generator (loadgen.py) gives N long-form streams with staggered arrival times; their
audio "arrives" in real time. The server loop repeatedly takes every stream whose next chunk is complete (its last mel
frame computable from the audio received so far) and runs one cache-aware step on them as a batch: per-stream encoder
caches live in a slot-indexed slab on the GPU and are gathered for the batch and scattered back after it; RNN-T
hypotheses (decoder state) are carried per stream. Streams join and leave at any step. A stream's first chunk has a
different size and no pre-encode drop (NeMo's streaming config), so first chunks and steady chunks run as separate
calls within one step; streams with different target languages also run as separate calls (stock NeMo applies one
language prompt per call).

The step is stock NeMo's conformer_stream_step split at its seams, nothing changed: encoder step
(encoder.cache_aware_stream_step, keep_all_outputs=False), language prompt (model._apply_prompt_to_encoded), RNN-T
greedy decoding with partial hypotheses (model.decoding.rnnt_decoder_predictions_tensor). Only the encoder step is
pluggable.

ENCODER INTERFACE (for custom engines). `--encoder stock` (default) or `--encoder FILE.py:build --engine PATH`.
build(model, engine_path, r) is called once and returns
    enc_step(sig, lens, cache_last_channel, cache_last_time, cache_last_channel_len, drop_extra_pre_encoded)
        -> (encoded, encoded_len, cache_last_channel_next, cache_last_time_next, cache_last_channel_len_next)
with the semantics of model.encoder.cache_aware_stream_step(..., keep_all_outputs=False). All CUDA, contiguous:
    sig [B, 128, W] float32; steady chunk W = 9 + 8(r+1) and drop = 2; first chunk W = 8(r+1) - 7 and drop = 0
    lens [B] int64 (less than W on a stream's last chunk, which is zero-padded to full width)
    cache_last_channel [24, B, 56, 1024] float32, cache_last_time [24, B, 1024, 8] float32, cache_last_channel_len [B]
    int64; encoded [B, 1024, T'] float32 (T' = r+1 on steady chunks), encoded_len [B] int64, caches as the inputs.
    --cache-dtype float16 keeps the cache slab (and the caches passed to enc_step) in float16; enc_step must then
    return float16 caches. The stock step casts to float32 and back.

Chunking follows NeMo's CacheAwareStreamingAudioBuffer exactly (chunk and shift sizes, pre-encode cache frames,
zero padding, length clamping), every chunk zero-padded to the full chunk width as the buffer does for all but the
longest stream in a batch. Settings follow the stock streaming script: float32, TF32 matmuls ("high"), default
greedy-batch RNN-T decoding config with fused_batch_size=-1, strip_lang_tags=true.

Audio lives once on the GPU (a pool of sources; a stream is a (source, start, length) reference that wraps round), so
thousands of streams cost no per-stream audio copies. Mel features: --mel chunk (default) computes each step's frames
from the audio received so far (a window of raw samples around the chunk, one batched preprocessor call), which is
what a live server must do; --mel full computes each stream's features over its whole audio when it joins, exactly as
the stock buffer does (not causal, cost outside the timed step; used in the equivalence test).

Scheduling: by default steps run on a grid of one chunk duration (--tick-ms -1): at each grid point every stream with
a complete chunk joins one batched step; a step that overruns the grid starts the next one at once. --tick-ms 0 runs
a step as soon as any chunk is complete (eager).

Clock: --clock real sleeps until audio arrives (a true real-time run); --clock virtual skips idle waits but advances
by the measured compute time of each step, so queueing and misses are the same as in a real-time run with the same
step times.

Measured: per-step wall time, batch composition and a CUDA-event split of mel / encoder / prompt+decoder time;
per-chunk latency (emit time minus the time the chunk's last needed sample arrived, "excluding algorithmic delay")
and the same from the time the chunk's first new sample arrived ("including"); final-token latency per stream (end of
its audio to its final transcript, excluding and including); deadline misses (a chunk whose step started after the
stream's next chunk was already complete, i.e. the stream is behind real time; also counted: chunks emitted after the
next chunk was complete); GPU memory, cache slab size, GPU utilisation samples. On a shared GPU none of the timing
numbers is reportable.

usage: python harness.py OUT_DIR --r 3 --n 8 [--source earnings22_full] [--max-s 120] [--stagger-s 30] \
       [--mel chunk|full] [--clock real|virtual] [--tick-ms -1] [--encoder stock|FILE.py:build --engine PATH]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import threading
import time

import numpy as np
import torch

import loadgen
from torch.profiler import record_function as RF

SR = 16000
CHUNK_MS = {0: 80, 1: 160, 3: 320, 6: 560, 13: 1120}
DEC_BUCKETS = os.environ.get("HARNESS_DEC_BUCKETS", "")   # "" none, "pow2" next power of two (min 8), "max" = N


MEL_CHECK = os.environ.get("HARNESS_MEL_CHECK", "0") == "1"
DEC_PAD_COPY = os.environ.get("HARNESS_DEC_PAD_COPY", "0") == "1"
SYNC_DEBUG = os.environ.get("HARNESS_SYNC_DEBUG", "0") == "1"


def dec_bucket(b: int, n: int) -> int:
    if DEC_BUCKETS == "pow2":
        return max(8, 1 << (b - 1).bit_length())
    if DEC_BUCKETS == "max":
        return n
    return b


CLONE_DEC_STATE = os.environ.get("HARNESS_CLONE_DEC_STATE", "0") == "1"   # debug option; not needed
# 8 Oct: HARNESS_SLAB_STEP=1 lets an encoder step with slab_step() (trt_step) gather, run and scatter piece by piece
# straight on the cache slab (no full-batch gather buffer or cache output: about 1 cache copy per stream instead of 3)
SLAB_STEP = os.environ.get("HARNESS_SLAB_STEP", "0") == "1"
# 8 Oct: HARNESS_STOCK_SLAB_STEP=1 does the same for NeMo's stock encoder step (StockStep.slab_step: NeMo's own call
# on pieces of at most HARNESS_STOCK_PIECE rows, float32 caches as shipped); separate from HARNESS_SLAB_STEP so a
# stock arm changes only when asked
STOCK_SLAB_STEP = os.environ.get("HARNESS_STOCK_SLAB_STEP", "0") == "1"
UTIL_S = float(os.environ.get("HARNESS_UTIL_S", "0.2"))     # GPU utilisation sampling period (NVML); 0 = off
PINNED_H2D = os.environ.get("HARNESS_PINNED_H2D", "0") == "1"


class _H2D:
    """small host-to-device uploads (per-step indices) through a ring of pinned buffers, asynchronous; a buffer is
    reused only after its previous copy has completed (event). HARNESS_PINNED_H2D=1; otherwise synchronous pageable
    copies as before. Values are identical either way."""

    def __init__(self, k: int = 64):
        self.ring = [None] * k
        self.i = 0

    def __call__(self, a) -> torch.Tensor:
        a = np.ascontiguousarray(np.asarray(a, dtype=np.int64))
        if not PINNED_H2D:
            return torch.from_numpy(a).cuda()
        n = a.size
        slot = self.ring[self.i]
        if slot is None or slot[0].numel() < n:
            slot = [torch.empty(max(n, 4096), dtype=torch.int64).pin_memory(), None]
            self.ring[self.i] = slot
        elif slot[1] is not None:
            slot[1].synchronize()
        self.i = (self.i + 1) % len(self.ring)
        slot[0][:n].numpy()[:] = a.reshape(-1)
        out = slot[0][:n].to("cuda", non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        slot[1] = ev
        return out.view(a.shape)


H2D = _H2D()


def load_model(name: str, r: int, lang: str | None, seed: int):
    """The model exactly as the stock streaming script sets it up (default config values)."""
    import gpu_reserve
    gpu_reserve.reserve(float(os.environ.get("RESERVE_HOST_GB", "10")), float(os.environ.get("RESERVE_CUDA_GB", "8")))
    import lightning.pytorch as pl
    import nemo.collections.asr as nemo_asr
    from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
    pl.seed_everything(seed)
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("high")
    ext = os.path.join(os.environ.get("LIVE_OUT", "/out"), "nemotron35_extracted")
    if name == "nvidia/nemotron-3.5-asr-streaming-0.6b" and os.path.exists(os.path.join(ext, "model_weights.ckpt")):
        # the same .nemo, extracted once (tar xf of the HF snapshot file): saves the per-process extraction (~20 s),
        # which matters with a fresh process per trial
        from nemo.core.connectors.save_restore_connector import SaveRestoreConnector
        c = SaveRestoreConnector()
        c.model_extracted_dir = ext
        m = nemo_asr.models.ASRModel.restore_from(ext, map_location=torch.device("cuda"), save_restore_connector=c)
    else:
        m = nemo_asr.models.ASRModel.from_pretrained(name, map_location=torch.device("cuda"))
    m.encoder.set_default_att_context_size(att_context_size=[56, r])
    dc = RNNTDecodingConfig(fused_batch_size=-1)
    # stock default: NeMo's CUDA-graph label-looping decoder. HARNESS_DEC_CUDA_GRAPHS=0 turns it off (for comparison).
    dc.greedy.use_cuda_graph_decoder = os.environ.get("HARNESS_DEC_CUDA_GRAPHS", "1") == "1"
    m.change_decoding_strategy(dc)
    if hasattr(m, "set_inference_prompt"):
        m.set_inference_prompt(lang if lang is not None else "auto")
        m.decoding.set_strip_lang_tags(True, lang_tag_pattern=None)
    m = m.to(device=torch.device("cuda"), dtype=torch.float32)
    m.eval()
    return m


def dec_graph_mode(model) -> str | None:
    """NeMo's label-looping decoder graph mode: 'full_graph' normally. NeMo silently falls back to 'no_while_loops'
    (slower) if full-graph capture fails ("Full CUDA graph compilation failed", seen twice on the shared GPU on
    6 Oct); recorded in every summary so a degraded decoder is visible."""
    comp = getattr(model.decoding.decoding, "decoding_computer", None)
    mode = getattr(comp, "cuda_graphs_mode", None)
    return str(mode) if mode is not None else None


def set_chunk(model, r: int) -> None:
    model.encoder.set_default_att_context_size(att_context_size=[56, r])


def make_encoder_step(model, encoder: str, engine: str | None, r: int):
    """'stock' -> NeMo's own encoder step; 'FILE.py:build' -> build(model, engine, r) from that file."""
    if encoder == "stock":
        return StockStep(model)
    path, _, fn = encoder.partition(":")
    spec = importlib.util.spec_from_file_location("custom_encoder", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, fn or "build")(model, engine, r)


class StockStep:
    """NeMo's own encoder step (model.encoder.cache_aware_stream_step, keep_all_outputs=False), unchanged numerics.
    __call__ is the stock path on gathered caches. slab_step (used when HARNESS_STOCK_SLAB_STEP=1, 8 Oct) runs the same
    NeMo call piece by piece (HARNESS_STOCK_PIECE rows, default 512) straight on the cache slab: gather a piece into
    persistent piece buffers, call NeMo, scatter its new caches back. No full-batch gather or full-batch cache output,
    so stock's per-stream memory is its float32 slab plus one piece. Bit-identity with __call__ is checked by
    check_stock_slab.py (a piece is a smaller batch for NeMo's GEMMs)."""

    def __init__(self, model):
        self.m = model
        self.piece = int(os.environ.get("HARNESS_STOCK_PIECE", "512"))
        self.bufs: dict = {}

    def __call__(self, sig, lens, cc, ct, cl, drop):
        dt = cc.dtype
        e, el, cc2, ct2, cl2 = self.m.encoder.cache_aware_stream_step(
            processed_signal=sig, processed_signal_length=lens, cache_last_channel=cc.float(),
            cache_last_time=ct.float(), cache_last_channel_len=cl, keep_all_outputs=False,
            drop_extra_pre_encoded=drop)
        return e, el, cc2.to(dt), ct2.to(dt), cl2

    def _buf(self, key, shape, dtype):
        n = int(np.prod(shape))
        b = self.bufs.get((key, dtype))
        if b is None or b.numel() < n:
            self.bufs.pop((key, dtype), None)
            with torch.inference_mode(False):
                b = self.bufs[(key, dtype)] = torch.empty(n, dtype=dtype, device="cuda")
        return b[:n].view(shape)

    def slab_step(self, sig, lens, slab_cc, slab_ct, slab_cl, slots, drop):
        B = sig.shape[0]
        es, els = [], []
        for i in range(0, B, self.piece):
            j = min(B, i + self.piece)
            sl = slots[i:j]
            cc = self._buf("cc", (slab_cc.shape[0], j - i) + tuple(slab_cc.shape[2:]), slab_cc.dtype)
            ct = self._buf("ct", (slab_ct.shape[0], j - i) + tuple(slab_ct.shape[2:]), slab_ct.dtype)
            torch.index_select(slab_cc, 1, sl, out=cc)
            torch.index_select(slab_ct, 1, sl, out=ct)
            e, el, cc2, ct2, cl2 = self(sig[i:j], lens[i:j], cc, ct, slab_cl.index_select(0, sl), drop)
            slab_cc.index_copy_(1, sl, cc2)
            slab_ct.index_copy_(1, sl, ct2)
            slab_cl.index_copy_(0, sl, cl2)
            del cc2, ct2
            es.append(e)
            els.append(el)
        if len(es) == 1:
            return es[0], els[0]
        T = max(x.shape[2] for x in es)
        if any(x.shape[2] != T for x in es):
            raise RuntimeError(f"encoder output widths differ between pieces: {[x.shape[2] for x in es]}")
        return torch.cat(es), torch.cat(els)


def _clone_state(x):
    """deep copy of a decoder state item (tensors cloned). NeMo's label-looping decoder with CUDA graphs returns
    per-hypothesis states that are views into its batched (graph-owned) buffers; those buffers are reused by the next
    call, which in a live server has a different batch composition. Stock streaming never changes the batch, so it never
    sees this; we clone so each stream owns its state."""
    import dataclasses
    if isinstance(x, torch.Tensor):
        return x.clone()
    if isinstance(x, (list, tuple)):
        return type(x)(_clone_state(v) for v in x)
    if isinstance(x, dict):
        return {k: _clone_state(v) for k, v in x.items()}
    if dataclasses.is_dataclass(x):
        return dataclasses.replace(x, **{f.name: _clone_state(getattr(x, f.name)) for f in dataclasses.fields(x)
                                         if f.init})
    return x


class MelGraph:
    """--mel-graph (custom arm): the per-chunk mel path (audio gather from the GPU pool, NeMo's preprocessor, frame
    gather into the encoder input) captured as one CUDA graph per (first-chunk or steady, batch bucket). The batch is
    padded to a power-of-two bucket (minimum 8) and every window to the fixed full-window length; padded rows and
    samples are zeros and are masked or dropped, so the result equals the eager path (checked with
    HARNESS_MEL_CHECK=1). One small host-to-device copy of per-row integers replaces about 60 kernel launches."""

    def __init__(self, srv: "Server"):
        self.srv = srv
        self.graphs: dict = {}

    def _geom(self, first: bool):
        srv = self.srv
        chunk, _, pre = srv._sizes(first)
        W = pre + chunk
        if first and isinstance(srv.cfg.pre_encode_cache_size, list):
            L = (chunk - 1) * srv.hop + srv.half
        else:
            L = (chunk + pre + 1) * srv.hop + srv.half
        return W, L

    def _build(self, first: bool, Bb: int):
        srv = self.srv
        W, L = self._geom(first)
        st = {"meta": torch.zeros(Bb, 7, dtype=torch.int64, device="cuda")}

        def body():
            m = st["meta"]
            start, slen, base, n, a, zp, nv = (m[:, k] for k in range(7))
            pos = torch.arange(L, device="cuda")
            idx = base[:, None] + (start[:, None] + pos[None, :]) % slen[:, None]
            x = torch.where(pos[None, :] < n[:, None], srv.pool[idx], 0.0)
            f, _ = srv.pre(input_signal=x, length=n.float())
            p = torch.arange(W, device="cuda")[None, :]
            src = (a[:, None] + p).clamp(0, f.size(-1) - 1)
            ok = (p >= zp[:, None]) & (p < (zp + nv)[:, None])
            return torch.gather(f, 2, src[:, None, :].expand(Bb, srv.F, W)) * ok[:, None, :]
        st["meta"][:, 1] = 1                                   # valid dummy rows for warm-up and capture
        st["meta"][:, 3] = L
        s_ = torch.cuda.Stream()
        s_.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s_), torch.inference_mode():
            for _ in range(2):
                body()
        torch.cuda.current_stream().wait_stream(s_)
        # capture with capture_begin/end, not the torch.cuda.graph context: that context calls empty_cache(), which
        # frees memory NeMo's captured decoder graphs still use (illegal memory access on their next replay; the
        # same NeMo behaviour as the empty_cache bug of 5 Oct)
        g = torch.cuda.CUDAGraph()
        s_.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s_), torch.inference_mode():
            g.capture_begin()
            st["out"] = body()
            g.capture_end()
        torch.cuda.current_stream().wait_stream(s_)
        st["graph"] = g
        return st

    def __call__(self, group, first, idx, lo, hi, zpad, sa, sb) -> torch.Tensor:
        srv = self.srv
        B = len(group)
        Bb = max(8, 1 << (B - 1).bit_length())
        key = (first, Bb)
        if key not in self.graphs:
            self.graphs[key] = self._build(first, Bb)
        st = self.graphs[key]
        meta = np.zeros((Bb, 7), dtype=np.int64)
        meta[:, 1] = 1
        meta[:, 3] = self._geom(first)[1]
        meta[:B, 0] = [s.ls.start for s in group] + sa
        meta[:B, 1] = [s.srclen for s in group]
        meta[:B, 2] = [s.base for s in group]
        meta[:B, 3] = sb - sa
        meta[:B, 4] = lo - sa // srv.hop - zpad
        meta[:B, 5] = zpad
        meta[:B, 6] = hi - lo
        st["meta"].copy_(H2D(meta))
        st["graph"].replay()
        return st["out"][:B]


class Live:
    __slots__ = ("ls", "lang", "slot", "n_samples", "n_frames", "feats", "idx", "step", "hyp", "done", "chunks",
                 "final_emit", "base", "srclen")

    def __init__(self, ls: loadgen.LoadStream, lang: str, slot: int, hop: int, base: int, srclen: int):
        self.ls, self.lang, self.slot = ls, lang, slot
        self.n_samples = ls.n_samples
        self.n_frames = ls.n_samples // hop + 1       # stock: width of the stream's feature tensor
        self.feats = None
        self.idx = 0                                  # next mel frame (stock buffer_idx)
        self.step = 0
        self.hyp = None
        self.done = False
        self.chunks = []                              # (ready, first_new, emit, n_tokens, first, step_start)
        self.final_emit = None
        self.base, self.srclen = base, srclen


class Server:
    def __init__(self, model, sources: list[np.ndarray], streams: list[loadgen.LoadStream], langs: list[str],
                 mel: str = "chunk", clock: str = "virtual", tick_s: float = 0.0, enc_step=None,
                 abort_backlog_s: float = 0.0, cache_dtype: str = "float32", decoder: str = "stock",
                 max_slots: int | None = None, admit_limit: int = 0, mel_graph: bool = False):
        from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
        self.m = model
        buf = CacheAwareStreamingAudioBuffer(model=model, online_normalization=False, pad_and_drop_preencoded=False)
        self.pre = buf.preprocessor           # the stock buffer's own preprocessor copy (dither 0, pad_to 0)
        self.cfg = model.encoder.streaming_cfg
        fcfg = model.cfg.preprocessor
        self.hop = int(round(fcfg.window_stride * SR))
        self.half = int(fcfg.n_fft) // 2
        self.F = model.encoder._feat_in
        self.mel, self.clock_mode, self.tick_s = mel, clock, tick_s
        self.enc_step = enc_step or make_encoder_step(model, "stock", None, 0)
        self.abort_backlog_s = abort_backlog_s
        self.aborted = None
        self.vnow, self.t0 = 0.0, None
        lens = [len(a) for a in sources]
        self.pool = torch.from_numpy(np.concatenate(sources).astype(np.float32)).cuda()
        self.src_base = np.cumsum([0] + lens[:-1]).tolist()
        self.src_len = lens
        self.n_slots = max_slots or len(streams)      # slots are recycled when streams leave
        self.admit_limit = admit_limit                # >0: closed loop (accuracy runs), at most this many live streams
        dt = getattr(torch, cache_dtype)              # float32 (stock) or float16 (a custom engine's boundary dtype)
        # 8 Oct: built from the one-stream initial state, broadcast in the slab dtype, instead of the initial state
        # for all slots in float32 then cast (a transient of twice a float16 slab: 26 GB at 4,096 slots, left as
        # reserved-but-unallocated memory); the values are the same for every slot
        c1, t1, l1 = model.encoder.get_initial_cache_state(batch_size=1)
        n = self.n_slots
        self.slab_cc = c1.to(dt).expand(c1.shape[0], n, *c1.shape[2:]).contiguous()
        self.slab_ct = t1.to(dt).expand(t1.shape[0], n, *t1.shape[2:]).contiguous()
        self.slab_cl = l1.expand(n, *l1.shape[1:]).contiguous()
        cc, ct = self.slab_cc, self.slab_ct
        self.slab_bytes = sum(t.numel() * t.element_size() for t in (cc, ct, self.slab_cl))
        self.pending = sorted(range(len(streams)), key=lambda i: streams[i].arrival_s)
        self.streams, self.langs = streams, langs
        self.live: list[Live] = []
        self.finished: list[Live] = []
        self.steps: list[dict] = []
        self.util: list[tuple[float, int]] = []
        self.cur_lang = None
        self.on_step = None          # optional callback(server) after every step (demo display); outside the timed step
        self.stop_at = 0.0           # >0: end the run at this time (demo)
        self.decoder = decoder
        self.mel_graph = MelGraph(self) if (mel_graph and mel == "chunk") else None
        self.lean = None
        if decoder == "lean":
            from lean_decoder import LeanDecoder
            max_s = max(ls.n_samples for ls in streams) / SR
            self.lean = LeanDecoder(model, self.n_slots, int(max_s * 12.5 * 2) + 64)   # 2 tokens per 80 ms frame
        elif decoder == "fused":
            # our fixed-shape CUDA-graph RNN-T decoder (fused_decoder.py): same slab interface as lean (self.lean),
            # every graph captured here, at start-up, at the full slot count, never re-captured
            from fused_decoder import FusedDecoder
            max_s = max(ls.n_samples for ls in streams) / SR
            self.lean = FusedDecoder(model, self.n_slots, int(max_s * 12.5 * 2) + 64, int(self.cfg.valid_out_len))
        elif decoder != "stock":
            raise ValueError(f"unknown decoder {decoder}")

    # ---- geometry, exactly as CacheAwareStreamingAudioBuffer.__iter__ ----
    def _sizes(self, first: bool):
        c = self.cfg
        pick = lambda v: (v[0] if first else v[1]) if isinstance(v, list) else v
        return pick(c.chunk_size), pick(c.shift_size), pick(c.pre_encode_cache_size)

    def _ready_time(self, s: Live) -> float:
        chunk = self._sizes(s.step == 0)[0]
        last = min(s.idx + chunk, s.n_frames) - 1
        need = last * self.hop + self.half
        smp = s.n_samples if (need >= s.n_samples or last == s.n_frames - 1) else need
        return s.ls.arrival_s + smp / SR

    # ---- clock ----
    def now(self) -> float:
        if self.clock_mode == "real":
            return time.perf_counter() - self.t0 if self.t0 is not None else 0.0
        return self.vnow

    def wait_until(self, t: float) -> None:
        if self.clock_mode == "real":
            d = t - self.now()
            if d > 0:
                time.sleep(d)
        else:
            self.vnow = max(self.vnow, t)

    # ---- audio and features ----
    def _gather_audio(self, group: list[Live], sa: list[int], sb: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """[B, L] zero-padded windows of stream samples [sa, sb) from the GPU pool (wrapping round each source)."""
        n = H2D([b - a for a, b in zip(sa, sb)])
        L = int(max(b - a for a, b in zip(sa, sb)))
        st = H2D([s.ls.start + a for s, a in zip(group, sa)])
        ln = H2D([s.srclen for s in group])
        bs = H2D([s.base for s in group])
        pos = torch.arange(L, device="cuda")
        idx = bs[:, None] + (st[:, None] + pos[None, :]) % ln[:, None]
        valid = pos[None, :] < n[:, None]
        return torch.where(valid, self.pool[idx], 0.0), n.float()

    def _feats_full(self, s: Live) -> torch.Tensor:
        x, n = self._gather_audio([s], [0], [s.n_samples])
        f, _ = self.pre(input_signal=x, length=n)
        return f[0]

    def _chunk_input(self, group: list[Live], first: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """the stock buffer's chunk for each stream: [pre-encode cache frames | chunk frames], zero-padded on the left
        when the cache is short and on the right past the stream's end; lengths clamped as the buffer does."""
        chunk, _, pre = self._sizes(first)
        W = pre + chunk
        B = len(group)
        idx = np.array([s.idx for s in group])
        nfr = np.array([s.n_frames for s in group])
        nsm = np.array([s.n_samples for s in group])
        hi = np.minimum(idx + chunk, nfr)
        lo = idx if (first and isinstance(self.cfg.pre_encode_cache_size, list)) else np.maximum(0, idx - pre)
        zpad = pre - (idx - lo)
        if self.mel == "full":
            sig = torch.zeros(B, self.F, W, device="cuda")
            for j, s in enumerate(group):
                f = s.feats[:, lo[j]:hi[j]]
                sig[j, :, zpad[j]:zpad[j] + f.size(-1)] = f
        elif self.mel_graph is not None:
            sa = np.maximum(0, (lo - 2) * self.hop)
            sb = np.minimum(nsm, (hi - 1) * self.hop + self.half)
            sig = self.mel_graph(group, first, idx, lo, hi, zpad, sa, sb)
            if MEL_CHECK:
                x, n = self._gather_audio(group, sa.tolist(), sb.tolist())
                f, _ = self.pre(input_signal=x, length=n)
                off = sa // self.hop
                p = torch.arange(W, device="cuda")[None, :]
                src = torch.from_numpy(lo - off - zpad).cuda()[:, None] + p
                ok = (p >= torch.from_numpy(zpad).cuda()[:, None]) & (p < torch.from_numpy(zpad + hi - lo).cuda()[:, None])
                ref = torch.gather(f, 2, src.clamp(0, f.size(-1) - 1)[:, None, :].expand(B, self.F, W)) * ok[:, None, :]
                d = (ref - sig).abs().max().item()
                self.mel_check_max = max(getattr(self, "mel_check_max", 0.0), d)
        else:
            sa = np.maximum(0, (lo - 2) * self.hop)
            sb = np.minimum(nsm, (hi - 1) * self.hop + self.half)
            x, n = self._gather_audio(group, sa.tolist(), sb.tolist())
            f, _ = self.pre(input_signal=x, length=n)                     # [B, F, Tf]
            off = sa // self.hop
            p = torch.arange(W, device="cuda")[None, :]
            src = H2D(lo - off - zpad)[:, None] + p                        # frame index in f for position p
            ok = (p >= H2D(zpad)[:, None]) & (p < H2D(zpad + hi - lo)[:, None])
            src = src.clamp(0, f.size(-1) - 1)
            sig = torch.gather(f, 2, src[:, None, :].expand(B, self.F, W)) * ok[:, None, :]
        lens = H2D(np.clip(nfr - idx + pre, 0, W))
        return sig.contiguous(), lens

    def _slab_mode(self) -> bool:
        if isinstance(self.enc_step, StockStep):
            return STOCK_SLAB_STEP
        return SLAB_STEP and hasattr(self.enc_step, "slab_step")

    def _alloc_gbuf(self):
        return (torch.empty(self.slab_cc[:, :self.n_slots].numel(), dtype=self.slab_cc.dtype, device="cuda"),
                torch.empty(self.slab_ct[:, :self.n_slots].numel(), dtype=self.slab_ct.dtype, device="cuda"))

    def _gather_bufs(self, B: int):
        """dense [L, B, ...] views of two persistent flat buffers sized for all slots (the cache gather targets)"""
        if getattr(self, "_gbuf", None) is None:
            with torch.inference_mode(False):        # normal tensors: updated in place inside and outside inference mode
                self._gbuf = self._alloc_gbuf()
        L = self.slab_cc.shape[0]
        sc = (L, B) + tuple(self.slab_cc.shape[2:])
        st = (L, B) + tuple(self.slab_ct.shape[2:])
        return (self._gbuf[0][:int(np.prod(sc))].view(sc), self._gbuf[1][:int(np.prod(st))].view(st))

    # ---- one call on a group of streams in the same phase and language ----
    def _run_group(self, group: list[Live], first: bool, ev: dict) -> None:
        _, shift, _ = self._sizes(first)
        ev["mel0"].append(torch.cuda.Event(enable_timing=True)); ev["mel0"][-1].record()
        with RF("live.mel"):
            sig, lens = self._chunk_input(group, first)
        slots = H2D([s.slot for s in group])
        drop = 0 if first else self.cfg.drop_extra_pre_encoded
        if self._slab_mode():
            ev["enc0"].append(torch.cuda.Event(enable_timing=True)); ev["enc0"][-1].record()
            with RF("live.encoder"):                      # gather, run and scatter piece by piece on the slab
                enc, enc_len = self.enc_step.slab_step(sig, lens, self.slab_cc, self.slab_ct, self.slab_cl, slots, drop)
            ev["dec0"].append(torch.cuda.Event(enable_timing=True)); ev["dec0"][-1].record()
        else:
            with RF("live.slab_gather"):
                cc, ct = self._gather_bufs(len(group))         # persistent: no full-size allocation per call
                torch.index_select(self.slab_cc, 1, slots, out=cc)
                torch.index_select(self.slab_ct, 1, slots, out=ct)
                cl = self.slab_cl.index_select(0, slots)
            ev["enc0"].append(torch.cuda.Event(enable_timing=True)); ev["enc0"][-1].record()
            with RF("live.encoder"):
                enc, enc_len, cc, ct, cl = self.enc_step(sig, lens, cc, ct, cl, drop)
            ev["dec0"].append(torch.cuda.Event(enable_timing=True)); ev["dec0"][-1].record()
            with RF("live.slab_scatter"):
                self.slab_cc.index_copy_(1, slots, cc)
                self.slab_ct.index_copy_(1, slots, ct)
                self.slab_cl.index_copy_(0, slots, cl)
        with RF("live.prompt"):
            enc = self.m._apply_prompt_to_encoded(enc)
        if self.lean is not None:
            if not (not first and all(s.idx + shift >= s.n_frames for s in group) and int(enc_len.max()) == 0):
                with RF("live.decoder"):
                    self.lean.step(enc, enc_len, slots, first)
            ev["end"].append(torch.cuda.Event(enable_timing=True)); ev["end"][-1].record()
            for s in group:
                s.idx += shift
                s.step += 1
            return
        partial = None if first else [s.hyp for s in group]
        B = len(group)
        if not first and all(s.idx + shift >= s.n_frames for s in group) and int(enc_len.max()) == 0:
            # nothing to decode (every stream in the call is on a final chunk that yields no encoder frame): the
            # hypotheses are unchanged, so skipping the decoder is exact.
            for s in group:
                s.idx += shift
                s.step += 1
            ev["end"].append(torch.cuda.Event(enable_timing=True)); ev["end"][-1].record()
            return
        Bp = dec_bucket(B, len(self.streams))
        if Bp > B:     # pad the decoder batch to a bucket: padded rows have length 0 and a fresh state, results dropped
            enc = torch.cat([enc, enc.new_zeros(Bp - B, *enc.shape[1:])])
            enc_len = torch.cat([enc_len, enc_len.new_zeros(Bp - B)])
            if partial is not None:
                import copy
                pad = copy.deepcopy(group[0].hyp) if DEC_PAD_COPY else None
                partial = partial + [copy.deepcopy(pad) if pad is not None else None for _ in range(Bp - B)]
        if SYNC_DEBUG:
            torch.cuda.synchronize()
            print(f"dec call first={first} B={B} enc={tuple(enc.shape)} len={enc_len.tolist()} "
                  f"ylen={[len(h.y_sequence) if h is not None else None for h in (partial or [])]} "
                  f"sids={[s.ls.sid for s in group]} idx={[s.idx for s in group]} nfr={[s.n_frames for s in group]}",
                  flush=True)
        with RF("live.decoder"):
            best = self.m.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=enc, encoded_lengths=enc_len, return_hypotheses=True, partial_hypotheses=partial)[:B]
        if SYNC_DEBUG:
            print(f"dec ok first={first} B={B} Bp={Bp} t={self.now():.2f}", flush=True)
            torch.cuda.synchronize()
        ev["end"].append(torch.cuda.Event(enable_timing=True)); ev["end"][-1].record()
        for j, s in enumerate(group):
            s.hyp = best[j]
            if CLONE_DEC_STATE and s.hyp.dec_state is not None:
                s.hyp.dec_state = _clone_state(s.hyp.dec_state)
            s.idx += shift
            s.step += 1

    def _sample_util(self, stop: threading.Event) -> None:
        if UTIL_S <= 0:
            return
        try:
            import pynvml
            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            while not stop.is_set():
                self.util.append((self.now(), pynvml.nvmlDeviceGetUtilizationRates(h).gpu))
                stop.wait(UTIL_S)
        except Exception as e:                        # a side measurement; record that it failed
            self.util.append((-1.0, -1))
            print("gpu util sampling unavailable:", repr(e), flush=True)

    def _admit(self, now: float, slot_free: list[int]) -> None:
        while self.pending and self.streams[self.pending[0]].arrival_s <= now:
            if self.admit_limit and len(self.live) >= self.admit_limit:
                break
            i = self.pending.pop(0)
            ls = self.streams[i]
            if self.admit_limit:
                ls.arrival_s = max(ls.arrival_s, now)      # its audio starts arriving when it is admitted
            if not slot_free:
                raise RuntimeError(f"all {self.n_slots} cache slots in use; raise max_slots")
            slot = slot_free.pop(0)
            self.slab_cc[:, slot].zero_()
            self.slab_ct[:, slot].zero_()
            self.slab_cl[slot] = 0
            s = Live(ls, self.langs[i], slot, self.hop, self.src_base[ls.src], self.src_len[ls.src])
            if self.lean is not None:
                self.lean.reset(slot)
            if self.mel == "full":
                s.feats = self._feats_full(s)
                if s.feats.size(-1) != s.n_frames:
                    raise RuntimeError(f"frame count {s.feats.size(-1)} != {s.n_frames}")
            self.live.append(s)

    def _gc_setup(self) -> None:
        """HARNESS_GC: 'default' (as shipped), 'disable' (gc.disable for the run), 'freeze' (gc.freeze then high
        thresholds). Always logs every collection (generation, duration, time) for the stall diagnosis."""
        import gc
        self.gc_events = []
        self._gc_t0 = None

        def cb(phase, info):
            if phase == "start":
                self._gc_t0 = time.perf_counter()
            elif self._gc_t0 is not None:
                self.gc_events.append((round(self.now(), 3), info.get("generation"),
                                       round((time.perf_counter() - self._gc_t0) * 1000, 2), info.get("collected")))
        gc.callbacks.append(cb)
        self._gc_cb = cb
        mode = os.environ.get("HARNESS_GC", "default")
        if mode == "disable":
            gc.collect()
            gc.disable()
        elif mode == "freeze":
            gc.collect()
            gc.freeze()
            gc.set_threshold(100000, 50, 1000)
        elif mode != "default":
            raise ValueError(f"HARNESS_GC={mode}")
        self.gc_mode = mode

    def _gc_teardown(self) -> None:
        import gc
        gc.callbacks.remove(self._gc_cb)
        if self.gc_mode == "disable":
            gc.enable()
        elif self.gc_mode == "freeze":
            gc.unfreeze()
            gc.set_threshold(700, 10, 10)

    def _trace_reinit(self) -> None:
        """record every CUDA-graph re-initialisation of NeMo's label-looping decoder (graph capture on batch growth)."""
        self.reinit_events = []
        comp = getattr(self.m.decoding.decoding, "decoding_computer", None)
        if comp is None or getattr(comp, "_live_traced", False):
            self._reinit_comp = comp
            return
        orig = comp._graph_reinitialize
        srv = self

        def wrapped(*args, **kw):
            t0 = time.perf_counter()
            r = orig(*args, **kw)
            torch.cuda.synchronize()
            x = args[0] if args else kw.get("encoder_output")
            srv.reinit_events.append((round(srv.now(), 3), tuple(x.shape[:2]) if x is not None else None,
                                      round((time.perf_counter() - t0) * 1000, 1)))
            return r
        comp._graph_reinitialize = wrapped
        comp._live_traced = True
        comp._live_srv_ref = lambda: srv
        self._reinit_comp = comp

    def prewarm(self) -> float:
        """Server warm-up: run NeMo's decoder once at the largest batch the server can form (all slots) so its
        CUDA graphs are captured before any audio arrives. Without it, NeMo re-captures them (100 to 400 ms, found
        6 Oct) every time continuous batching reaches a new maximum batch size, which stalls every live stream.
        NeMo is not changed; any server would warm up like this. HARNESS_PREWARM=0 turns it off."""
        t0 = time.perf_counter()
        T = int(os.environ.get("HARNESS_PREWARM_T", "0")) or int(self.cfg.valid_out_len) + 2
        sizes, b = [], 64
        while b < self.n_slots:                   # grow in steps (64, 128, ... then all slots)
            sizes.append(b)
            b *= 2
        sizes.append(self.n_slots)
        # HARNESS_PREWARM=0 skips only the decoder part (on the H100, decoder pre-warm at >= 384 slots crashed NeMo's
        # graphs, 6 Oct); the encoder and mel warm-up (HARNESS_WARM_ENC) is separate
        dec_sizes = sizes if os.environ.get("HARNESS_PREWARM", "1") == "1" else []
        if self.decoder == "fused":               # NeMo's decoder is never called; fused graphs are captured in init
            dec_sizes = []
        with torch.inference_mode():
            for nb in dec_sizes:
                x = torch.zeros(nb, self.m.encoder._feat_out, T, device="cuda")
                self.m.decoding.rnnt_decoder_predictions_tensor(
                    encoder_output=x, encoded_lengths=torch.full((nb,), T, device="cuda", dtype=torch.int64),
                    return_hypotheses=True)
                torch.cuda.synchronize()
        if os.environ.get("HARNESS_WARM_ENC", "1") == "1":
            self._warm_encoder_and_mel(sizes)
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    def _warm_encoder_and_mel(self, sizes: list[int]) -> None:
        """Warm the encoder step and the mel graph at every batch size the server can reach (6 Oct): in a fresh
        process the first call at each new largest batch pays one-off costs (TensorRT scratch growth and first
        execution at a new shape, mel-graph capture of a new bucket, allocator growth), all at the end of the arrival
        ramp, which made fresh-process trials fail on start-up transients. Zero inputs into scratch caches; the
        slab and the streams are untouched. HARNESS_WARM_ENC=0 turns it off."""
        L = self.slab_cc.shape[0]
        with torch.inference_mode():
            # largest size first, inputs as views of the persistent gather buffers (zeroed): every persistent buffer
            # (here and in the encoder step) is allocated once at its final size, so the warm-up neither holds a
            # second set of full-size caches nor leaves freed blocks of every intermediate size behind (7 Oct H100:
            # out of memory in this warm-up at 2,990 to 3,500 slots with 8 to 13 GB reserved but unallocated)
            for nb in sorted(set([min(8, self.n_slots)] + sizes), reverse=True):
                if self._slab_mode():
                    # on the slab itself (rows 0..nb-1; every slot is re-zeroed when a stream is admitted)
                    slots = torch.arange(nb, device="cuda")
                    for first in (True, False):
                        chunk, _, pre = self._sizes(first)
                        W = pre + chunk
                        sig = torch.zeros(nb, self.F, W, device="cuda")
                        lens = torch.full((nb,), W, device="cuda", dtype=torch.int64)
                        self.enc_step.slab_step(sig, lens, self.slab_cc, self.slab_ct, self.slab_cl, slots,
                                                0 if first else self.cfg.drop_extra_pre_encoded)
                    self._warm_mel(nb)
                    torch.cuda.synchronize()
                    continue
                cc, ct = self._gather_bufs(nb)
                cc.zero_()
                ct.zero_()
                cl = torch.zeros(nb, dtype=self.slab_cl.dtype, device="cuda")
                for first in (True, False):
                    chunk, _, pre = self._sizes(first)
                    W = pre + chunk
                    sig = torch.zeros(nb, self.F, W, device="cuda")
                    lens = torch.full((nb,), W, device="cuda", dtype=torch.int64)
                    self.enc_step(sig, lens, cc, ct, cl, 0 if first else self.cfg.drop_extra_pre_encoded)
                self._warm_mel(nb)
                torch.cuda.synchronize()
        if self.mel_graph is not None:            # outside inference_mode: the graph's input buffer is updated
            if True:                              # in place on every step (an inference tensor would refuse that)
                top = max(8, 1 << (self.n_slots - 1).bit_length())
                b = 8
                while b <= top:
                    for first in (True, False):
                        if (first, b) not in self.mel_graph.graphs:
                            self.mel_graph.graphs[(first, b)] = self.mel_graph._build(first, b)
                    b *= 2
                torch.cuda.synchronize()

    def _warm_mel(self, nb: int) -> None:
        """eager mel path (no mel graph): the preprocessor at this batch for the full window and for short windows
        like a stream's final chunk (8 Oct: first-time FFT plans and allocations must not happen while serving)"""
        if self.mel != "chunk" or self.mel_graph is not None:
            return
        for first in (True, False):
            chunk, _, pre = self._sizes(first)
            full = (chunk + pre + 1) * self.hop + self.half
            for L in sorted({full, max(self.half + 1, full // 2), self.half + self.hop}):
                x = torch.zeros(nb, L, device="cuda")
                self.pre(input_signal=x, length=torch.full((nb,), float(L), device="cuda"))

    def run(self) -> None:
        self.prewarm_s = self.prewarm()
        self._trace_reinit()
        self._gc_setup()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.t0 = time.perf_counter()
        self.vnow = 0.0
        stop = threading.Event()
        th = threading.Thread(target=self._sample_util, args=(stop,), daemon=True)
        th.start()
        slot_free = list(range(self.n_slots))
        while self.pending or self.live:
            now = self.now()
            if self.stop_at and now > self.stop_at:
                self.aborted = f"stopped at t={now:.1f} s (stop_at)"
                break
            self._admit(now, slot_free)
            rts = [self._ready_time(s) for s in self.live]
            ready = [s for s, t in zip(self.live, rts) if t <= now]
            if not ready:
                can_admit = self.pending and not (self.admit_limit and len(self.live) >= self.admit_limit)
                # (closed loop: a pending stream whose arrival has passed but has no free slot must not be a wake-up
                # time, or the virtual clock never advances; found 6 Oct on the H100 with a fast engine)
                nxt = rts + ([self.streams[self.pending[0]].arrival_s] if can_admit else [])
                t = min(nxt)
                if self.tick_s > 0 and self.live and t > now:
                    t = float(np.ceil(t / self.tick_s - 1e-9) * self.tick_s)    # next grid point at or after t
                self.wait_until(t)
                continue
            oldest = min(t for t in rts if t <= now)
            if self.abort_backlog_s > 0 and now - oldest > self.abort_backlog_s:
                self.aborted = f"backlog {now - oldest:.2f} s at t={now:.1f} s with {len(self.live)} live streams"
                break
            t_start = time.perf_counter()
            meta = [(s, rt, s.ls.arrival_s + s.idx * self.hop / SR, s.step == 0)
                    for s, rt in zip(self.live, rts) if rt <= now]
            groups: dict[tuple, list[Live]] = {}
            for s in ready:
                groups.setdefault((s.step == 0, s.lang), []).append(s)
            ev = {"mel0": [], "enc0": [], "dec0": [], "end": []}
            for (first, lang), g in sorted(groups.items(), key=lambda kv: (not kv[0][0], kv[0][1])):
                if hasattr(self.m, "set_inference_prompt") and lang != self.cur_lang:
                    self.m.set_inference_prompt(lang)
                    self.cur_lang = lang
                self._run_group(g, first, ev)
            torch.cuda.synchronize()
            wall = time.perf_counter() - t_start
            if self.clock_mode == "virtual":
                self.vnow = now + wall
            emit = self.now()
            for s, rt, fn, first in meta:
                s.chunks.append((rt, fn, emit, len(s.hyp.y_sequence) if s.hyp is not None else -1, first, now))
                if s.idx >= s.n_frames:
                    s.done = True
                    s.final_emit = emit
                    if self.lean is not None:               # text only at the end (outside the timed step)
                        from types import SimpleNamespace
                        txt, ids = self.lean.text(s.slot)
                        s.hyp = SimpleNamespace(text=txt, y_sequence=ids)
            split = {k: sum(a.elapsed_time(b) for a, b in zip(ev[k0], ev[k1]))
                     for k, k0, k1 in (("mel_ms", "mel0", "enc0"), ("enc_ms", "enc0", "dec0"), ("dec_ms", "dec0", "end"))}
            self.steps.append({"t": round(now, 4), "wall_s": wall, "B": len(ready),
                               "n_first": sum(1 for x in meta if x[3]), "n_calls": len(groups),
                               "n_live": len(self.live), **{k: round(v, 3) for k, v in split.items()}})
            if self.on_step is not None:
                self.on_step(self)
            done = [s for s in self.live if s.done]
            if done:
                self.live = [s for s in self.live if not s.done]
                for s in done:
                    self.finished.append(s)
                    slot_free.append(s.slot)
                    s.feats = None
        stop.set()
        th.join(timeout=2)
        self.peak_mem = torch.cuda.max_memory_allocated()
        self._gc_teardown()

    def summary(self, steady_chunk_s: float) -> dict:
        lat_ex, lat_in, miss, late_emit, total, final_ex, final_in = [], [], 0, 0, 0, [], []
        late_penult, late_wait = 0, 0
        waits = []
        for s in self.finished + self.live:
            for k, (rt, fn, em, _, _, st) in enumerate(s.chunks):
                total += 1
                lat_ex.append(em - rt)
                lat_in.append(em - fn)
                nxt = s.chunks[k + 1][0] if k + 1 < len(s.chunks) else rt + steady_chunk_s
                is_late = st > nxt + 1e-9      # behind real time: the next chunk was already complete at step start
                miss += is_late
                late_emit += em > nxt          # emitted after the next chunk was complete
                if is_late and k == len(s.chunks) - 2:
                    late_penult += 1           # the next chunk is the stream's short final remainder
                late_wait += st - rt > steady_chunk_s + 1e-9    # alternative: waited more than one chunk duration
                waits.append(st - rt)
        for s in self.finished:
            final_ex.append(s.final_emit - (s.ls.arrival_s + s.n_samples / SR))
            final_in.append(s.final_emit - s.chunks[-1][1])     # from the first new sample of the last chunk
        q = lambda v, p: float(np.percentile(v, p)) * 1000 if v else None
        walls = [x["wall_s"] for x in self.steps]
        stat = lambda v: {"p50": q(v, 50), "p95": q(v, 95), "max": q(v, 100)}
        return {
            "n_streams": len(self.streams), "n_finished": len(self.finished), "aborted": self.aborted,
            "n_steps": len(self.steps), "n_chunks": total, "deadline_misses": int(miss),
            "chunks_emitted_after_next_ready": int(late_emit),
            "late_second_to_last_chunk": int(late_penult), "late_by_wait_over_one_chunk": int(late_wait),
            "wait_ms": stat(waits),            # step start minus chunk ready: how far behind real time a chunk ran
            "chunk_latency_ms_excl_algo": stat(lat_ex), "chunk_latency_ms_incl_algo": stat(lat_in),
            "final_token_latency_ms_excl_algo": stat(final_ex), "final_token_latency_ms_incl_algo": stat(final_in),
            "step_wall_ms": stat(walls),
            "step_split_ms_mean": {k: float(np.mean([x[k] for x in self.steps])) if self.steps else None
                                   for k in ("mel_ms", "enc_ms", "dec_ms")},
            "batch_size": {"mean": float(np.mean([x["B"] for x in self.steps])) if self.steps else None,
                           "max": max((x["B"] for x in self.steps), default=0)},
            "peak_gpu_mem_allocated_gb": self.peak_mem / 1e9, "cache_slab_gb": self.slab_bytes / 1e9,
            "cache_bytes_per_stream_mb": self.slab_bytes / max(1, self.n_slots) / 1e6,
            "enc_slab_step": self._slab_mode(),
            "mel_graph": self.mel_graph is not None, "mel_check_max_abs_diff": getattr(self, "mel_check_max", None),
            "decoder_graph_reinits": self.reinit_events[:100], "decoder_graph_mode": dec_graph_mode(self.m), "prewarm_s": round(self.prewarm_s, 3),
            "fused_decoder": self.lean.stats() if hasattr(self.lean, "stats") else None,
            "gc_mode": self.gc_mode, "gc_gen2_count": sum(1 for e in self.gc_events if e[1] == 2),
            "gc_gen2_ms": [e[2] for e in self.gc_events if e[1] == 2][:50],
            "gc_gen2_t": [e[0] for e in self.gc_events if e[1] == 2][:50],
            "gc_total_ms": round(sum(e[2] for e in self.gc_events), 1),
            "steps_over_250ms": [(round(x["t"], 2), round(x["wall_s"] * 1000), x["B"]) for x in self.steps
                                 if x["wall_s"] > 0.25][:50],
            "gpu_util_mean_pct": float(np.mean([u for _, u in self.util if u >= 0])) if any(u >= 0 for _, u in self.util) else None,
        }


def run(out: str | None, model, sources, streams, langs, mel: str, clock: str, r: int, seed: int,
        extra: dict | None = None, tick_ms: float = 0.0, enc_step=None, abort_backlog_s: float = 0.0,
        write_steps: bool = True, cache_dtype: str = "float32", decoder: str = "stock",
        mel_graph: bool = False) -> dict:
    srv = Server(model, sources, streams, langs, mel, clock, tick_ms / 1000, enc_step, abort_backlog_s, cache_dtype,
                 decoder, mel_graph=mel_graph)
    srv.run()
    res = {"r": r, "chunk_ms": CHUNK_MS[r], "mel": mel, "clock": clock, "tick_ms": tick_ms, "seed": seed,
           "cache_dtype": cache_dtype, "dec_cuda_graphs": os.environ.get("HARNESS_DEC_CUDA_GRAPHS", "1") == "1",
           "dec_buckets": DEC_BUCKETS, "decoder": decoder,
           **(extra or {}), **srv.summary(CHUNK_MS[r] / 1000)}
    if out:
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "transcripts.jsonl"), "w") as f:
            for s in sorted(srv.finished, key=lambda s: s.ls.sid):
                f.write(json.dumps({"sid": s.ls.sid, "src": s.ls.src, "start": s.ls.start,
                                    "arrival_s": s.ls.arrival_s, "duration_s": s.n_samples / SR,
                                    "n_chunks": len(s.chunks), "pred_text": s.hyp.text if s.hyp is not None else "",
                                    "text": s.ls.text}, ensure_ascii=False) + "\n")
        if os.environ.get("HARNESS_CHUNKS", "0") == "1":    # per-chunk records: ready, first new sample, step
            with open(os.path.join(out, "chunks.jsonl"), "w") as f:     # start, emit (s), first-chunk flag
                for s in sorted(srv.finished, key=lambda s: s.ls.sid):
                    f.write(json.dumps({"sid": s.ls.sid, "chunks": [[round(c[0], 4), round(c[1], 4), round(c[5], 4),
                                        round(c[2], 4), int(c[4])] for c in s.chunks]}) + "\n")
        if write_steps:
            with open(os.path.join(out, "steps.jsonl"), "w") as f:
                for x in srv.steps:
                    f.write(json.dumps(x) + "\n")
        json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=1)
    del srv
    # no torch.cuda.empty_cache() here: it released memory still referenced by NeMo's CUDA-graph decoder, and the
    # next run's first decoder call then failed with "illegal memory access" (found 5 Oct; this was the whole crash)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--r", type=int, default=3)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--source", default="earnings22_full")
    ap.add_argument("--max-s", type=float, default=120)
    ap.add_argument("--stagger-s", type=float, default=30)
    ap.add_argument("--mel", default="chunk", choices=["full", "chunk"])
    ap.add_argument("--clock", default="real", choices=["real", "virtual"])
    ap.add_argument("--tick-ms", type=float, default=-1, help="step grid; -1 = the chunk duration, 0 = eager")
    ap.add_argument("--lang", default="en-US")
    ap.add_argument("--encoder", default="stock")
    ap.add_argument("--engine", default=None)
    ap.add_argument("--cache-dtype", default="float32", choices=["float32", "float16"])
    ap.add_argument("--decoder", default="stock", choices=["stock", "lean", "fused"])
    ap.add_argument("--mel-graph", action="store_true")
    ap.add_argument("--model", default="nvidia/nemotron-3.5-asr-streaming-0.6b")
    ap.add_argument("--seed", type=int, default=20261005)
    a = ap.parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    print(f"seed {a.seed}", flush=True)
    sources, streams = loadgen.from_specs(loadgen.make_streams(a.source, a.n, a.seed, a.stagger_s, a.max_s))
    m = load_model(a.model, a.r, a.lang, a.seed)
    enc = make_encoder_step(m, a.encoder, a.engine, a.r)
    tick = CHUNK_MS[a.r] if a.tick_ms < 0 else a.tick_ms
    res = run(a.out, m, sources, streams, [a.lang] * len(streams), a.mel, a.clock, a.r, a.seed,
              {"source": a.source, "n": a.n, "max_s": a.max_s, "stagger_s": a.stagger_s, "model": a.model,
               "encoder": a.encoder, "engine": a.engine,
               "note": "shared GPU: timing, latency, utilisation and miss counts are NOT reportable"}, tick, enc,
              cache_dtype=a.cache_dtype, decoder=a.decoder, mel_graph=a.mel_graph)
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
