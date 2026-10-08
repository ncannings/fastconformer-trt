# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Nigel Cannings. Part of fastconformer-trt (see LICENSE and NOTICE).
"""TensorRT cache-aware encoder step for the live harness (P4): a drop-in for model.encoder.cache_aware_stream_step.

Hypothesis tested: a TensorRT engine (FP16, or FP8 PTQ on the linear layers) of the streaming encoder step gives the
same transcripts as stock NeMo at a lower step time. Ablate with the harness's --encoder stock.

Harness usage: --encoder /w/trt/trt_step.py:build --engine /out/trt/ml_r13/steady_fp8_b512.plan
The engine path names the steady-chunk engine; the first-chunk engine is the same name with "steady_" replaced by
"first_". If that file does not exist, or a call's chunk width or drop is not the one an engine was built for, the
call runs on the stock PyTorch encoder step instead and is counted (self.fallbacks, printed by report()); nothing is
silently approximated. Batches larger than the engine's maximum batch are split into engine-sized pieces.

Semantics: exactly encoder.cache_aware_stream_step(..., keep_all_outputs=False): the engine computes the step with all
outputs kept and this wrapper slices the encoder output to valid_out_len. Caches are float16 at the engine boundary:
float32 caches from the harness are cast on the way in and back to float32 on the way out (two extra elementwise
passes; a float16 slab avoids them, and caches in float16 are passed straight through).
"""
from __future__ import annotations

import os

import numpy as np
import torch

# TRT_LEGACY_CALL=1: the pre-7 Oct call path (per-call output allocation, torch.cat of the pieces), kept for the
# equivalence check of the persistent-buffer path
LEGACY = os.environ.get("TRT_LEGACY_CALL", "0") == "1"
IN = ["audio_signal", "length", "cache_last_channel", "cache_last_time", "cache_last_channel_len"]
OUTS = ["outputs", "encoded_lengths", "cache_last_channel_next", "cache_last_time_next", "cache_last_channel_next_len"]


def _torch_dtype(trt, dt):
    return {trt.float32: torch.float32, trt.float16: torch.float16, trt.bfloat16: torch.bfloat16,
            trt.int64: torch.int64, trt.int32: torch.int32}[dt]


class Engine:
    """One deserialised engine with one execution context, run on torch's current CUDA stream."""

    def __init__(self, path: str, shared: dict | None = None):
        """shared: buffers (persistent outputs, staging, TensorRT scratch) shared with the other engine of the same
        TrtStep. Safe because calls are serialised through the caller's stream (each call waits for it and it waits
        for the call), and every output is consumed before the next call."""
        import tensorrt as trt
        self.trt = trt
        self.path = path
        plug = os.environ.get("TRT_PLUGIN_LIB", "")
        if not plug and "ffn" in os.path.basename(path):     # FFNFp8 plugin engines (plugins/ffn_surgery.py)
            plug = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plugins", "libffn_fp8.so")
        if plug:
            import ctypes
            ctypes.CDLL(plug)
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, "")
        self.runtime = trt.Runtime(self.logger)
        with open(path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"could not deserialise {path}")
        # activation memory sized for the shapes of each call, not the profile maximum (B up to 1,024 at r=13 needs
        # about 4.5 GB; a small batch needs a fraction of it); the buffer only grows
        self.ctx = self.engine.create_execution_context(trt.ExecutionContextAllocationStrategy.USER_MANAGED)
        self.shared = shared if shared is not None else {}
        self.bufs: dict = self.shared.setdefault("bufs", {})
        self.stream = None
        self.dtype = {n: _torch_dtype(trt, self.engine.get_tensor_dtype(n)) for n in IN + OUTS}
        mn, opt, mx = self.engine.get_tensor_profile_shape("audio_signal", 0)
        self.max_b, self.width, self.min_t = int(mx[0]), int(mx[2]), int(mn[2])
        cap = int(os.environ.get("TRT_MAX_CALL_B", "0"))      # optional cap on rows per engine call (activation
        if cap > 0:                                           # memory; 6 Oct H100: 1,600 first chunks at r=13 asked
            self.max_b = min(self.max_b, cap)                 # for 33 GB of scratch)

    def _buf(self, key, shape, dtype) -> torch.Tensor:
        """a dense view of the given shape on a persistent flat buffer that only grows (allocated at the largest call,
        i.e. at the server's warm-up, not per call)"""
        n = int(np.prod(shape))
        key = (key, dtype)
        b = self.bufs.get(key)
        if b is None or b.numel() < n:
            self.bufs.pop(key, None)                      # free the smaller one first (no transient double)
            b = None
            with torch.inference_mode(False):            # normal tensor: written in and outside inference mode
                self.bufs[key] = b = torch.empty(n, dtype=dtype, device="cuda")
        return b[:n].view(shape)

    def __call__(self, sig, lens, cc, ct, cl, out=None):
        """out: optional dict of persistent output views to write into (by output name); otherwise outputs go to this
        engine's own persistent buffers. Inputs that are not dense in the engine's dtype, or that share storage with
        an output buffer, are staged in persistent input buffers (copy_ casts in place); nothing is allocated per
        call once the buffers have reached the largest batch."""
        if LEGACY:
            return self._call_legacy(sig, lens, cc, ct, cl)
        ctx = self.ctx
        B = sig.shape[0]
        obufs = {}
        for n in OUTS:
            if out is not None and n in out:
                obufs[n] = out[n]
        ins = []
        for n, t in zip(IN, [sig, lens, cc, ct, cl]):
            sp = t.untyped_storage().data_ptr()
            outs_ = [v for k, v in self.bufs.items() if k[0].startswith(("out_", "full_"))] + list(obufs.values())
            alias = any(sp == o.untyped_storage().data_ptr() for o in outs_)
            if t.dtype != self.dtype[n] or not t.is_contiguous() or alias:
                st = self._buf("in_" + n, tuple(t.shape), self.dtype[n])
                st.copy_(t)
                t = st
            ins.append(t)
            ctx.set_input_shape(n, tuple(t.shape))
            ctx.set_tensor_address(n, t.data_ptr())
        need = self.ctx.update_device_memory_size_for_shapes()
        if need > 0:
            self.ctx.set_device_memory(self._buf("scratch", (need,), torch.uint8).data_ptr(), need)
        outs = []
        for n in OUTS:
            shp = tuple(ctx.get_tensor_shape(n))
            if any(s < 0 for s in shp):
                raise RuntimeError(f"unresolved output shape {n} {shp}")
            o = obufs.get(n)
            if o is None:
                o = self._buf("out_" + n, shp, self.dtype[n])
            elif tuple(o.shape) != shp or o.dtype != self.dtype[n] or not o.is_contiguous():
                raise RuntimeError(f"output view for {n}: {tuple(o.shape)} {o.dtype}, engine gives {shp} {self.dtype[n]}")
            ctx.set_tensor_address(n, o.data_ptr())
            outs.append(o)
        cur = torch.cuda.current_stream()
        if cur.cuda_stream == 0:
            if self.stream is None:
                self.stream = torch.cuda.Stream()
            self.stream.wait_stream(cur)
            run_on = self.stream
        else:
            run_on = cur
        if not ctx.execute_async_v3(run_on.cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        if run_on is not cur:
            cur.wait_stream(run_on)
            for t in (sig, lens, cc, ct, cl):              # caller tensors read on the private stream
                t.record_stream(run_on)
        return outs

    def _call_legacy(self, sig, lens, cc, ct, cl):
        ins = [sig, lens, cc, ct, cl]
        ctx = self.ctx
        for n, t in zip(IN, ins):
            t = t if t.dtype == self.dtype[n] else t.to(self.dtype[n])
            t = t.contiguous()
            ins[IN.index(n)] = t
            ctx.set_input_shape(n, tuple(t.shape))
            ctx.set_tensor_address(n, t.data_ptr())
        need = self.ctx.update_device_memory_size_for_shapes()
        if need > 0:
            self.ctx.set_device_memory(self._buf("scratch", (need,), torch.uint8).data_ptr(), need)
        outs = []
        for n in OUTS:
            shp = tuple(ctx.get_tensor_shape(n))
            if any(s < 0 for s in shp):
                raise RuntimeError(f"unresolved output shape {n} {shp}")
            o = torch.empty(shp, dtype=self.dtype[n], device="cuda")
            ctx.set_tensor_address(n, o.data_ptr())
            outs.append(o)
        # TensorRT adds stream synchronisations when enqueued on the legacy default stream, which is where the
        # harness runs; enqueue on a private stream ordered after, and before, the caller's stream instead
        cur = torch.cuda.current_stream()
        if cur.cuda_stream == 0:
            if self.stream is None:
                self.stream = torch.cuda.Stream()
            self.stream.wait_stream(cur)
            run_on = self.stream
        else:
            run_on = cur
        if not ctx.execute_async_v3(run_on.cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        if run_on is not cur:
            cur.wait_stream(run_on)
            for t in ins + outs:                           # allocator: these tensors were used on the private stream
                t.record_stream(run_on)
        self._keep = ins                                   # inputs must live until the stream has consumed them
        return outs


class TrtStep:
    def __init__(self, model, steady_path: str, r: int):
        self.m = model
        cfg = model.encoder.streaming_cfg
        pick = lambda v, i: v[i] if isinstance(v, list) else v
        self.valid = int(cfg.valid_out_len)
        self.drop = int(cfg.drop_extra_pre_encoded)
        import re
        mr = re.search(r"_r(\d+)/", steady_path)
        if mr and int(mr.group(1)) != r:                    # an engine path for another chunk size: use this r's
            alt = steady_path.replace(mr.group(0), f"_r{r}/")
            if not os.path.exists(alt):
                raise FileNotFoundError(f"no engine for r={r}: {alt}")
            print(f"TrtStep: r={r}, using {alt} instead of {steady_path}", flush=True)
            steady_path = alt
        self.shared: dict = {}                              # one set of buffers for both engines
        self.steady = Engine(steady_path, self.shared)
        first_path = os.path.join(os.path.dirname(steady_path), os.path.basename(steady_path).replace("steady_", "first_"))
        self.first = Engine(first_path, self.shared) if first_path != steady_path and os.path.exists(first_path) else None
        self.w_steady = pick(cfg.chunk_size, 1) + pick(cfg.pre_encode_cache_size, 1)
        self.w_first = pick(cfg.chunk_size, 0) + pick(cfg.pre_encode_cache_size, 0)
        if self.steady.width != self.w_steady:
            raise ValueError(f"engine {steady_path} is for width {self.steady.width}, model r={r} needs {self.w_steady}")
        if self.first is not None and self.first.width != self.w_first:
            raise ValueError(f"engine {first_path} is for width {self.first.width}, model needs {self.w_first}")
        self.fallbacks = 0
        self.calls = 0
        import atexit
        atexit.register(lambda: print(self.report(), flush=True))     # harness runs print the fallback count at exit
        print(f"TrtStep: steady {steady_path} (max B {self.steady.max_b}), first "
              f"{first_path if self.first else 'stock PyTorch'}; valid_out_len {self.valid}", flush=True)

    def stock(self, sig, lens, cc, ct, cl, drop, keep_all):
        return self.m.encoder.cache_aware_stream_step(
            processed_signal=sig, processed_signal_length=lens, cache_last_channel=cc, cache_last_time=ct,
            cache_last_channel_len=cl, keep_all_outputs=keep_all, drop_extra_pre_encoded=drop)

    def __call__(self, sig, lens, cc, ct, cl, drop, keep_all_outputs=False):
        self.calls += 1
        W = sig.shape[-1]
        eng = self.steady if drop == self.drop and drop > 0 else (self.first if drop == 0 else None)
        if eng is None or W > eng.width or W < eng.min_t:
            self.fallbacks += 1
            return self.stock(sig, lens, cc, ct, cl, drop, keep_all_outputs)
        B = sig.shape[0]
        if B <= eng.max_b:
            e, el, ncc, nct, ncl = eng(sig, lens, cc, ct, cl)
        elif not LEGACY:
            # each piece runs into the engine's persistent piece buffers and is copied straight into its slice of
            # persistent full-batch outputs: no torch.cat, no transient full-size allocation (7 Oct: the cat asked
            # for 10 GB at 3,300 streams on the H100)
            full = None
            for i in range(0, B, eng.max_b):
                j = min(B, i + eng.max_b)
                outv = None
                if full is not None:                        # batch-first outputs: write straight into the full ones
                    outv = {"outputs": full[0][i:j], "encoded_lengths": full[1][i:j],
                            "cache_last_channel_next_len": full[4][i:j]}
                p = eng(sig[i:j], lens[i:j], cc[:, i:j], ct[:, i:j], cl[i:j], out=outv)
                if full is None:
                    T2 = p[0].shape[2]
                    full = [eng._buf("full_e", (B,) + tuple(p[0].shape[1:]), p[0].dtype),
                            eng._buf("full_el", (B,), p[1].dtype),
                            eng._buf("full_cc", (p[2].shape[0], B) + tuple(p[2].shape[2:]), p[2].dtype),
                            eng._buf("full_ct", (p[3].shape[0], B) + tuple(p[3].shape[2:]), p[3].dtype),
                            eng._buf("full_cl", (B,), p[4].dtype)]
                if p[0].shape[2] != T2:
                    raise RuntimeError(f"encoder output width differs between pieces: {p[0].shape[2]} vs {T2}")
                if outv is None:
                    full[0][i:j].copy_(p[0])
                    full[1][i:j].copy_(p[1])
                    full[4][i:j].copy_(p[4])
                full[2][:, i:j].copy_(p[2])               # caches are batch-second: piece buffers, then copied
                full[3][:, i:j].copy_(p[3])
            e, el, ncc, nct, ncl = full
        else:
            parts = [eng(sig[i:i + eng.max_b], lens[i:i + eng.max_b], cc[:, i:i + eng.max_b], ct[:, i:i + eng.max_b],
                         cl[i:i + eng.max_b]) for i in range(0, B, eng.max_b)]
            e, el, ncc, nct, ncl = (torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts]),
                                    torch.cat([p[2] for p in parts], 1), torch.cat([p[3] for p in parts], 1),
                                    torch.cat([p[4] for p in parts]))
        if not keep_all_outputs and self.valid > 0:          # streaming_post_process for chunked_limited
            e = e[:, :, :self.valid]
            el = torch.clamp(el, max=self.valid)
        # the caches come back in the caller's dtype; when that is the engine's dtype they are this engine's
        # persistent buffers, valid until its next call (the harness copies them into its slab at once)
        return (e.float().contiguous(), el.long(), ncc.to(cc.dtype), nct.to(ct.dtype), ncl.long())

    def slab_step(self, sig, lens, slab_cc, slab_ct, slab_cl, slots, drop):
        """Piece-wise step straight on the caller's cache slabs (8 Oct; harness HARNESS_SLAB_STEP=1): for each piece of
        at most max B rows, gather that piece's caches from the slab rows `slots` into piece-sized buffers, run the
        engine, and scatter the new caches straight back into the same slab rows. No full-batch gather or output
        buffer, so the per-stream memory is the slab alone (about 3.1 MB at r=13, float16). Same arithmetic as
        __call__ on gathered caches (pieces are disjoint rows, and a piece's inputs are gathered before its outputs
        are written). Returns (encoded, encoded_len) as __call__ does. Falls back to __call__ plus a scatter when no
        engine fits the call or the slab dtype differs from the engine's (counted as fallbacks in the first case)."""
        self.calls += 1
        W = sig.shape[-1]
        eng = self.steady if drop == self.drop and drop > 0 else (self.first if drop == 0 else None)
        if (eng is None or W > eng.width or W < eng.min_t or slab_cc.dtype != eng.dtype["cache_last_channel"]
                or slab_ct.dtype != eng.dtype["cache_last_time"]):
            self.calls -= 1
            cc = slab_cc.index_select(1, slots)
            ct = slab_ct.index_select(1, slots)
            cl = slab_cl.index_select(0, slots)
            e, el, cc2, ct2, cl2 = self(sig, lens, cc, ct, cl, drop)
            slab_cc.index_copy_(1, slots, cc2)
            slab_ct.index_copy_(1, slots, ct2)
            slab_cl.index_copy_(0, slots, cl2)
            return e, el
        B = sig.shape[0]
        mb = eng.max_b
        full_e = full_el = None
        for i in range(0, B, mb):
            j = min(B, i + mb)
            b = j - i
            sl = slots[i:j]
            cc = eng._buf("pc_cc", (slab_cc.shape[0], b) + tuple(slab_cc.shape[2:]), slab_cc.dtype)
            ct = eng._buf("pc_ct", (slab_ct.shape[0], b) + tuple(slab_ct.shape[2:]), slab_ct.dtype)
            torch.index_select(slab_cc, 1, sl, out=cc)
            torch.index_select(slab_ct, 1, sl, out=ct)
            cl = slab_cl.index_select(0, sl)
            outv = None
            if full_e is not None:
                outv = {"outputs": full_e[i:j], "encoded_lengths": full_el[i:j]}
            p = eng(sig[i:j], lens[i:j], cc, ct, cl, out=outv)
            if full_e is None:
                full_e = eng._buf("full_e", (B,) + tuple(p[0].shape[1:]), p[0].dtype)
                full_el = eng._buf("full_el", (B,), p[1].dtype)
                full_e[i:j].copy_(p[0])
                full_el[i:j].copy_(p[1])
            elif p[0].shape[2] != full_e.shape[2]:
                raise RuntimeError(f"encoder output width differs between pieces: {p[0].shape[2]} vs {full_e.shape[2]}")
            slab_cc.index_copy_(1, sl, p[2])
            slab_ct.index_copy_(1, sl, p[3])
            slab_cl.index_copy_(0, sl, p[4].to(slab_cl.dtype))
        e, el = full_e, full_el
        if self.valid > 0:                                  # keep_all_outputs=False, as __call__
            e = e[:, :, :self.valid]
            el = torch.clamp(el, max=self.valid)
        return e.float().contiguous(), el.long()

    def report(self) -> str:
        return f"TrtStep calls {self.calls}, stock fallbacks {self.fallbacks}"


def build(model, engine_path: str, r: int):
    """live/harness.py ENCODER INTERFACE entry point."""
    return TrtStep(model, engine_path, r)


def patch_encoder(model, engine_path: str, r: int) -> TrtStep:
    """Route model.encoder.cache_aware_stream_step through the engine (for NeMo's own streaming script)."""
    step = TrtStep(model, engine_path, r)

    def cass(processed_signal, processed_signal_length=None, cache_last_channel=None, cache_last_time=None,
             cache_last_channel_len=None, keep_all_outputs=True, drop_extra_pre_encoded=None, bypass_pre_encode=False):
        if cache_last_channel is None or bypass_pre_encode:
            step.fallbacks += 1
            return step.stock(processed_signal, processed_signal_length, cache_last_channel, cache_last_time,
                              cache_last_channel_len, drop_extra_pre_encoded, keep_all_outputs)
        drop = step.drop if drop_extra_pre_encoded is None else drop_extra_pre_encoded
        return step(processed_signal, processed_signal_length, cache_last_channel, cache_last_time,
                    cache_last_channel_len, drop, keep_all_outputs)
    model.encoder.cache_aware_stream_step = cass
    return step


def patch_class(engine_path: str) -> None:
    """Route every ConformerEncoder.cache_aware_stream_step through an engine built lazily on first use (for NeMo's
    stock streaming script, which loads the model itself). The steps are kept in STEPS for reporting."""
    import types
    from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder
    orig = ConformerEncoder.cache_aware_stream_step

    def cass(self, processed_signal, processed_signal_length=None, cache_last_channel=None, cache_last_time=None,
             cache_last_channel_len=None, keep_all_outputs=True, drop_extra_pre_encoded=None, bypass_pre_encode=False):
        if not hasattr(self, "_trt_step"):
            self._trt_step = TrtStep(types.SimpleNamespace(encoder=_Unpatched(self, orig)), engine_path,
                                     self.att_context_size[1])
            STEPS.append(self._trt_step)
        st = self._trt_step
        if cache_last_channel is None or bypass_pre_encode:
            st.fallbacks += 1
            return orig(self, processed_signal, processed_signal_length, cache_last_channel, cache_last_time,
                        cache_last_channel_len, keep_all_outputs, drop_extra_pre_encoded, bypass_pre_encode)
        drop = st.drop if drop_extra_pre_encoded is None else drop_extra_pre_encoded
        return st(processed_signal, processed_signal_length, cache_last_channel, cache_last_time,
                  cache_last_channel_len, drop, keep_all_outputs)
    ConformerEncoder.cache_aware_stream_step = cass


class _Unpatched:
    """the encoder as TrtStep sees it: its streaming config and the original (stock) step"""

    def __init__(self, enc, orig):
        self._enc, self._orig = enc, orig

    @property
    def streaming_cfg(self):
        return self._enc.streaming_cfg

    def cache_aware_stream_step(self, **kw):
        return self._orig(self._enc, **kw)


STEPS: list = []
