"""Step 3: parakeet-tdt-0.6b-v3 with the encoder running in a TensorRT engine (trt_export.py + trtexec), the
preprocessor and greedy TDT decoding (CUDA-graph decoder) in NeMo. Weights unchanged. Reports RTF on dev-clean (batch 32)
and WER on the fixed 400-utterance dev-clean / Earnings-22 samples.
--pipeline: preprocessing + encoder run in a producer thread on one CUDA stream and the decoder consumes on a second
stream, so decoding batch i overlaps encoding batch i+1 (the serial path's wall time is the sum of the three stages).
usage: trt_rtf.py ENGINE OUT_JSON [--reps 2] [--pipeline]"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import queue
import random
import threading
import time

import jiwer
import tensorrt as trt
import torch
import torch.nn.functional as Fn
from whisper_normalizer.english import EnglishTextNormalizer

import finetune_stride as fs

DT = {trt.float32: torch.float32, trt.float16: torch.float16, trt.bfloat16: torch.bfloat16,
      trt.int64: torch.int64, trt.int32: torch.int32, trt.bool: torch.bool}


class TRTEncoder:
    def __init__(self, path: str):
        lib = os.environ.get("FFN_PLUGIN_LIB")                  # engines with FFNFp8 nodes (plugins/)
        if lib:
            ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
        self.rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.eng = self.rt.deserialize_cuda_engine(open(path, "rb").read())
        self.ctx = self.eng.create_execution_context()
        self.names = [self.eng.get_tensor_name(i) for i in range(self.eng.num_io_tensors)]
        self.inputs = [n for n in self.names if self.eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.outputs = [n for n in self.names if self.eng.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        print({n: (str(self.eng.get_tensor_dtype(n)), str(self.eng.get_tensor_shape(n))) for n in self.names}, flush=True)

    def __call__(self, feats: torch.Tensor, lengths: torch.Tensor):
        vals = {"audio_signal": feats, "length": lengths}
        for n in self.inputs:
            t = vals[n].to(DT[self.eng.get_tensor_dtype(n)]).contiguous()
            vals[n] = t
            self.ctx.set_input_shape(n, tuple(t.shape))
            self.ctx.set_tensor_address(n, t.data_ptr())
        outs = {}
        for n in self.outputs:
            shp = tuple(self.ctx.get_tensor_shape(n))
            o = torch.empty(shp, dtype=DT[self.eng.get_tensor_dtype(n)], device="cuda")
            outs[n] = o
            self.ctx.set_tensor_address(n, o.data_ptr())
        self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        o = [outs[n] for n in self.outputs]
        return o[0], o[1]


TRACE = [] if os.environ.get("PIPE_TRACE") else None      # pipeline timeline (debug; adds syncs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("engine")
    ap.add_argument("out")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--batch", type=int, default=32, help="encoder (TensorRT) batch")
    ap.add_argument("--dec-batch", type=int, default=32, help="utterances per decoder call")
    ap.add_argument("--wer-sets", default="dev_clean,earnings22")
    ap.add_argument("--wer-n", type=int, default=400)
    ap.add_argument("--rtf-set", default="dev_clean")
    ap.add_argument("--pipeline", action="store_true")
    ap.add_argument("--green", type=int, default=0, help="with --pipeline: give the decoder a green context of this "
                    "many SMs and the encoder the rest (green_streams.py), so decoding runs alongside encoding")
    ap.add_argument("--frame-budget", type=int, default=0, help="encoder sub-batches packed to this many padded encoder "
                    "frames (80 ms) instead of --batch utterances, up to --max-batch (needs an engine built for it)")
    ap.add_argument("--max-batch", type=int, default=128)
    ap.add_argument("--ckpt", default="", help="fine-tuned checkpoint (finetune_stride.py): its decoder and joint "
                    "weights are loaded (the encoder is the engine, exported from the same checkpoint)")
    ap.add_argument("--dec-priority", action="store_true", help="decoder stream at high CUDA priority, so its many tiny "
                    "kernels are scheduled between the encoder's blocks instead of queueing behind them")
    ap.add_argument("--dec-streams", type=int, default=1, help="with --pipeline: decoder instances (own CUDA-graph "
                    "state, stream and thread) consuming batches concurrently; the decoder is many tiny kernels")
    ap.add_argument("--fast-joint", action="store_true", help="fast_joint.py: fused joint + argmax Triton kernels in "
                    "NeMo's TDT greedy decoder (2 kernels instead of about 7 per joint evaluation)")
    ap.add_argument("--dump-hyps", action="store_true", help="store the normalised transcripts in the output json")
    ap.add_argument("--fast-hyps", action="store_true", help="fast_hyps.py: copy only the used prefix of the decoder's "
                    "buffers to the host, one sync per batch (same outputs)")
    ap.add_argument("--pre-compile", action="store_true", help="torch.compile the mel featurizer (fuses its "
                    "elementwise and reduction kernels; 2.6x faster preprocessing)")
    ap.add_argument("--cpu-pre", action="store_true", help="mel features on the CPU in a worker thread, ahead of the "
                    "GPU (with --pipeline); the GPU is the bottleneck, so CPU preprocessing overlaps it")
    ap.add_argument("--dec-bf16", action="store_true", help="prediction and joint network weights in bf16 (fp32 joint "
                    "GEMMs at every step are most of the decoder's GPU time; NeMo disables autocast inside decoding)")
    a = ap.parse_args()
    import nemo.collections.asr as nemo_asr
    m = fs.load_model(nemo_asr, cuda_graphs=True).eval()
    enc = TRTEncoder(a.engine)
    if a.ckpt:
        ck = torch.load(a.ckpt, map_location="cuda", weights_only=True)
        sd = m.state_dict()
        sd.update(ck["trained"])
        m.load_state_dict(sd)
        print("loaded decoder/joint from", a.ckpt, flush=True)
    if a.fast_joint:
        import fast_joint
        fast_joint.install()
    if a.fast_hyps:
        import fast_hyps
        fast_hyps.install()
    if a.pre_compile:
        m.preprocessor.featurizer = torch.compile(m.preprocessor.featurizer, dynamic=True)
    if a.dec_bf16:
        m.decoder.to(torch.bfloat16)
        if hasattr(m, "joint"):
            m.joint.to(torch.bfloat16)

    def sub_batches(sl):
        """[(start, end)] over a length-sorted group: --batch utterances each, or with --frame-budget as many as fit
        (count x longest <= budget encoder frames, at most --max-batch)."""
        n = sl.shape[0]
        if not a.frame_budget:
            return [(b, min(b + a.batch, n)) for b in range(0, n, a.batch)]
        fr = (sl.cpu() // 1280 + 1).tolist()
        out, b = [], 0
        while b < n:
            e = b + 1
            while e < n and e - b < a.max_batch and (e + 1 - b) * max(fr[b:e + 1]) <= a.frame_budget:
                e += 1
            out.append((b, e))
            b = e
        return out

    def encode(sig, sl):
        """Preprocess and encode in TensorRT batches of a.batch; one padded tensor for the whole decoder batch."""
        es, ls = [], []
        for b0, b1 in sub_batches(sl):
            sb = sl[b0:b1]
            # trim the sub-batch to its own longest utterance (the group's audio is padded to the group's longest),
            # but keep at least 1 s (the engine profile's minimum is 100 feature frames)
            f, fl = m.preprocessor(input_signal=sig[b0:b1, :max(int(sb.max()), 16000)], length=sb)
            e, l = enc(f, fl)
            es.append(e)
            ls.append(l)
        T = max(e.shape[2] for e in es)
        dt = torch.bfloat16 if a.dec_bf16 else torch.float32
        return torch.cat([Fn.pad(x, (0, T - x.shape[2])) for x in es]).to(dt), torch.cat(ls).long()

    is_ctc = not hasattr(m, "joint")

    def dec(e, l):
        if is_ctc:                                                   # CTC models: decoder conv + greedy
            lp = m.decoder(encoder_output=e)
            hs = m.decoding.ctc_decoder_predictions_tensor(lp, decoder_lengths=l, return_hypotheses=True)
            hs = hs[0] if isinstance(hs, tuple) else hs
            return [h.text if isinstance(h.text, str) else "" for h in hs]
        hs = m.decoding.rnnt_decoder_predictions_tensor(encoder_output=e, encoded_lengths=l, return_hypotheses=True)
        hs = hs[0] if isinstance(hs, tuple) else hs
        return [h.text if isinstance(h.text, str) else "" for h in hs]

    def decode(sig, sl):
        return dec(*encode(sig, sl))

    pre_cpu = None
    if a.cpu_pre:
        pre_cpu = m.from_config_dict(m.cfg.preprocessor).cpu().float().eval()     # fresh CPU module (a moved copy
        pre_cpu.featurizer.dither = m.preprocessor.featurizer.dither              # keeps its filterbank on the GPU)
        pre_cpu.featurizer.pad_to = m.preprocessor.featurizer.pad_to
        assert pre_cpu.featurizer.fb.device.type == "cpu"

    def feats_cpu(sig, sl):
        """Whole decoder batch on the CPU -> pinned features (uploaded by the producer)."""
        with torch.inference_mode():
            f, fl = pre_cpu(input_signal=sig, length=sl)
        return f.pin_memory(), fl

    def encode_feats(f, fl):
        """Encode precomputed features in TensorRT batches of a.batch, each trimmed to its own longest utterance."""
        es, ls = [], []
        for b0 in range(0, f.shape[0], a.batch):
            fb, lb = f[b0:b0 + a.batch], fl[b0:b0 + a.batch]
            e, l = enc(fb[:, :, :int(lb.max())], lb)
            es.append(e)
            ls.append(l)
        T = max(e.shape[2] for e in es)
        dt = torch.bfloat16 if a.dec_bf16 else torch.float32
        return torch.cat([Fn.pad(x, (0, T - x.shape[2])) for x in es]).to(dt), torch.cat(ls).long()

    s_enc = torch.cuda.Stream()
    s_dec = torch.cuda.Stream(priority=-1) if a.dec_priority else torch.cuda.Stream()
    if a.green:
        import green_streams
        s_enc, s_dec, _, split = green_streams.make_streams(a.green)
        print("green contexts: encoder / decoder SMs", split, flush=True)
    from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTBPEDecoding
    # extra decoder instances: fresh decoding objects (own CUDA-graph state) sharing the stateless networks
    decodings = [m.decoding] + ([] if not hasattr(m, "joint") else
                                [RNNTBPEDecoding(decoding_cfg=m.cfg.decoding, decoder=m.decoder, joint=m.joint,
                                                 tokenizer=m.tokenizer) for _ in range(a.dec_streams - 1)])
    dec_streams = [s_dec] + [torch.cuda.Stream() for _ in range(a.dec_streams - 1)]

    def decode_all(batches):
        """Texts for each (sig, sl) batch; serial, or pipelined over two streams with --pipeline."""
        if not a.pipeline:
            return [decode(sig, sl) for sig, sl in batches]
        q: queue.Queue = queue.Queue(maxsize=2)
        err = []

        def produce():
            try:
                with torch.inference_mode(), torch.cuda.stream(s_enc):
                    if pre_cpu is not None:
                        from concurrent.futures import ThreadPoolExecutor
                        pool = ThreadPoolExecutor(1)
                        futs = [pool.submit(feats_cpu, sig.cpu() if sig.is_cuda else sig, sl.cpu())
                                for sig, sl in batches]
                    for i, (sig, sl) in enumerate(batches):
                        if pre_cpu is not None:
                            f, fl = futs[i].result()
                            e, l = encode_feats(f.to("cuda", non_blocking=True), fl.to("cuda", non_blocking=True))
                        else:
                            e, l = encode(sig, sl)
                        ev = torch.cuda.Event()
                        ev.record(s_enc)
                        while not err:                               # stop if a consumer died (no deadlock)
                            try:
                                if TRACE is not None: ev.synchronize(); TRACE.append(("enc_done", time.time()))
                                q.put((e, l, ev), timeout=1.0)
                                break
                            except queue.Full:
                                pass
                        if err:
                            return
            except Exception as ex:                                  # surface producer failures, never hang
                err.append(ex)
            q.put(None)
        out = {}

        def consume(decoding, stream):
            try:
                with torch.inference_mode(), torch.cuda.stream(stream):
                    while True:
                        with take:
                            item = q.get()
                            if item is None:
                                q.put(None)                          # let the other consumers finish too
                                return
                            idx = counter[0]
                            counter[0] += 1
                        e, l, ev = item
                        if TRACE is not None: TRACE.append(("dec_start", time.time()))
                        stream.wait_event(ev)
                        e.record_stream(stream)
                        l.record_stream(stream)
                        if is_ctc:
                            out[idx] = dec(e, l)
                            continue
                        hs = decoding.rnnt_decoder_predictions_tensor(encoder_output=e, encoded_lengths=l,
                                                                      return_hypotheses=True)
                        hs = hs[0] if isinstance(hs, tuple) else hs
                        out[idx] = [h.text if isinstance(h.text, str) else "" for h in hs]
                        if TRACE is not None: TRACE.append(("dec_done", time.time()))
            except Exception as ex:
                err.append(ex)
        if not is_ctc:
            # capture every decoder instance's CUDA graphs now, at the largest shape, one at a time: a capture while
            # other threads launch work fails (e.g. torch.compile recompiling the featurizer in the producer), and the
            # graphs are re-captured only when the shape grows
            Tmax = max(int(sl.max()) for _, sl in batches) // 1280 + 2      # encoder frames (8 x 10 ms hops)
            Bmax = max(sig.shape[0] for sig, _ in batches)
            dt = torch.bfloat16 if a.dec_bf16 else torch.float32
            for d_ in decodings:
                e0 = torch.zeros(Bmax, m.cfg.encoder.d_model, Tmax, device="cuda", dtype=dt)
                l0 = torch.full((Bmax,), Tmax, device="cuda", dtype=torch.long)
                d_.rnnt_decoder_predictions_tensor(encoder_output=e0, encoded_lengths=l0, return_hypotheses=True)
            torch.cuda.synchronize()
        take, counter = threading.Lock(), [0]
        th = threading.Thread(target=produce)
        th.start()
        cons = [threading.Thread(target=consume, args=(d, st)) for d, st in zip(decodings, dec_streams)]
        for c in cons:
            c.start()
        for c in cons:
            c.join()
        th.join()
        torch.cuda.synchronize()
        if err:
            raise err[0]
        return [out[i] for i in range(len(out))]

    res = {"engine": a.engine, "batch": a.batch, "dec_batch": a.dec_batch, "pipeline": a.pipeline, "dec_bf16": a.dec_bf16, "cpu_pre": a.cpu_pre, "pre_compile": a.pre_compile, "fast_hyps": a.fast_hyps, "ckpt": a.ckpt, "dec_streams": a.dec_streams, "dec_priority": a.dec_priority, "fast_joint": a.fast_joint, "green": a.green, "frame_budget": a.frame_budget, "max_batch": a.max_batch}
    norm = EnglishTextNormalizer()
    with torch.inference_mode():
        for s in [x for x in a.wer_sets.split(",") if x]:
            norm = fs.normaliser_for(s)
            rr = fs.rows_of(s)
            idx = list(range(len(rr)))
            random.Random(20261005).shuffle(idx)
            sel = [rr[i] for i in idx[:a.wer_n]]
            au = [fs.load_audio(r) for r in sel]
            od = sorted(range(len(sel)), key=lambda k: len(au[k]))
            hyps = [None] * len(sel)
            groups = [od[b:b + a.dec_batch] for b in range(0, len(od), a.dec_batch)]
            for ids, texts in zip(groups, decode_all([fs.batch_audio([au[k] for k in ids]) for ids in groups])):
                for k, h in zip(ids, texts):
                    hyps[k] = h
            pairs = [(norm(fs.ref_text(r)), norm(h) or "<empty>") for r, h in zip(sel, hyps)]
            pairs = [(r, h) for r, h in pairs if r]                 # empty references (some Earnings-22 rows) excluded
            res[s] = 100 * jiwer.wer([r for r, _ in pairs], [h for _, h in pairs])
            res.setdefault("hyps", {})[s] = [h for _, h in pairs] if a.dump_hyps else None
            print(s, res[s], flush=True)
        rows = fs.rows_of(a.rtf_set)
        auds = [fs.load_audio(r) for r in rows]
        audio_s = sum(len(w) for w in auds) / 16000
        order = sorted(range(len(auds)), key=lambda k: len(auds[k]))
        sigs = [fs.batch_audio([auds[k] for k in order[b:b + a.dec_batch]]) for b in range(0, len(order), a.dec_batch)]
        if a.cpu_pre:                                              # audio starts in host memory
            sigs = [(sg.cpu(), sl.cpu()) for sg, sl in sigs]

        def run():
            torch.cuda.synchronize()
            t0 = time.time()
            decode_all(sigs)
            torch.cuda.synchronize()
            return time.time() - t0
        run()
        t0 = time.time()
        walls = [run() for _ in range(a.reps)]
        if TRACE is not None:
            TRACE.clear(); run()
            t0_ = TRACE[0][1]
            print("TRACE", [(k, round(t - t0_, 3)) for k, t in TRACE], flush=True)
        st = {"pre": 0.0, "enc": 0.0, "dec": 0.0}
        for sig, sl in sigs:                                       # one pass with synchronised stage timing
            sig, sl = sig.cuda(), sl.cuda()                        # (GPU preprocessing in this diagnostic pass)
            es, ls = [], []
            for b0, b1 in sub_batches(sl):
                torch.cuda.synchronize(); t1 = time.time()
                sb = sl[b0:b1]
                f, fl = m.preprocessor(input_signal=sig[b0:b1, :max(int(sb.max()), 16000)], length=sb)
                torch.cuda.synchronize(); t2 = time.time()
                e_, l_ = enc(f, fl)
                torch.cuda.synchronize(); t3 = time.time()
                st["pre"] += t2 - t1; st["enc"] += t3 - t2
                es.append(e_); ls.append(l_)
            T = max(x.shape[2] for x in es)
            e_ = torch.cat([Fn.pad(x, (0, T - x.shape[2])) for x in es]).to(torch.bfloat16 if a.dec_bf16 else torch.float32)
            l_ = torch.cat(ls).long()
            torch.cuda.synchronize(); t4 = time.time()
            dec(e_, l_)
            torch.cuda.synchronize(); st["dec"] += time.time() - t4
        res["stages_s"] = st
        res.update({"audio_s": audio_s, "walls_s": walls, "rtf": audio_s / sorted(walls)[len(walls) // 2],
                    "timed_window": [t0, time.time()]})
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k in ("dev_clean", "earnings22", "test_clean", "rtf", "stages_s")}), flush=True)


if __name__ == "__main__":
    main()
