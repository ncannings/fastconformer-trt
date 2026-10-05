"""Stock arm of the stock-versus-final comparison: the unmodified NeMo model (bf16 autocast, NeMo's CUDA-graph greedy
decoder, batch 32 sorted by length), any FastConformer model via ASR_MODEL (TDT or CTC). Reports WER on the chosen
sets and RTF over --reps timed passes of --rtf-set, with the timed window for the wall-power log.
usage: stock_rtf.py OUT_JSON [--wer-sets test_clean] [--wer-n 3000] [--rtf-set test_clean] [--reps 5]"""
from __future__ import annotations

import argparse
import json
import random
import time

import jiwer
import torch
from whisper_normalizer.english import EnglishTextNormalizer

import finetune_stride as fs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--wer-sets", default="test_clean")
    ap.add_argument("--wer-n", type=int, default=3000)
    ap.add_argument("--rtf-set", default="test_clean")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    import nemo.collections.asr as nemo_asr
    m = fs.load_model(nemo_asr, cuda_graphs=True).eval()
    is_ctc = not hasattr(m, "joint")

    def decode(sig, sl):
        f, fl = m.preprocessor(input_signal=sig, length=sl)
        e, l = m.encoder(audio_signal=f, length=fl)
        if is_ctc:
            hs = m.decoding.ctc_decoder_predictions_tensor(m.decoder(encoder_output=e), decoder_lengths=l,
                                                           return_hypotheses=True)
        else:
            hs = m.decoding.rnnt_decoder_predictions_tensor(encoder_output=e, encoded_lengths=l, return_hypotheses=True)
        hs = hs[0] if isinstance(hs, tuple) else hs
        return [h.text if isinstance(h.text, str) else "" for h in hs]

    res = {"model": fs.MODEL, "harness": "stock NeMo, bf16 autocast, CUDA-graph greedy decoder", "batch": a.batch}
    norm = EnglishTextNormalizer()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for s in [x for x in a.wer_sets.split(",") if x]:
            norm = fs.normaliser_for(s)
            rows = fs.rows_of(s)
            idx = list(range(len(rows)))
            random.Random(20261005).shuffle(idx)
            sel = [rows[i] for i in idx[:a.wer_n]]
            au = [fs.load_audio(r) for r in sel]
            od = sorted(range(len(sel)), key=lambda k: len(au[k]))
            hyps = [None] * len(sel)
            for b in range(0, len(od), a.batch):
                ids = od[b:b + a.batch]
                sig, sl = fs.batch_audio([au[k] for k in ids])
                for k, h in zip(ids, decode(sig, sl)):
                    hyps[k] = h
            pairs = [(norm(fs.ref_text(r)), norm(h) or "<empty>") for r, h in zip(sel, hyps)]
            pairs = [(r, h) for r, h in pairs if r]
            res[s] = 100 * jiwer.wer([r for r, _ in pairs], [h for _, h in pairs])
            print(s, res[s], flush=True)
        rows = fs.rows_of(a.rtf_set)
        auds = [fs.load_audio(r) for r in rows]
        audio_s = sum(len(w) for w in auds) / 16000
        order = sorted(range(len(auds)), key=lambda k: len(auds[k]))
        sigs = [fs.batch_audio([auds[k] for k in order[b:b + a.batch]]) for b in range(0, len(order), a.batch)]

        def run():
            torch.cuda.synchronize()
            t0 = time.time()
            for sig, sl in sigs:
                decode(sig, sl)
            torch.cuda.synchronize()
            return time.time() - t0
        run()
        t0 = time.time()
        walls = [run() for _ in range(a.reps)]
        res.update({"audio_s": audio_s, "walls_s": walls, "rtf": audio_s / sorted(walls)[len(walls) // 2],
                    "timed_window": [t0, time.time()]})
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in res if k not in ("walls_s", "timed_window")}), flush=True)


if __name__ == "__main__":
    main()
