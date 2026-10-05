"""Accuracy probe before any NVFP4 kernel work: ModelOpt fake quantisation of the stock encoder's linears, FP8
everywhere (the deployed scheme) against NVFP4 in the feed-forward linears only, on the 2,000-utterance dev-clean and
Earnings-22 samples (seed 20261005). Calibration: 128 dev-clean utterances (seed 20261009), as lean_export.py.
Hypothesis: FP4 GEMMs run at about twice the FP8 rate at lower energy per operation on the power-limited GB10, and the
FF blocks are about 45% of the encoder, so NVFP4 FF would pay if feed-forward-only quantisation holds WER (all-linear
NVFP4 did not: dev-clean 1.68 -> 2.31 on 400 utterances).
usage: nvfp4_probe.py OUT_JSON VARIANT     VARIANT in fp8 | ff | ff_keep2 | ff_awq"""
from __future__ import annotations

import copy
import json
import random
import sys

import jiwer
import torch
from whisper_normalizer.english import EnglishTextNormalizer

import finetune_stride as fs

out, variant = sys.argv[1], sys.argv[2]
import modelopt.torch.quantization as mtq
import nemo.collections.asr as nemo_asr

m = fs.load_model(nemo_asr, cuda_graphs=True).eval()
nl = len(m.encoder.layers)
base = mtq.NVFP4_AWQ_LITE_CFG if variant == "ff_awq" else mtq.FP8_DEFAULT_CFG
cfg = copy.deepcopy(base)
fp8 = copy.deepcopy(mtq.FP8_DEFAULT_CFG)["quant_cfg"]
fp4 = copy.deepcopy(mtq.NVFP4_DEFAULT_CFG)["quant_cfg"]
if variant == "ff_awq":                                  # AWQ-lite base (NVFP4 everywhere), then non-FF back to FP8
    for k in ("*weight_quantizer", "*input_quantizer"):
        for mod in ("self_attn", "conv"):
            cfg["quant_cfg"][f"*{mod}*{k}"] = fp8[k]
if variant in ("ff", "ff_keep2"):
    for k in ("*weight_quantizer", "*input_quantizer"):
        cfg["quant_cfg"]["*feed_forward*" + k] = fp4[k]
if variant == "ff_keep2":                                # first and last two layers' FF stay FP8
    for l in (0, 1, nl - 2, nl - 1):
        for k in ("*weight_quantizer", "*input_quantizer"):
            cfg["quant_cfg"][f"*layers.{l}.feed_forward*" + k] = fp8[k]
cfg["quant_cfg"]["*conv*"] = {"enable": False}           # as the deployed engine: convolutions not quantised
cfg["quant_cfg"]["*pre_encode*"] = {"enable": False}

rows = fs.rows_of("dev_clean")
cal = rows[:]
random.Random(20261009).shuffle(cal)
auds = sorted([fs.load_audio(r) for r in cal[:128]], key=len)
batches = [fs.batch_audio(auds[b:b + 16]) for b in range(0, len(auds), 16)]


def loop(enc):
    with torch.no_grad():
        for s_, l_ in batches:
            f, fl = m.preprocessor(input_signal=s_, length=l_)
            enc(audio_signal=f, length=fl)


mtq.quantize(m.encoder, cfg, loop)
res = {"variant": variant}
norm = EnglishTextNormalizer()
with torch.inference_mode():
    for s in ("dev_clean", "earnings22"):
        rr = fs.rows_of(s)
        idx = list(range(len(rr)))
        random.Random(20261005).shuffle(idx)
        sel = [rr[i] for i in idx[:2000]]
        au = [fs.load_audio(r) for r in sel]
        od = sorted(range(len(sel)), key=lambda k: len(au[k]))
        hyps = [None] * len(sel)
        for b in range(0, len(od), 32):
            ids = od[b:b + 32]
            sig, sl = fs.batch_audio([au[k] for k in ids])
            f, fl = m.preprocessor(input_signal=sig, length=sl)
            e, l = m.encoder(audio_signal=f, length=fl)
            hs = m.decoding.rnnt_decoder_predictions_tensor(encoder_output=e, encoded_lengths=l, return_hypotheses=True)
            hs = hs[0] if isinstance(hs, tuple) else hs
            for k, h in zip(ids, hs):
                hyps[k] = h.text
        pairs = [(norm(fs.ref_text(r)), norm(h) or "<empty>") for r, h in zip(sel, hyps)]
        pairs = [(r, h) for r, h in pairs if r]
        res[s] = 100 * jiwer.wer([r for r, _ in pairs], [h for _, h in pairs])
        print(variant, s, round(res[s], 3), flush=True)
json.dump(res, open(out, "w"), indent=1)
