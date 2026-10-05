"""Which LeanEncoder option breaks equivalence for a given ASR_MODEL: max abs diff vs NeMo's encoder (fp32, 4
dev-clean utterances) for the plain lean forward and each option on its own. usage: lean_equiv_probe.py"""
import torch

import finetune_stride as fs
from lean_encoder import LeanEncoder

import nemo.collections.asr as nemo_asr
m = nemo_asr.models.ASRModel.from_pretrained(fs.MODEL).cuda().eval()
m.preprocessor.featurizer.dither = 0.0
m.preprocessor.featurizer.pad_to = 0
rows = fs.rows_of("dev_clean")
with torch.inference_mode():
    sig, sl = fs.batch_audio([fs.load_audio(r) for r in rows[:4]])
    f, fl = m.preprocessor(input_signal=sig, length=sl)
    e0, l0 = m.encoder(audio_signal=f, length=fl)
    valid = (torch.arange(e0.shape[2], device=e0.device)[None] < l0[:, None])[:, None, :]
    for name, kw in [("plain", {}), ("premask_once", dict(premask_once=True)), ("dw_shift", dict(dw_shift=True)),
                     ("rel_shift", dict(rel_shift=True)), ("heads_first", dict(heads_first=True)),
                     ("fused_qkv", dict(heads_first=True, fused_qkv=True))]:
        e1, l1 = LeanEncoder(m.encoder, **kw).cuda().eval()(f, fl)
        print(f"{name:13s} max abs diff {float(((e0 - e1).abs() * valid).max()):.5f} ref max {float(e0.abs().max()):.3f} "
              f"lengths equal {bool(torch.equal(l0, l1))}", flush=True)
