# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Nigel Cannings. Part of fastconformer-trt (see LICENSE and NOTICE).
# Wraps NVIDIA NeMo's (Apache-2.0) encoder.cache_aware_stream_step for ONNX export.
"""Shared pieces for the TensorRT cache-aware encoder step (P4): model loading as the stock streaming script does it,
an export wrapper around the encoder's streaming step, and stock NeMo chunk iteration for calibration and checks.

The engine computes exactly encoder.cache_aware_stream_step(..., keep_all_outputs=True) for one fixed value of
drop_extra_pre_encoded (it is a Python int baked into the graph), so each chunk size gets two engines: "first" (the
first chunk of a stream, drop 0, NeMo's first-chunk width) and "steady" (every later chunk). The valid_out_len
truncation (keep_all_outputs=False) is a slice done outside the engine. The language prompt projection stays outside
(model._apply_prompt_to_encoded, as conformer_stream_step applies it). Cache inputs and outputs keep NeMo's layout,
[layers, B, T, D] and [layers, B, D, K], with the batch on axis 1, so a harness slab can be indexed directly; they are
float16 at the engine boundary (cast to float32 inside the graph) to halve cache traffic and memory.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch

SEED = 20261005
MODELS = {"ml": "nvidia/nemotron-3.5-asr-streaming-0.6b", "en": "nvidia/nemotron-speech-streaming-en-0.6b"}
CHUNK_MS = {0: 80, 1: 160, 3: 320, 6: 560, 13: 1120}
OUT = os.path.join(os.environ.get("LIVE_OUT", "/out"), "trt")


def seed_all(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"seed {seed}", flush=True)


def load_model(name: str, r: int, lang: str | None = "en-US"):
    """As the stock streaming script and live/harness.py: float32, TF32 matmuls, default greedy RNN-T batch decoding."""
    import nemo.collections.asr as nemo_asr
    from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("high")
    m = nemo_asr.models.ASRModel.from_pretrained(name, map_location=torch.device("cuda"))
    m.encoder.set_default_att_context_size(att_context_size=[56, r])
    m.change_decoding_strategy(RNNTDecodingConfig(fused_batch_size=-1))
    if hasattr(m, "set_inference_prompt") and lang is not None:
        m.set_inference_prompt(lang)
        m.decoding.set_strip_lang_tags(True, lang_tag_pattern=None)
    m = m.to(device=torch.device("cuda"), dtype=torch.float32).eval()
    return m


def geometry(enc) -> dict:
    c = enc.streaming_cfg
    pick = lambda v, i: v[i] if isinstance(v, list) else v
    g = {"first_w": pick(c.chunk_size, 0) + pick(c.pre_encode_cache_size, 0),
         "steady_w": pick(c.chunk_size, 1) + pick(c.pre_encode_cache_size, 1),
         "drop": int(c.drop_extra_pre_encoded), "valid_out_len": int(c.valid_out_len),
         "cache_len": int(c.last_channel_cache_size), "conv_cache": int(enc.conv_context_size[0]),
         "layers": len(enc.layers), "d_model": int(enc.d_model), "feat_in": int(enc._feat_in),
         "chunk_size": c.chunk_size, "shift_size": c.shift_size, "pre_encode_cache_size": c.pre_encode_cache_size,
         "att_context_size": list(enc.att_context_size), "att_context_style": enc.att_context_style,
         "self_attention_model": enc.self_attention_model}
    return g


class StepExport(torch.nn.Module):
    """encoder streaming step with a fixed drop_extra_pre_encoded; caches float16 at the boundary, all outputs kept."""

    def __init__(self, enc, drop: int):
        super().__init__()
        self.enc = enc
        self.drop = drop

    def forward(self, audio_signal, length, cache_last_channel, cache_last_time, cache_last_channel_len):
        enc = self.enc
        prev = enc.streaming_cfg.drop_extra_pre_encoded
        enc.streaming_cfg.drop_extra_pre_encoded = self.drop
        try:
            e, el, cc, ct, cl = enc.forward_internal(audio_signal, length, cache_last_channel=cache_last_channel.float(),
                                                     cache_last_time=cache_last_time.float(),
                                                     cache_last_channel_len=cache_last_channel_len)
            e, el, cc, ct, cl = enc.streaming_post_process((e, el, cc, ct, cl), keep_all_outputs=True)
        finally:
            enc.streaming_cfg.drop_extra_pre_encoded = prev
        return e, el, cc.half(), ct.half(), cl


def stock_chunks(model, audios: list[np.ndarray]):
    """Yield (step, chunk_audio, chunk_lengths, is_last) for a batch of utterances exactly as NeMo's stock streaming
    script feeds them (CacheAwareStreamingAudioBuffer, online_normalization False, pad_and_drop_preencoded False)."""
    from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
    buf = CacheAwareStreamingAudioBuffer(model=model, online_normalization=False, pad_and_drop_preencoded=False)
    for a in audios:
        buf.append_audio(a, stream_id=-1)
    for k, (ch, ln) in enumerate(iter(buf)):
        yield k, ch, ln, buf.is_buffer_empty()


def dev_audios(set_name: str, n: int, seed: int = SEED) -> list[np.ndarray]:
    """n utterances of a training-free set (LibriSpeech dev-clean or a FLEURS dev set), seeded shuffle."""
    import finetune_stride as fs
    rows = fs.rows_of(set_name)
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    return [fs.load_audio(rows[i]) for i in idx[:n]]
