# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Nigel Cannings. Part of fastconformer-trt (see LICENSE and NOTICE).
"""P0 environment check: versions, model load, language-prompt keys, preprocessor settings and the cache-aware
streaming geometry (chunk, shift, pre-encode cache, drop) for every att_context_size the model supports.
usage: python env_check.py [MODEL]"""
from __future__ import annotations

import json
import sys

import torch


def main() -> None:
    import nemo
    import nemo.collections.asr as nemo_asr
    name = sys.argv[1] if len(sys.argv) > 1 else "nvidia/nemotron-3.5-asr-streaming-0.6b"
    print("nemo", nemo.__version__, "torch", torch.__version__, "cuda", torch.version.cuda, flush=True)
    m = nemo_asr.models.ASRModel.from_pretrained(name, map_location="cpu")
    print("class", type(m).__name__)
    print("preprocessor", json.dumps({k: str(v) for k, v in m.cfg.preprocessor.items()}))
    enc = m.cfg.encoder
    print("encoder", {k: str(enc.get(k)) for k in ("att_context_size", "att_context_style", "att_context_probs",
                                                  "conv_context_size", "causal_downsampling", "subsampling",
                                                  "subsampling_factor", "n_layers", "d_model", "conv_kernel_size",
                                                  "self_attention_model")})
    print("decoding", json.dumps({k: str(v) for k, v in m.cfg.decoding.items()}))
    for k in ("prompt_dictionary", "model_defaults", "prompt"):
        if k in m.cfg:
            print(k, str(m.cfg[k])[:3000])
    for attr in ("prompt_dictionary", "_prompt_dict", "lang_dict"):
        if hasattr(m, attr):
            print("attr", attr, str(getattr(m, attr))[:3000])
    for ctx in ([56, 0], [56, 1], [56, 3], [56, 6], [56, 13]):
        m.encoder.set_default_att_context_size(att_context_size=ctx)
        m.encoder.setup_streaming_params()
        print(ctx, m.encoder.streaming_cfg)
    n = sum(p.numel() for p in m.parameters())
    print("params", n)


if __name__ == "__main__":
    main()
