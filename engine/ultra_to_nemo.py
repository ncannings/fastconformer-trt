"""Convert a Hugging Face transformers parakeet_tdt checkpoint of the parakeet-tdt-0.6b-v3 architecture (e.g.
moondream/parakeet-ultra, full precision) into NeMo: the v3 NeMo model with its weights replaced, saved as .nemo so
the whole pipeline (lean_export, trt_rtf, stock_rtf via ASR_MODEL=/data/<name>.nemo) runs it unchanged. Refuses
unless every NeMo parameter/buffer is filled exactly once with a matching shape (extra HF tensors such as a VAD head
are listed and ignored). usage: ultra_to_nemo.py HF_REPO OUT.nemo"""
import re
import sys

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

import nemo.collections.asr as nemo_asr

repo, out = sys.argv[1], sys.argv[2]
hf = load_file(snapshot_download(repo) + "/model.safetensors")
m = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
sd = m.state_dict()

RULES = [  # (HF regex, NeMo template)
    (r"encoder\.subsampling\.layers\.(\d+)\.(weight|bias)", r"encoder.pre_encode.conv.\1.\2"),
    (r"encoder\.subsampling\.linear\.(weight|bias)", r"encoder.pre_encode.out.\1"),
    (r"encoder\.layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)", r"encoder.layers.\1.self_attn.linear_q.\2"),
    (r"encoder\.layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)", r"encoder.layers.\1.self_attn.linear_k.\2"),
    (r"encoder\.layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)", r"encoder.layers.\1.self_attn.linear_v.\2"),
    (r"encoder\.layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)", r"encoder.layers.\1.self_attn.linear_out.\2"),
    (r"encoder\.layers\.(\d+)\.self_attn\.relative_k_proj\.(weight|bias)", r"encoder.layers.\1.self_attn.linear_pos.\2"),
    (r"encoder\.layers\.(\d+)\.self_attn\.bias_u", r"encoder.layers.\1.self_attn.pos_bias_u"),
    (r"encoder\.layers\.(\d+)\.self_attn\.bias_v", r"encoder.layers.\1.self_attn.pos_bias_v"),
    (r"encoder\.layers\.(\d+)\.conv\.norm\.(.+)", r"encoder.layers.\1.conv.batch_norm.\2"),
    (r"encoder\.layers\.(\d+)\.(.+)", r"encoder.layers.\1.\2"),
    (r"decoder\.embedding\.weight", r"decoder.prediction.embed.weight"),
    (r"decoder\.lstm\.(.+)", r"decoder.prediction.dec_rnn.lstm.\1"),
    (r"decoder\.decoder_projector\.(weight|bias)", r"joint.pred.\1"),
    (r"encoder_projector\.(weight|bias)", r"joint.enc.\1"),
    (r"joint\.head\.(weight|bias)", r"joint.joint_net.2.\1"),
]
new, filled, ignored = {}, set(), []
for k, v in hf.items():
    for pat, tpl in RULES:
        if re.fullmatch(pat, k):
            nk = re.sub(pat, tpl, k)
            break
    else:
        ignored.append(k)
        continue
    if nk not in sd:
        raise SystemExit(f"{k} -> {nk}: not a NeMo key")
    if nk in filled:
        raise SystemExit(f"{nk} filled twice")
    t = sd[nk]
    if v.shape != t.shape:
        if v.numel() == t.numel() and v.dim() + 1 == t.dim() and t.shape[-1] == 1:   # Linear -> 1x1 Conv1d
            v = v.reshape(t.shape)
        else:
            raise SystemExit(f"{k} {tuple(v.shape)} vs {nk} {tuple(t.shape)}")
    new[nk] = v.to(t.dtype)
    filled.add(nk)
missing = [k for k in sd if k not in filled and not k.endswith("num_batches_tracked")
           and not k.startswith("preprocessor.")]
if missing:
    raise SystemExit(f"{len(missing)} NeMo tensors not filled, e.g. {missing[:8]}")
sd.update(new)
m.load_state_dict(sd)
print(f"filled {len(filled)} tensors; ignored HF tensors: {sorted(set(re.sub(r'\.\d+\.', '.N.', k) for k in ignored))}")
m.save_to(out)
print("saved", out, flush=True)
