"""Check LeanEncoder against NeMo's encoder (fp32, 4 dev-clean utterances), then export ONNX: plain, and with
ModelOpt FP8 PTQ on the linear layers (128 dev utterances calibration; convolutions not quantised).
usage: lean_export.py OUT_DIR"""
import copy
import os
import random
import sys

import torch

import finetune_stride as fs
from lean_encoder import LeanEncoder

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
import nemo.collections.asr as nemo_asr
m = fs.asr_model(nemo_asr).cuda().eval()
m.preprocessor.featurizer.dither = 0.0
m.preprocessor.featurizer.pad_to = 0
CKPT = os.environ.get("LEAN_CKPT", "")          # fine-tuned (finetune_stride.py) weights; sets the pool layer
CK_MERGES = None
CK_SKIP = []
if CKPT:
    ck = torch.load(CKPT, map_location="cuda", weights_only=True)
    sd = m.state_dict()
    sd.update(ck["trained"])
    m.load_state_dict(sd)
    CK_SKIP = list(ck["drop"])                       # dropped layers are skipped (identity), as in training
    CK_MERGES = fs.merges_for(ck["pool_at"], [], ck.get("rule", "pool2"), ck.get("pools"))
    print("loaded", CKPT, "pools", {k: v.__name__ for k, v in CK_MERGES.items()}, "dropped layers", CK_SKIP, flush=True)
PRE1 = os.environ.get("LEAN_PREMASK_ONCE", "0") == "1"
HALF = os.environ.get("LEAN_HALF", "0") == "1"
DWS = os.environ.get("LEAN_DWSHIFT", "0") == "1"
RSH = os.environ.get("LEAN_RELSHIFT", "0") == "1"
P43 = int(os.environ.get("LEAN_POOL43", "-1"))
HB = os.environ.get("LEAN_HB", "0") == "1"
QKV_PLUGIN = os.environ.get("LEAN_QKV_PLUGIN", "0") == "1"
QKV = os.environ.get("LEAN_QKV", "0") == "1" or QKV_PLUGIN
HB = HB or QKV_PLUGIN
SUBM = os.environ.get("LEAN_SUBMASK", "0") == "1"
lean = LeanEncoder(m.encoder, premask_once=PRE1, dw_shift=DWS, rel_shift=RSH, pool43_after=P43,
                   heads_first=HB, fused_qkv=QKV, pools=CK_MERGES, skip=CK_SKIP, sub_masked=SUBM).cuda().eval()
tag = ("_ck" if CKPT else "") + ("_pm1" if PRE1 else "") + ("_h" if HALF else "") + ("_dws" if DWS else "") + ("_rs" if RSH else "") + (f"_p43l{P43}" if P43 >= 0 else "") + ("_hb" if HB else "") + ("_qkv" if QKV else "") + ("p" if QKV_PLUGIN else "") + ("_attn" if os.environ.get("LEAN_ATTN_PLUGIN", "0") == "1" else "") + ("_sm" if SUBM else "")
rows = fs.rows_of("dev_clean")
with torch.inference_mode():
    sig, sl = fs.batch_audio([fs.load_audio(r) for r in rows[:4]])
    f, fl = m.preprocessor(input_signal=sig, length=sl)
    e0, l0 = m.encoder(audio_signal=f, length=fl)
    e1, l1 = lean(f, fl)
    if CK_MERGES is not None:                                       # fine-tuned pooled model: against its reference
        import encoder_merge as em
        e0, l0 = em.encode_multi(m, f, fl, CK_MERGES, CK_SKIP)
        valid = (torch.arange(e0.shape[2], device=e0.device)[None] < l0[:, None])[:, None, :]
        print("EQUIV (fine-tuned, pooled) max abs diff", float(((e0 - e1).abs() * valid).max()), "ref max",
              float(e0.abs().max()), "lengths equal", bool(torch.equal(l0, l1)), flush=True)
    elif P43 >= 0:                                                  # pooled output is shorter: no equivalence
        print("POOLED lengths", l0.tolist(), "->", l1.tolist(), flush=True)
    else:
        valid = (torch.arange(e0.shape[2], device=e0.device)[None] < l0[:, None])[:, None, :]
        print("EQUIV max abs diff", float(((e0 - e1).abs() * valid).max()), "ref max", float(e0.abs().max()),
              "lengths equal", bool(torch.equal(l0, l1)), flush=True)


def export(model, path):
    nf = int(m.cfg.preprocessor.features)                         # 128 mel (0.6B v3), 80 (1.1B models)
    f_ex = torch.randn(2, nf, 1600, device="cuda", dtype=torch.float16 if HALF else torch.float32)
    l_ex = torch.tensor([1600, 1200], device="cuda")
    torch.onnx.export(model, (f_ex, l_ex), path, input_names=["audio_signal", "length"],
                      output_names=["outputs", "encoded_lengths"], opset_version=17, dynamo=False,
                      dynamic_axes={"audio_signal": {0: "B", 2: "T"}, "length": {0: "B"},
                                    "outputs": {0: "B", 2: "T2"}, "encoded_lengths": {0: "B"}})
    print("exported", path, flush=True)


if HALF:
    lean = lean.half()
with torch.no_grad():
    export(lean, os.path.join(out, f"lean{tag}.onnx"))
import modelopt.torch.quantization as mtq
random.Random(20261009).shuffle(rows)
CALIB = os.environ.get("LEAN_CALIB", "")   # "" : 128 English dev-clean; "multi": 32 dev-clean + 4 per FLEURS dev language
if CALIB == "multi":
    import glob
    cal = rows[:32]
    for f in sorted(glob.glob(os.path.join(fs.DATA, "fleurs_dev_*.parquet"))):
        lr = fs.rows_of(os.path.basename(f)[:-8])
        random.Random(20261009).shuffle(lr)
        cal += lr[:4]
    print("calibration: multilingual,", len(cal), "utterances", flush=True)
else:
    cal = rows[:128]
auds = sorted([fs.load_audio(r) for r in cal], key=len)
batches = [fs.batch_audio(auds[b:b + 16]) for b in range(0, len(auds), 16)]
QUANT = os.environ.get("LEAN_QUANT", "fp8")       # fp8 | nvfp4 | nvfp4ff (NVFP4 in the feed-forward modules, FP8 elsewhere)
cfg = copy.deepcopy(mtq.NVFP4_DEFAULT_CFG if QUANT == "nvfp4" else mtq.FP8_DEFAULT_CFG)
if QUANT == "nvfp4ff":
    fp4 = copy.deepcopy(mtq.NVFP4_DEFAULT_CFG)["quant_cfg"]
    for k in ("*weight_quantizer", "*input_quantizer"):
        cfg["quant_cfg"]["*feed_forward*" + k] = fp4[k]
cfg["quant_cfg"]["*conv*"] = {"enable": False}
cfg["quant_cfg"]["*pre_encode*"] = {"enable": False}


def loop(enc):
    with torch.no_grad():
        for s_, l_ in batches:
            ff, ffl = m.preprocessor(input_signal=s_, length=l_)
            enc(ff.half() if HALF else ff, ffl)
mtq.quantize(lean, cfg, loop)
ATTN_PLUGIN = os.environ.get("LEAN_ATTN_PLUGIN", "0") == "1"
if QKV_PLUGIN:
    lean.prepare_qkv_plugin()                  # emit QKVHeads nodes in the FP8 export (after calibration)
if ATTN_PLUGIN:
    lean.attn_plugin = True                    # RelPosAttn, with q/k/v from QKVHeads or from TensorRT's own GEMM
    if os.environ.get("LEAN_POSCACHE", "0") == "1":
        lean.prepare_pos_cache()
        with torch.no_grad():   # exact check of the slicing: cached rows vs a full-precision projection of NeMo's pos_emb
            worst = 0.0
            for T in (40, 333, 750):
                _, pe_t = lean.pos_enc(x=torch.zeros(1, T, lean.layers[0].self_attn.linear_pos.in_features, device="cuda"))
                o = lean.pos_L - T
                for li, (pc, cc) in lean.pos_cache.items():
                    att = lean.layers[li].self_attn
                    w = att.linear_pos.weight.float()
                    pr = torch.nn.functional.linear(pe_t[0].float(), w).view(-1, att.h, att.d_k).permute(1, 0, 2)
                    worst = max(worst, float((pc[:, o:o + 2 * T - 1].float() - pr).abs().max() / pr.abs().max()))
            print(f"POSCACHE slice check: worst relative max diff {worst:.2e} over T=40/333/750, all layers", flush=True)
            if worst > 2e-3:
                raise SystemExit("position cache rows do not match NeMo's pos_emb projection")
        print("position table cached:", len(lean.pos_cache), "layers", flush=True)
from contextlib import nullcontext
from modelopt.torch.quantization.export_onnx import configure_linear_module_onnx_quantizers
# NVFP4: activations exported as TensorRT dynamic block quantisation, weights static (then fp4_post.py)
with torch.no_grad(), (configure_linear_module_onnx_quantizers(lean) if "nvfp4" in QUANT else nullcontext()):
    export(lean, os.path.join(out, f"lean{tag}_{QUANT}.onnx"))
if QKV_PLUGIN:
    import numpy as np
    from lean_encoder import qkv_side_data
    np.savez(os.path.join(out, f"lean{tag}_{QUANT}.qkv.npz"), **qkv_side_data())
    print("saved QKVHeads side data", flush=True)
