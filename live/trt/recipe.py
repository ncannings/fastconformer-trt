# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Nigel Cannings. Part of fastconformer-trt (see LICENSE and NOTICE).
"""FP8 PTQ recipes for the streaming encoder (ModelOpt 0.44), shared by export_step.py (ONNX for TensorRT),
fq_step.py (fake-quantised PyTorch step for fast WER screening in the harness) and sensitivity.py.

Hypothesis tested: the FP8 encoder's small WER loss (about +1.5% relative at 1.12 s, +0.9% at 80 ms on the Spark,
+1.9% and +1.8% on the H100) comes from a few sensitive linear layers and from activation scales set by the single
largest value seen in calibration, so keeping the sensitive layers in FP16 and/or choosing activation scales by a
percentile or by FP8 rounding error over more (training-free) calibration audio restores a real margin.
Ablate by recipe "base" (the part 1 and part 2 recipe, unchanged).

A recipe is a name from RECIPES or a spec string "key=value;key=value":
  data   comma list of set:count, sets from dev_clean, dev_other, fleurs_dev_en_us (never a test set)
  act    max (ModelOpt max calibration) | p<percentile> (e.g. p99.999) | mse (per-tensor FP8 rounding-error search
         over a reservoir sample of each input) ; weights always per-tensor max as before
  skip   comma list of quantizer name patterns kept in FP16, e.g. layers.0.,layers.23.,linear_out ; shorthand
         L<i> = layers.<i>. (the trailing dot keeps L1 from matching layers.10)
  batch  calibration batch size (default 16)
The default skips (convolutions, subsampling, LayerNorm inputs) always apply.
"""
from __future__ import annotations

import copy
import fnmatch

import numpy as np
import torch

RECIPES = {
    "base": "data=dev_clean:64;act=max",
    "big": "data=dev_clean:256,dev_other:128,fleurs_dev_en_us:128;act=max;batch=32",
    "p99999": "data=dev_clean:256,dev_other:128,fleurs_dev_en_us:128;act=p99.999;batch=32",
    "p9999": "data=dev_clean:256,dev_other:128,fleurs_dev_en_us:128;act=p99.99;batch=32",
    "p99999s": "data=dev_clean:256,dev_other:128,fleurs_dev_en_us:128;act=p99.999;batch=32;skip=L20,L21",
    "s2021": "data=dev_clean:64;act=max;skip=L20,L21",
    "amse": "data=dev_clean:256,dev_other:128,fleurs_dev_en_us:128;act=mse;batch=32",
}


def parse(spec: str) -> dict:
    spec = RECIPES.get(spec, spec)
    r = {"data": [("dev_clean", 64)], "act": "max", "skip": [], "batch": 16, "spec": spec}
    for part in [p for p in spec.split(";") if p.strip()]:
        k, _, v = part.partition("=")
        k, v = k.strip(), v.strip()
        if k == "data":
            r["data"] = [(s.split(":")[0], int(s.split(":")[1])) for s in v.split(",")]
        elif k == "act":
            r["act"] = v
        elif k == "skip":
            r["skip"] = [f"layers.{x[1:]}." if x[:1] == "L" and x[1:].isdigit() else x for x in v.split(",") if x]
        elif k == "batch":
            r["batch"] = int(v)
        else:
            raise ValueError(f"unknown recipe key {k!r} in {spec!r}")
    for s, _ in r["data"]:
        if "test" in s or not s.startswith(("dev_", "fleurs_dev_")):
            raise ValueError(f"calibration set {s} is not a training-free dev set")
    return r


def calib_audio(rec: dict, seed: int) -> list[np.ndarray]:
    import common as C
    out = []
    for k, (s, n) in enumerate(rec["data"]):
        out += C.dev_audios(s, n, seed + 1000 * k)
    return sorted(out, key=len)


def modelopt_cfg():
    import modelopt.torch.quantization as mtq
    cfg = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
    off = [{"quantizer_name": "*conv*", "enable": False}, {"quantizer_name": "*pre_encode*", "enable": False},
           {"quantizer_name": "*norm*", "enable": False}]
    if isinstance(cfg["quant_cfg"], list):
        cfg["quant_cfg"] += off
    else:
        for e in off:
            cfg["quant_cfg"][e["quantizer_name"]] = {"enable": False}
    return cfg


def _input_quantizers(enc):
    from modelopt.torch.quantization.nn import TensorQuantizer
    return [(n, m) for n, m in enc.named_modules()
            if isinstance(m, TensorQuantizer) and n.endswith("input_quantizer") and m.is_enabled]


def fp8_fake(x: torch.Tensor, amax: float) -> torch.Tensor:
    s = amax / 448.0
    return (x / s).clamp(-448, 448).to(torch.float8_e4m3fn).float() * s


def _refine_activations(enc, rec, loop, log=print) -> dict:
    """After max calibration: collect a reservoir sample of |x| per input quantizer (plus its max) and set amax by
    a percentile or by the FP8 rounding-error minimum (multipliers 0.30 to 1.00 of the max, step 0.02)."""
    qs = _input_quantizers(enc)
    per_call = 8192
    cap = 1 << 20
    samples = {n: [] for n, _ in qs}
    gens = {}
    hooks = []
    for n, m in qs:
        def hook(mod, args, n=n):
            x = args[0].detach().float().abs().reshape(-1)
            if sum(t.numel() for t in samples[n]) >= cap:
                return
            k = min(per_call, x.numel())
            g = gens.get(x.device)
            if g is None:
                g = gens[x.device] = torch.Generator(device=x.device).manual_seed(20261005)
            idx = torch.randint(0, x.numel(), (k,), device=x.device, generator=g)
            samples[n].append(x[idx])
        hooks.append(m.register_forward_pre_hook(hook))
    for _, m in qs:
        m.disable_quant()                                  # statistics of the float inputs
    try:
        loop(enc)
    finally:
        for h in hooks:
            h.remove()
        for _, m in qs:
            m.enable_quant()
    changed = {}
    act = rec["act"]
    for n, m in qs:
        x = torch.cat(samples[n])
        old = float(m.amax)
        if act.startswith("p"):
            q = float(act[1:]) / 100.0
            k = max(1, int(round((1.0 - q) * x.numel())))         # quantile by top-k (torch.quantile caps size)
            new = min(old, float(torch.topk(x, k).values[-1]))
        elif act == "mse":
            best, new = None, old
            xs = x                                         # |x|: FP8 rounding is symmetric
            for mult in np.arange(0.30, 1.0001, 0.02):
                a = old * float(mult)
                err = float(((fp8_fake(xs, a) - xs) ** 2).mean())
                if best is None or err < best:
                    best, new = err, a
        else:
            raise ValueError(f"act={act}")
        m.amax = torch.tensor(new, device=m.amax.device, dtype=m.amax.dtype)
        changed[n] = (old, new)
    ratios = [b / a for a, b in changed.values() if a > 0]
    log(f"activation amax refined ({act}) on {len(changed)} quantizers: new/max ratio min {min(ratios):.3f}, "
        f"median {float(np.median(ratios)):.3f}, max {max(ratios):.3f}")
    return changed


def quantize(model, rec: dict | str, run_stream, seed: int = 20261005, log=print) -> dict:
    """Quantise model.encoder in place. run_stream(model, audios) streams a batch through the stock step."""
    import modelopt.torch.quantization as mtq
    if isinstance(rec, str):
        rec = parse(rec)
    cal = calib_audio(rec, seed)
    log(f"recipe {rec['spec']!r}: {len(cal)} calibration utterances, {sum(map(len, cal)) / 16000:.0f} s, batches of "
        f"{rec['batch']}, act {rec['act']}, FP16 kept for {rec['skip'] or 'nothing extra'}")

    def loop(_enc):
        with torch.no_grad():
            for b in range(0, len(cal), rec["batch"]):
                run_stream(model, cal[b:b + rec["batch"]])
    mtq.quantize(model.encoder, modelopt_cfg(), loop)
    if rec["act"] != "max":
        _refine_activations(model.encoder, rec, loop, log)
    from modelopt.torch.quantization.nn import TensorQuantizer
    off = 0
    for n, m in model.encoder.named_modules():
        if isinstance(m, TensorQuantizer) and m.is_enabled and any(p in n for p in rec["skip"]):
            m.disable()
            off += 1
    nq = len(_input_quantizers(model.encoder))
    log(f"FP16 kept: {off} quantizers disabled by skip patterns; enabled input quantizers now {nq}")
    return rec
