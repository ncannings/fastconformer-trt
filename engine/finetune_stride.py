"""Faster Parakeet: fixed frame-rate reduction inside the encoder (stride-2 average pooling after chosen layers),
optionally with layers removed, fine-tuned by self-distillation so it stays as accurate (PREREG_SPEED.md).

Train: the student is parakeet-tdt-0.6b-v3 with pool2 after each layer in --pool-at (and --drop layers skipped);
targets are the frozen unmodified model's own greedy transcripts (self-distillation, any audio, never a test set);
NeMo's TDT loss; layers from the first pooling point on, plus decoder and joint, are trained. Runs in the NGC NeMo
container (run_in_container.sh).
Eval: WER against the reference on the fixed 400-utterance dev-clean and Earnings-22 samples (seed 20261005),
batched decoding.
RTF: real-time factor of the full pipeline (preprocessor, encoder, greedy TDT decoding) at batch 32, length-sorted,
on all of dev-clean, for the modified model and the unmodified one through the same code path.

usage:
  python finetune_stride.py train OUT --pool-at 4 [--drop 0,2] --epochs 2 [--data train100] [--hours 3]
  python finetune_stride.py eval OUT/ckpt.pt OUT/eval.json            (ckpt 'base' = unmodified model)
  python finetune_stride.py rtf OUT/ckpt.pt OUT/rtf.json [--set dev_clean]
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import random
import re
import time

import jiwer
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
from scipy.signal import resample_poly
from whisper_normalizer.english import EnglishTextNormalizer

import encoder_merge as em

DATA = os.environ.get("ASR_DATA", os.path.expanduser("~/asr_data"))
MODEL = os.environ.get("ASR_MODEL", "nvidia/parakeet-tdt-0.6b-v3")     # other FastConformer models via ASR_MODEL


def load_audio(row) -> np.ndarray:
    wav, sr = sf.read(io.BytesIO(row["audio"]["bytes"]))
    if wav.ndim > 1:
        wav = wav.mean(1)
    if sr != 16000:
        g = np.gcd(sr, 16000)
        wav = resample_poly(wav, 16000 // g, sr // g)
    return wav.astype(np.float32)


def normaliser_for(set_name: str):
    """Whisper's English normaliser for English sets; the language-neutral basic normaliser (lowercase, punctuation
    and symbols to spaces, no spelling rules) for the non-English FLEURS sets (fleurs_<code>)."""
    if set_name.startswith("fleurs_") and not set_name.startswith("fleurs_en"):
        from whisper_normalizer.basic import BasicTextNormalizer
        return BasicTextNormalizer()
    return EnglishTextNormalizer()


def ref_text(row) -> str:
    t = row.get("text") or row.get("transcription") or ""
    return re.sub(r"<[^>]*>", " ", t)


def asr_model(nemo_asr):
    """ASR_MODEL as a hub name, or a local .nemo file (e.g. a converted third-party checkpoint, ultra_to_nemo.py)."""
    if MODEL.endswith(".nemo"):
        return nemo_asr.models.ASRModel.restore_from(MODEL)
    return nemo_asr.models.ASRModel.from_pretrained(MODEL)


def load_model(nemo_asr, cuda_graphs: bool = False):
    m = asr_model(nemo_asr).cuda()
    m.preprocessor.featurizer.dither = 0.0
    m.preprocessor.featurizer.pad_to = 0
    from omegaconf import open_dict
    d = m.cfg.decoding
    with open_dict(d):
        d.greedy.use_cuda_graph_decoder = cuda_graphs
        # batched greedy for every model (parakeet-tdt-1.1b ships the per-utterance "greedy"); a config choice, equal
        # for the stock and final arms; ASR_DECODING=greedy restores the as-shipped setting for that model
        if "strategy" in d and d.strategy in ("greedy", "greedy_batch"):
            d.strategy = os.environ.get("ASR_DECODING", "greedy_batch")
    m.change_decoding_strategy(d)
    return m


POOL_RULE = os.environ.get("POOL_RULE", "pool2")                 # pool2 (keep 1/2) or pool43 (keep 3/4)


def merges_for(pool_at: list[int], drop: list[int], rule: str | None = None, pools: dict | None = None):
    """{layer: merge fn}: one rule for every pool_at layer, or per-layer rules from pools {layer: rule name}."""
    if pools:
        return {int(k): getattr(em, r) for k, r in pools.items()}
    fn = getattr(em, rule or POOL_RULE)
    return {k: fn for k in pool_at}


def parse_pools(spec: str) -> dict:
    """'4:pool43,12:pool2' -> {4: 'pool43', 12: 'pool2'}"""
    return {int(k): r for k, r in (x.split(":") for x in spec.split(",") if x)}


def apply_drop(m, drop: list[int]):
    """Remove layers by turning their residual branches off (zero outputs); used for layer-removal variants."""
    hs = []
    for l in drop:
        layer = m.encoder.layers[l]
        for mod in (layer.feed_forward1, layer.conv, layer.self_attn, layer.feed_forward2):
            hs.append(mod.register_forward_hook(lambda mod, args, out: torch.zeros_like(out)))
    return hs


def batch_audio(ws: list[np.ndarray]):
    L = max(len(w) for w in ws)
    sig = torch.zeros(len(ws), L, device="cuda")
    for j, w in enumerate(ws):
        sig[j, :len(w)] = torch.from_numpy(w)
    return sig, torch.tensor([len(w) for w in ws], device="cuda")


def decode(m, merges, sig, sl, skip=()) -> list[str]:
    f, fl = m.preprocessor(input_signal=sig, length=sl)
    e, l = em.encode_multi(m, f, fl, merges, skip)
    hs = m.decoding.rnnt_decoder_predictions_tensor(encoder_output=e, encoded_lengths=l, return_hypotheses=True)
    hs = hs[0] if isinstance(hs, tuple) else hs
    return [h.text if isinstance(h.text, str) else "" for h in hs]


def rows_of(name: str) -> list[dict]:
    files = sorted(glob.glob(os.path.join(DATA, name, "*.parquet"))) or [os.path.join(DATA, f"{name}.parquet")]
    out = []
    for f in files:
        t = pq.read_table(f)
        cols = [c for c in ("audio", "text", "transcription") if c in t.column_names]
        out += t.select(cols).to_pylist()
    return out


def sparse24_masks(m, which: str, calib=None):
    """2:4 masks (2 kept of every 4 along the input dimension) for the encoder's linears in `which` (comma list of ff,
    att, pw): applied once now and after every optimiser step. Score |W| (magnitude), or with calib (a list of
    (sig, len) training batches) Wanda's |W| * ||x_in||_2 per input channel. -> [(weight, mask)]"""
    groups = set(which.split(","))
    out = []
    norms = {}
    if calib:
        hooks = []

        def hook(mod, inp, _out):
            x = inp[0].detach().float()
            x = x.transpose(1, 2) if isinstance(mod, torch.nn.Conv1d) else x      # -> [..., in]
            sq = x.reshape(-1, x.shape[-1]).pow(2).sum(0)
            norms[mod] = norms.get(mod, 0) + sq
        for layer in m.encoder.layers:
            for mod in (layer.feed_forward1.linear1, layer.feed_forward1.linear2, layer.feed_forward2.linear1,
                        layer.feed_forward2.linear2, layer.self_attn.linear_q, layer.self_attn.linear_k,
                        layer.self_attn.linear_v, layer.self_attn.linear_out, layer.conv.pointwise_conv1,
                        layer.conv.pointwise_conv2):
                hooks.append(mod.register_forward_hook(hook))
        with torch.no_grad():
            for sig, sl in calib:
                f, fl = m.preprocessor(input_signal=sig, length=sl)
                m.encoder(audio_signal=f, length=fl)
        for h in hooks:
            h.remove()
    for layer in m.encoder.layers:
        mods = []
        if "ff" in groups:
            mods += [layer.feed_forward1.linear1, layer.feed_forward1.linear2,
                     layer.feed_forward2.linear1, layer.feed_forward2.linear2]
        if "att" in groups:
            a_ = layer.self_attn
            mods += [a_.linear_q, a_.linear_k, a_.linear_v, a_.linear_out]
        if "pw" in groups:
            mods += [layer.conv.pointwise_conv1, layer.conv.pointwise_conv2]
        for mod in mods:
            w = mod.weight
            w2 = w.detach().reshape(w.shape[0], -1)          # [out, in] (pointwise convs are [out, in, 1])
            score = w2.abs() * (norms[mod].sqrt()[None, :] if mod in norms else 1.0)
            g = score.reshape(w2.shape[0], -1, 4)
            mk = torch.zeros_like(g, dtype=torch.bool).scatter_(2, g.topk(2, dim=2).indices, True)
            mk = mk.reshape(w.shape).to(w.dtype)
            with torch.no_grad():
                w.mul_(mk)
            out.append((w, mk))
    return out


def train(a) -> None:
    import nemo.collections.asr as nemo_asr
    torch.manual_seed(a.seed)
    pools = parse_pools(a.pools) if a.pools else {}
    pool_at = sorted(pools) if pools else [int(x) for x in a.pool_at.split(",") if x != ""]
    drop = [int(x) for x in a.drop.split(",") if x != ""]
    teacher = load_model(nemo_asr).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    m = load_model(nemo_asr)
    if a.init:                                              # continue from a fine-tuned checkpoint (same variant)
        ck = torch.load(a.init, map_location="cuda", weights_only=True)
        same = ck["pool_at"] == pool_at and ck["drop"] == drop and (pools or ck.get("rule", "pool2") == POOL_RULE)
        if not same:                                        # weights carry over (pooling has no parameters)
            print("init from a different pooling variant:", {k: ck[k] for k in ("pool_at", "drop", "rule")}, flush=True)
        sd = m.state_dict()
        sd.update(ck["trained"])
        m.load_state_dict(sd)
        print("init from", a.init, flush=True)
    first = min(pool_at + drop) if (pool_at or drop) else 0
    train_params = []
    for p in m.parameters():
        p.requires_grad = False
    for layer in (m.encoder.layers if a.train_all else m.encoder.layers[first + 1:]):
        for p in layer.parameters():
            p.requires_grad = True
            train_params.append(p)
    for mod in (m.decoder, m.joint):
        for p in mod.parameters():
            p.requires_grad = True
            train_params.append(p)
    calib = None
    if a.sparse24 and a.sparse_method == "wanda":            # calibration from training data (not dev/test)
        cr = pq.read_table(sorted(glob.glob(os.path.join(DATA, "train100", "*.parquet")))[0]).slice(0, 128)
        ws = [load_audio(r) for r in cr.to_pylist()]
        ws = sorted([w for w in ws if len(w) <= 16 * 16000], key=len)
        calib = [batch_audio(ws[b:b + 16]) for b in range(0, len(ws), 16)]
    masks = sparse24_masks(m, a.sparse24, calib) if a.sparse24 else []   # 2:4 structured sparsity (fixed)
    if masks:
        print({"sparse24": a.sparse24, "sparse_method": a.sparse_method, "masked_weights": len(masks)}, flush=True)
    opt = torch.optim.AdamW(train_params, lr=a.lr, weight_decay=0.0)
    t_sched = {"t0": time.time()}

    def lr_factor(s):
        f = min(1.0, (s + 1) / a.warmup)
        if a.decay:                                         # linear decay over the time budget, to 5%
            f *= max(0.05, 1.0 - (time.time() - t_sched["t0"]) / (a.hours * 3600))
        return f
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    drop_h = []                                             # dropped layers are skipped outright (encode_multi)
    merges = merges_for(pool_at, drop, pools=pools)
    # stream shard by shard (shards in shuffled order, rows shuffled within each shard): loading every shard at
    # once (train-clean-360 + AMI, ~35 GB) would squeeze GPU memory on the GB10's unified memory
    shards = []
    for d in a.data.split(","):
        fs = sorted(glob.glob(os.path.join(DATA, d, "*.parquet"))) or [os.path.join(DATA, f"{d}.parquet")]
        shards += fs
    n_rows = sum(pq.ParquetFile(f).metadata.num_rows for f in shards)

    def stream():
        rng = random.Random(a.seed)
        ep = 0
        while True:
            order = shards[:]
            rng.shuffle(order)
            for f in order:
                t = pq.read_table(f)
                cols = [c for c in ("audio", "text", "transcription") if c in t.column_names]
                rs = t.select(cols).to_pylist()
                del t
                rng.shuffle(rs)
                for r in rs:
                    yield r
            ep += 1
    src = stream()
    print({"shards": len(shards), "rows": n_rows}, flush=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)
    log = open(os.path.join(a.out, "train_log.jsonl"), "a")
    t_end = time.time() + a.hours * 3600
    total = int(a.epochs * n_rows)
    step, i = 0, 0

    norm = EnglishTextNormalizer()
    dev = rows_of("dev_clean")
    random.Random(20261008).shuffle(dev)
    dev = [(load_audio(r), norm(ref_text(r))) for r in dev[:96]]

    def quick_dev() -> float:
        m.eval()
        hyps, refs = [], []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for b in range(0, len(dev), 16):
                ch = dev[b:b + 16]
                sig, sl = batch_audio([w for w, _ in ch])
                hyps += [norm(h) or "<empty>" for h in decode(m, merges, sig, sl, drop)]
                refs += [r for _, r in ch]
        return 100 * jiwer.wer(refs, hyps)

    def save():
        torch.save({"pool_at": pool_at, "drop": drop, "rule": POOL_RULE, "pools": pools, "sparse24": a.sparse24, "sparse_method": a.sparse_method,
                    "trained": {n: v.detach().cpu() for n, v in m.state_dict().items()}},
                   os.path.join(a.out, "ckpt.pt"))
    print({"step": 0, "quick_dev_wer": quick_dev()}, flush=True)
    while time.time() < t_end and i < total:
        ws = []
        while len(ws) < a.batch and i < total:
            w = load_audio(next(src))
            if 0.5 <= len(w) / 16000 <= a.max_s:
                ws.append(w)
            i += 1
        sig, sl = batch_audio(ws)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            f, fl = teacher.preprocessor(input_signal=sig, length=sl)
            e0, l0 = teacher.encoder(audio_signal=f, length=fl)
            hyps = teacher.decoding.rnnt_decoder_predictions_tensor(encoder_output=e0, encoded_lengths=l0,
                                                                    return_hypotheses=True)
            hyps = hyps[0] if isinstance(hyps, tuple) else hyps
        toks = [list(h.y_sequence.tolist() if hasattr(h.y_sequence, "tolist") else h.y_sequence) or [0] for h in hyps]
        U = max(len(t) for t in toks)
        tgt = torch.zeros(len(ws), U, dtype=torch.long, device="cuda")
        tl = torch.tensor([len(t) for t in toks], device="cuda")
        for j, t in enumerate(toks):
            tgt[j, :len(t)] = torch.tensor(t)
        m.train()
        if not a.train_all:
            for layer in m.encoder.layers[:first + 1]:
                layer.eval()
        # batch norm stays in eval mode: train-mode batch statistics on padded, pooled batches do not match the
        # running statistics used at inference (S1: loss 0.09 in training, no WER recovery at evaluation)
        for mod in m.modules():
            if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
                mod.eval()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            e, l = em.encode_multi(m, f, fl, merges, drop)
            dec, dl, _ = m.decoder(targets=tgt, target_length=tl)
            jnt = m.joint.joint(e.transpose(1, 2), dec.transpose(1, 2))
        loss_tdt = m.loss(log_probs=jnt.float(), targets=tgt, input_lengths=l, target_lengths=dl).mean()
        loss = loss_tdt
        loss_kd = torch.zeros((), device="cuda")
        if a.kd_w > 0:
            # feature distillation: the teacher's encoder output, pooled to the student's frame rate, at every frame
            with torch.no_grad():
                te, tlen = e0.transpose(1, 2).float(), l0.to(torch.int64)
                for k in pool_at:
                    te, tlen = merges[k](te, tlen)
            se = e.transpose(1, 2).float()
            T = min(se.shape[1], te.shape[1])
            msk = (torch.arange(T, device="cuda")[None] < torch.minimum(l, tlen)[:, None]).float()[..., None]
            # relative MSE: normalised by the teacher's mean square, so kd_w is comparable across layers and scales
            num = (((se[:, :T] - te[:, :T]) ** 2) * msk).sum()
            den = ((te[:, :T] ** 2) * msk).sum().clamp(min=1e-8)
            loss_kd = num / den
            loss = loss_tdt + a.kd_w * loss_kd
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            for w, mk in masks:                             # keep pruned weights at zero
                w.mul_(mk)
        step += 1
        if step % 25 == 0:
            rec = {"step": step, "seen": i, "loss": float(loss), "loss_tdt": float(loss_tdt), "loss_kd": float(loss_kd), "frames_in": int(fl.sum() // 8),
                   "frames_out": int(l.sum()), "t": time.time()}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(rec, flush=True)
        if step % 500 == 0:
            q = quick_dev()
            log.write(json.dumps({"step": step, "quick_dev_wer": q}) + "\n")
            log.flush()
            print({"step": step, "quick_dev_wer": q}, flush=True)
        if step % 1000 == 0:
            save()
    save()
    for h in drop_h:
        h.remove()
    print("TRAIN DONE", step, "steps", i, "utterances", flush=True)


class FP8Linear(torch.nn.Module):
    """nn.Linear replacement running its GEMM in FP8 (Transformer Engine, delayed scaling, E4M3 forward). Rows are
    padded to a multiple of 16 as TE's FP8 GEMMs require."""
    def __init__(self, lin: torch.nn.Linear):
        super().__init__()
        import transformer_engine.pytorch as te
        from transformer_engine.common.recipe import DelayedScaling, Format
        self.te = te
        self.recipe = DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=16, amax_compute_algo="max")
        self.lin = te.Linear(lin.in_features, lin.out_features, bias=lin.bias is not None,
                             params_dtype=torch.bfloat16, device="cuda")
        with torch.no_grad():
            self.lin.weight.copy_(lin.weight.to(torch.bfloat16))
            if lin.bias is not None:
                self.lin.bias.copy_(lin.bias.to(torch.bfloat16))

    def forward(self, x):
        shp = x.shape
        x2 = x.reshape(-1, shp[-1]).to(torch.bfloat16)
        n = x2.shape[0]
        pad = (-n) % 16
        if pad:
            x2 = torch.cat([x2, x2.new_zeros(pad, shp[-1])])
        with self.te.fp8_autocast(enabled=True, fp8_recipe=self.recipe):
            y = self.lin(x2)
        return y[:n].reshape(*shp[:-1], y.shape[-1])


def to_fp8(m):
    n = 0
    for layer in m.encoder.layers:
        for parent in layer.modules():
            for name, child in list(parent.named_children()):
                if type(child) is torch.nn.Linear and child.in_features % 16 == 0 and child.out_features % 16 == 0:
                    setattr(parent, name, FP8Linear(child))
                    n += 1
    return n


def fast_paths(m, sdpa: bool, compile_mode: str, bf16: bool = False, fp8: bool = False):
    """Engineering speedups with the model's own numerics: fused attention (NeMo's use_pytorch_sdpa) and
    torch.compile of each conformer layer."""
    if bf16:                                                # weights held in bf16 once (no per-call autocast casts)
        m.encoder.to(torch.bfloat16)                       # the encoder is ~88% of the time; decoder stays fp32
    if fp8:
        print({"fp8_linears": to_fp8(m)}, flush=True)
    if sdpa:
        for layer in m.encoder.layers:
            layer.self_attn.use_pytorch_sdpa = True
            if not hasattr(layer.self_attn, "use_pytorch_sdpa_backends"):
                layer.self_attn.use_pytorch_sdpa_backends = []
    if compile_mode:
        for i, layer in enumerate(m.encoder.layers):
            m.encoder.layers[i] = torch.compile(layer, dynamic=True, mode=compile_mode)


def load_variant(nemo_asr, ckpt: str, cuda_graphs: bool = False):
    m = load_model(nemo_asr, cuda_graphs).eval()
    if ckpt == "base":
        return m, {}, []
    if ckpt.startswith("pool:"):                            # untrained variant, e.g. pool:8 or pool:4,14[/skip:0,2]
        spec = ckpt[5:].split("/skip:")
        pool_at = [int(x) for x in spec[0].split(",") if x]
        skip = [int(x) for x in spec[1].split(",")] if len(spec) > 1 else []
        return m, merges_for(pool_at, []), skip
    ck = torch.load(ckpt, map_location="cuda", weights_only=True)
    sd = m.state_dict()
    sd.update(ck["trained"])
    m.load_state_dict(sd)
    return m, merges_for(ck["pool_at"], ck["drop"], ck.get("rule", "pool2"), ck.get("pools")), ck["drop"]


def evaluate(a) -> None:
    import nemo.collections.asr as nemo_asr
    m, merges, skip = load_variant(nemo_asr, a.ckpt)
    fast_paths(m, a.sdpa, a.compile, a.bf16, a.fp8)
    norm = EnglishTextNormalizer()
    res = {"ckpt": a.ckpt, "sdpa": a.sdpa, "compile": a.compile, "fp8": a.fp8, "sets": {}}
    for s in a.sets.split(","):
        rows = rows_of(s)
        idx = list(range(len(rows)))
        random.Random(a.seed).shuffle(idx)
        sel = [rows[i] for i in idx[:a.n]]
        auds = [load_audio(r) for r in sel]
        order = sorted(range(len(sel)), key=lambda k: len(auds[k]))
        hyps = [None] * len(sel)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for b in range(0, len(order), 16):
                ids = order[b:b + 16]
                sig, sl = batch_audio([auds[k] for k in ids])
                for k, h in zip(ids, decode(m, merges, sig, sl, skip)):
                    hyps[k] = h
        refs = [norm(ref_text(r)) for r in sel]
        hy = [norm(h) or "<empty>" for h in hyps]
        o = jiwer.process_words(refs, hy)
        nw = sum(len(r.split()) for r in refs)
        res["sets"][s] = {"n": len(sel), "wer_pct": 100 * jiwer.wer(refs, hy), "sub_pct": 100 * o.substitutions / nw,
                          "del_pct": 100 * o.deletions / nw, "ins_pct": 100 * o.insertions / nw}
        res.setdefault("hyps", {})[s] = hy
        print(s, res["sets"][s], flush=True)
    json.dump(res, open(a.out_json, "w"), indent=1)


def rtf(a) -> None:
    import nemo.collections.asr as nemo_asr
    m, merges, skip = load_variant(nemo_asr, a.ckpt, cuda_graphs=True)
    fast_paths(m, a.sdpa, a.compile, a.bf16, a.fp8)
    rows = rows_of(a.set)
    auds = [load_audio(r) for r in rows]
    audio_s = sum(len(w) for w in auds) / 16000
    order = sorted(range(len(auds)), key=lambda k: len(auds[k]))
    batches = [order[b:b + a.batch] for b in range(0, len(order), a.batch)]
    sigs = [batch_audio([auds[k] for k in ids]) for ids in batches]

    def run():
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for sig, sl in sigs:
                decode(m, merges, sig, sl, skip)
        torch.cuda.synchronize()
        return time.time() - t0
    run()                                                       # warm-up
    t_timed0 = time.time()
    walls = [run() for _ in range(a.reps)]
    t_timed1 = time.time()
    # stage breakdown (one pass): preprocessor, encoder, greedy TDT decoding
    st = {"pre": 0.0, "enc": 0.0, "dec": 0.0}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for sig, sl in sigs:
            torch.cuda.synchronize(); t0 = time.time()
            f, fl = m.preprocessor(input_signal=sig, length=sl)
            torch.cuda.synchronize(); t1 = time.time()
            e, l = em.encode_multi(m, f, fl, merges, skip)
            torch.cuda.synchronize(); t2 = time.time()
            m.decoding.rnnt_decoder_predictions_tensor(encoder_output=e, encoded_lengths=l, return_hypotheses=True)
            torch.cuda.synchronize(); t3 = time.time()
            st["pre"] += t1 - t0; st["enc"] += t2 - t1; st["dec"] += t3 - t2
    w = sorted(walls)[len(walls) // 2]
    res = {"ckpt": a.ckpt, "compile": a.compile, "sdpa": a.sdpa, "bf16": a.bf16, "fp8": a.fp8, "set": a.set, "utterances": len(auds), "audio_s": audio_s, "batch": a.batch,
           "walls_s": walls, "rtf": audio_s / w, "stages_s": st, "timed_window": [t_timed0, t_timed1]}
    json.dump(res, open(a.out_json, "w"), indent=1)
    print(json.dumps(res), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("out")
    t.add_argument("--pool-at", default="")
    t.add_argument("--drop", default="")
    t.add_argument("--data", default="train100")
    t.add_argument("--epochs", type=float, default=1.0)
    t.add_argument("--hours", type=float, default=3.0)
    t.add_argument("--pools", default="", help="per-layer pooling, e.g. 4:pool43,12:pool2 (overrides --pool-at)")
    t.add_argument("--init", default="", help="start from this fine-tuned checkpoint (same pool/drop variant)")
    t.add_argument("--sparse24", default="", help="2:4 structured sparsity on these encoder linears: ff,att,pw")
    t.add_argument("--sparse-method", default="magnitude", choices=["magnitude", "wanda"])
    t.add_argument("--decay", action="store_true", help="linear learning-rate decay over the time budget")
    t.add_argument("--lr", type=float, default=2e-5)
    t.add_argument("--warmup", type=int, default=200)
    t.add_argument("--batch", type=int, default=12)
    t.add_argument("--max-s", type=float, default=16.0)
    t.add_argument("--seed", type=int, default=20261007)
    t.add_argument("--kd-w", type=float, default=0.0, help="weight of the encoder-output distillation loss")
    t.add_argument("--train-all", action="store_true", help="train every encoder layer, not only after the first pool")
    e = sub.add_parser("eval")
    e.add_argument("ckpt")
    e.add_argument("out_json")
    e.add_argument("--sets", default="dev_clean,earnings22")
    e.add_argument("--n", type=int, default=400)
    e.add_argument("--seed", type=int, default=20261005, help="20261008 with --n 96 = the training quick-dev set")
    e.add_argument("--sdpa", action="store_true")
    e.add_argument("--compile", default="")
    e.add_argument("--bf16", action="store_true")
    e.add_argument("--fp8", action="store_true")
    r = sub.add_parser("rtf")
    r.add_argument("ckpt")
    r.add_argument("out_json")
    r.add_argument("--set", default="dev_clean")
    r.add_argument("--batch", type=int, default=32)
    r.add_argument("--reps", type=int, default=3)
    r.add_argument("--compile", default="", help="torch.compile mode for each encoder layer, e.g. default or max-autotune-no-cudagraphs")
    r.add_argument("--sdpa", action="store_true", help="NeMo's fused attention path (use_pytorch_sdpa)")
    r.add_argument("--bf16", action="store_true", help="weights in bf16 (as the 1 Oct NeMo R4 bar)")
    r.add_argument("--fp8", action="store_true", help="encoder linears in FP8 (Transformer Engine)")
    a = ap.parse_args()
    {"train": train, "eval": evaluate, "rtf": rtf}[a.cmd](a)


if __name__ == "__main__":
    main()
