"""A lean, export-friendly re-implementation of parakeet-tdt-0.6b-v3's FastConformer encoder forward (same weights,
no training), written to cut memory traffic in the TensorRT engine:
  - no fp32 casts: attention runs in the model's working precision (NeMo forces fp32 attention);
  - the relative-position term is a single gather with an index matrix (T-1-i+j) built once per batch, instead of
    NeMo's pad + reshape + slice rel_shift in every layer;
  - the key padding bias and the frame mask are built once per batch and reused by every layer;
  - the conv module's pointwise convolutions are plain matmuls on [B, T, C] (no transposes around Conv1d);
  - inference batch norm is folded into the depthwise convolution;
  - dw_shift=True writes the depthwise convolution as K shifted multiply-adds on [B, T, C], so TensorRT can fuse it
    with the GLU, the frame mask and the activation (its grouped Conv kernel is about 5x its memory cost on GB10);
  - rel_shift=True uses NeMo's pad + reshape + slice for the relative-position term instead of the index gather
    (TensorRT expands the [T, T] index to [B, H, T, T] and gathers, about 20 ms per 32 x 16 s batch);
  - heads_first=True computes attention in [H, B, T, .] layout, so the relative-position term is one matmul per head
    against the shared position projection (TensorRT replicates that projection across the batch otherwise, about
    10 ms per 32 x 16 s batch);
  - fused_qkv=True (with heads_first) replaces linear_q/k/v by one 1024 -> 3072 linear (weights concatenated, so
    FP8 PTQ uses one weight scale for the three) and one permute to [3, H, B, T, dk];
  - qkv_plugin_export (set after FP8 calibration of the fused QKV linear, by lean_export.py with LEAN_QKV_PLUGIN=1)
    emits the TensorRT plugin node QKVHeads (plugins/qkv_heads.cu), whose GEMM epilogue writes q_u, k, v
    heads-first; q_v is folded as bd = q_u.p + (pos_bias_v - pos_bias_u).p; with attn_plugin also set, attention is
    the RelPosAttn plugin (relpos_attn.py) instead of TensorRT's position matmul, shift and attention;
  - pool43_after=L (an accuracy trade, not an equivalence): after layer L, every 4 frames become 3 (the last two
    averaged, encoder_merge.pool43), so later layers and the decoder see 25% fewer frames. Untrained.
LeanEncoder(m.encoder)(feats, lengths) -> (encoded [B, D, T'], lengths'), matching NeMo's encoder output.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as Fn


_QKV_CONST: dict = {}


class _QKVHeadsOp(torch.autograd.Function):
    """ONNX: one QKVHeads node (FP8 input -> q_u, k, v [H, B, T, dk] fp16). Torch: the same maths, for tracing."""

    @staticmethod
    def forward(ctx, hq, key):
        d = _QKV_CONST[key]
        B, T, _ = hq.shape
        y = (hq.float() @ d["wf"].t() + d["bias"]).view(B, T, 3, d["H"], d["dk"]).permute(2, 3, 0, 1, 4)
        return tuple(y[i].contiguous().to(torch.float16) for i in range(3))

    @staticmethod
    def symbolic(g, hq, key):
        # weights/bias/scale are attached by qkv_surgery.py from the side file lean_export.py writes (the exporter
        # cannot carry an int8 tensor attribute)
        return g.op("trt::QKVHeads", hq, layer_i=int(key), plugin_version_s="1", plugin_namespace_s="", outputs=3)


def qkv_side_data() -> dict:
    """Per layer: e4m3 weight bytes, bias (incl. pos_bias_u for q), alpha, dims, for qkv_surgery.py."""
    import numpy as np
    out = {}
    for k, d in _QKV_CONST.items():
        out[f"w{k}"] = d["w8"].numpy().astype(np.int8)
        out[f"b{k}"] = d["bias"].cpu().numpy().astype(np.float32)
        out[f"a{k}"] = np.array(d["alpha"], np.float32)
        out[f"d{k}"] = np.array([d["K"], d["H"], d["dk"]], np.int32)
    return out


_SUB_CONST: dict = {}


class _SubConv02Op(torch.autograd.Function):
    """ONNX: one SubConv02 node (masked, NHWC) + Transpose back to NCHW. Torch: NeMo's MaskedConvSequential maths
    for conv.0, ReLU, conv.2 (rows past each stage's length zeroed before the next layer)."""

    @staticmethod
    def forward(ctx, x, lengths):
        d = _SUB_CONST["w"]
        x = x.float()
        T = x.shape[2]
        L1 = (lengths + 1) // 2
        L2 = (L1 + 1) // 2
        tm = lambda n, L: (torch.arange(n, device=x.device)[None, :] < L[:, None])[:, None, :, None].to(x.dtype)
        h = Fn.conv2d(x * tm(T, lengths), d["w0"], d["b0"], stride=2, padding=1)
        h = Fn.relu(h * tm(h.shape[2], L1))
        h = Fn.conv2d(h, d["w2"], d["b2"], stride=2, padding=1, groups=h.shape[1])
        return (h * tm(h.shape[2], L2)).half()

    @staticmethod
    def symbolic(g, x, lengths):
        d = _SUB_CONST["w"]
        fl = lambda t: t.float().reshape(-1).tolist()
        y = g.op("trt::SubConv02", x, lengths, w0_f=fl(d["w0"]), b0_f=fl(d["b0"]), w2_f=fl(d["w2"]), b2_f=fl(d["b2"]), nhwc_i=1, masked_i=1, plugin_version_s="1", plugin_namespace_s="")
        return g.op("Transpose", y, perm_i=[0, 3, 1, 2])


class _RelPosAttnOp(torch.autograd.Function):
    """ONNX: one RelPosAttn node (fused relative-position attention, relpos_attn.py; cubin attached by
    qkv_surgery.py). Torch: the reference maths, for tracing."""

    @staticmethod
    def forward(ctx, q_u, k, v, p, c, lengths, dk):
        from relpos_attn import reference
        return reference(q_u, k, v, p, c, lengths).to(torch.float16)

    @staticmethod
    def symbolic(g, q_u, k, v, p, c, lengths, dk):
        return g.op("trt::RelPosAttn", q_u, k, v, p, c, lengths, dk_i=dk, plugin_version_s="1", plugin_namespace_s="")


class LeanEncoder(torch.nn.Module):
    def __init__(self, enc, premask_once: bool = False, dw_shift: bool = False, rel_shift: bool = False,
                 pool43_after: int = -1, heads_first: bool = False, fused_qkv: bool = False, pools: dict | None = None,
                 skip=(), sub_masked: bool = False):
        super().__init__()
        self.sub_masked = sub_masked     # fused subsampling (SubConv02) with NeMo's per-layer masking
        self.skip = set(skip)            # layers removed by distilled layer dropping (finetune_stride --drop)
        import encoder_merge as em
        # frame pooling after given layers: {layer: fn} (pool43_after=L is {L: pool43}); trained variants only
        self.pools = dict(pools or {})
        if pool43_after >= 0:
            self.pools[pool43_after] = em.pool43
        self.fused_qkv = fused_qkv
        self.heads_first = heads_first
        self.pool43_after = pool43_after
        self.rel_shift = rel_shift
        self.premask_once = premask_once
        self.dw_shift = dw_shift
        self.pre_encode = enc.pre_encode
        self.pos_enc = enc.pos_enc
        self.xscale = getattr(enc, "xscale", None)
        self.layers = enc.layers
        self.d = enc.d_model
        # fold batch norm into the depthwise conv (inference)
        self.dw_w, self.dw_b = [], []
        for layer in self.layers:
            c = layer.conv
            bn = c.batch_norm
            w = c.depthwise_conv.weight                                            # [C, 1, K]
            b = c.depthwise_conv.bias if c.depthwise_conv.bias is not None else torch.zeros(w.shape[0], device=w.device)
            s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            self.dw_w.append(torch.nn.Parameter((w * s[:, None, None]).detach(), requires_grad=False))
            self.dw_b.append(torch.nn.Parameter(((b - bn.running_mean) * s + bn.bias).detach(), requires_grad=False))
        self.dw_w = torch.nn.ParameterList(self.dw_w)
        self.dw_b = torch.nn.ParameterList(self.dw_b)
        # pointwise convs as nn.Linear modules (so FP8 quantisation tools see them)
        self.pw1, self.pw2 = torch.nn.ModuleList(), torch.nn.ModuleList()
        for layer in self.layers:
            c = layer.conv
            for src, dst in ((c.pointwise_conv1, self.pw1), (c.pointwise_conv2, self.pw2)):
                w = src.weight.squeeze(-1)
                lin = torch.nn.Linear(w.shape[1], w.shape[0], bias=src.bias is not None).to(w.device)
                with torch.no_grad():
                    lin.weight.copy_(w)
                    if src.bias is not None:
                        lin.bias.copy_(src.bias)
                dst.append(lin)
        self.qkv = torch.nn.ModuleList()
        if fused_qkv:
            for layer in self.layers:
                a = layer.self_attn
                ws = [a.linear_q.weight, a.linear_k.weight, a.linear_v.weight]
                bs = [a.linear_q.bias, a.linear_k.bias, a.linear_v.bias]
                lin = torch.nn.Linear(ws[0].shape[1], 3 * ws[0].shape[0], bias=bs[0] is not None).to(ws[0].device)
                with torch.no_grad():
                    lin.weight.copy_(torch.cat(ws))
                    if bs[0] is not None:
                        lin.bias.copy_(torch.cat(bs))
                self.qkv.append(lin)
        self.kernel = self.layers[0].conv.depthwise_conv.kernel_size[0]
        self.pad = (self.kernel - 1) // 2

    def _attn(self, att, h, pos_emb, key_bias, rel_idx, B, T, D):
        q = att.linear_q(h).view(B, T, att.h, att.d_k)
        k = att.linear_k(h).view(B, T, att.h, att.d_k).transpose(1, 2)
        v = att.linear_v(h).view(B, T, att.h, att.d_k).transpose(1, 2)
        p = att.linear_pos(pos_emb).view(1, -1, att.h, att.d_k).transpose(1, 2)    # [1, H, 2T-1, dk]
        q_u = (q + att.pos_bias_u).transpose(1, 2)
        q_v = (q + att.pos_bias_v).transpose(1, 2)
        bd = torch.matmul(q_v, p.transpose(-2, -1))                                # [B, H, T, 2T-1]
        if self.rel_shift:                                                         # NeMo rel_shift, then crop
            bd = Fn.pad(bd, (1, 0)).view(B, att.h, -1, T)[:, :, 1:].reshape(B, att.h, T, 2 * T - 1)[..., :T]
        else:
            bd = torch.gather(bd, 3, rel_idx[None, None].expand(B, att.h, T, T))   # [B, H, T, T]
        bias = bd / math.sqrt(att.d_k) + key_bias
        a = Fn.scaled_dot_product_attention(q_u, k, v, attn_mask=bias)
        return att.linear_out(a.transpose(1, 2).reshape(B, T, D))

    def _attn_hb(self, att, h, pos_emb, key_bias, rel_idx, B, T, D, li):
        """Heads-first: q, k, v as [H, B, T, dk]; the position term is [H, B*T, dk] @ [H, dk, 2T-1]."""
        H, dk = att.h, att.d_k
        if getattr(self, "qkv_plugin_export", False):
            q_u, k, v = self._qkv_plugin(li, att, h)
            if getattr(self, "attn_plugin", False) and getattr(self, "pos_cache", None):
                # input-independent position term, precomputed for the longest T and sliced (prepare_pos_cache)
                o = self.pos_L - T
                ph = self.pos_cache[li][0][:, o:o + 2 * T - 1]                      # [H, 2T-1, dk] fp16
                c = self.pos_cache[li][1][:, o:o + 2 * T - 1]                       # [H, 2T-1] fp32
                a = _RelPosAttnOp.apply(q_u, k, v, ph.contiguous(), c.contiguous(), self._cur_len.to(torch.int32), dk)
                return att.linear_out(a.to(h.dtype))
            p = att.linear_pos(pos_emb).view(-1, H, dk).permute(1, 2, 0)          # [H, dk, 2T-1]
            c = torch.einsum("hd,hdr->hr", (att.pos_bias_v - att.pos_bias_u).to(p.dtype), p)
            if getattr(self, "attn_plugin", False):                               # fused attention kernel
                ph = p.permute(0, 2, 1).contiguous().half()                         # [H, 2T-1, dk]
                a = _RelPosAttnOp.apply(q_u, k, v, ph, c.float().contiguous(), self._cur_len.to(torch.int32), dk)
                return att.linear_out(a.to(h.dtype))                                # [B, T, D]
            # attention stays in the plugin's fp16 (casting q/k/v back to the graph's fp32 adds copies)
            bd = (torch.matmul(q_u.reshape(H, B * T, dk), p.half()) + c.half()[:, None, :]).view(H, B, T, 2 * T - 1)
            a = self._attn_core(att, q_u, k, v, bd, key_bias.half(), rel_idx, B, T)
            return att.linear_out(a.to(h.dtype).permute(1, 2, 0, 3).reshape(B, T, D))
        if self.fused_qkv:
            qkv = self.qkv[li](h).view(B, T, 3, H, dk).permute(2, 3, 0, 1, 4)    # [3, H, B, T, dk]
            q, k, v = qkv[0], qkv[1], qkv[2]
            q_u = q + att.pos_bias_u[:, None, None, :]
            q_v = (q + att.pos_bias_v[:, None, None, :]).reshape(H, B * T, dk)
        else:
            q = att.linear_q(h).view(B, T, H, dk)
            k = att.linear_k(h).view(B, T, H, dk).permute(2, 0, 1, 3)
            v = att.linear_v(h).view(B, T, H, dk).permute(2, 0, 1, 3)
            q_u = (q + att.pos_bias_u).permute(2, 0, 1, 3)                         # [H, B, T, dk]
            q_v = (q + att.pos_bias_v).permute(2, 0, 1, 3).reshape(H, B * T, dk)
        if getattr(self, "attn_plugin", False):    # RelPosAttn on q/k/v from TensorRT's GEMM (no QKVHeads plugin)
            if getattr(self, "pos_cache", None):
                o = self.pos_L - T
                ph = self.pos_cache[li][0][:, o:o + 2 * T - 1]
                c = self.pos_cache[li][1][:, o:o + 2 * T - 1]
            else:
                pp = att.linear_pos(pos_emb).view(-1, H, dk).permute(1, 2, 0)
                c = torch.einsum("hd,hdr->hr", (att.pos_bias_v - att.pos_bias_u).to(pp.dtype), pp).float()
                ph = pp.permute(0, 2, 1).half()
            a = _RelPosAttnOp.apply(q_u.half().contiguous(), k.half().contiguous(), v.half().contiguous(), ph.contiguous(),
                                    c.contiguous(), self._cur_len.to(torch.int32), dk)
            return att.linear_out(a.to(h.dtype))
        p = att.linear_pos(pos_emb).view(-1, H, dk).permute(1, 2, 0)              # [H, dk, 2T-1]
        bd = torch.matmul(q_v, p).view(H, B, T, 2 * T - 1)
        return self._attn_tail(att, q_u, k, v, bd, key_bias, rel_idx, B, T, D)

    def _attn_tail(self, att, q_u, k, v, bd, key_bias, rel_idx, B, T, D):
        a = self._attn_core(att, q_u, k, v, bd, key_bias, rel_idx, B, T)
        return att.linear_out(a.permute(1, 2, 0, 3).reshape(B, T, D))

    def _attn_core(self, att, q_u, k, v, bd, key_bias, rel_idx, B, T):
        """Relative shift of the position scores, key mask, SDPA: [H, B, T, dk] -> [H, B, T, dk]."""
        H, dk = att.h, att.d_k
        if self.rel_shift:
            bd = Fn.pad(bd, (1, 0)).view(H, B, -1, T)[:, :, 1:].reshape(H, B, T, 2 * T - 1)[..., :T]
        else:
            bd = torch.gather(bd, 3, rel_idx[None, None].expand(H, B, T, T))
        bias = bd / math.sqrt(dk) + key_bias.view(1, B, 1, T)
        return Fn.scaled_dot_product_attention(q_u, k, v, attn_mask=bias)         # [H, B, T, dk]

    def prepare_qkv_plugin(self) -> None:
        """After FP8 calibration and before export (not under tracing: the tracer cannot map FP8 dtypes): quantise each
        fused QKV weight exactly as ModelOpt's per-tensor FP8 scheme and fold pos_bias_u into the q bias."""
        for li, layer in enumerate(self.layers):
            if li in self.skip:                          # dropped layer: never run, never calibrated
                continue
            att, lin = layer.self_attn, self.qkv[li]
            with torch.no_grad():
                s_x = float(lin.input_quantizer.amax) / 448.0
                w = lin.weight.detach().float()
                H, dk = att.h, att.d_k
                n = H * dk                                           # q, k, v each get their own per-tensor scale
                s_ws = [float(w[i * n:(i + 1) * n].abs().max()) / 448.0 for i in range(3)]
                w8 = torch.cat([(w[i * n:(i + 1) * n] / s_ws[i]).clamp(-448, 448) for i in range(3)]).to(
                    torch.float8_e4m3fn)
                wf = torch.cat([w8[i * n:(i + 1) * n].float() * s_ws[i] for i in range(3)])
                b = lin.bias.detach().float().clone() if lin.bias is not None else torch.zeros(w.shape[0], device=w.device)
                b[:H * dk] += att.pos_bias_u.detach().float().reshape(-1)
                _QKV_CONST[li] = {"w8": w8.view(torch.int8).cpu(), "wf": wf, "bias": b,
                                  "alpha": [s_x * sw for sw in s_ws], "K": w.shape[1], "H": H, "dk": dk}
        self.qkv_plugin_export = True

    def _qkv_plugin(self, li, att, h):
        """Fake-quantise h with the calibrated input quantizer (exports Q/DQ; qkv_surgery.py then feeds the plugin
        the Q output) and call the QKVHeads op."""
        hq = self.qkv[li].input_quantizer(h)
        return _QKVHeadsOp.apply(hq, li)

    def prepare_pos_cache(self, max_frames: int = 750):
        """Precompute every layer's projected relative-position table for T up to max_frames (encoder frames; 750 =
        the engine's 6000 input frames / 8) in full precision, so the export slices it instead of running linear_pos
        and the (pos_bias_v - pos_bias_u) einsum per batch. NeMo's pe holds positions L-1 .. -(L-1) with position 0
        at index pe.size(1) // 2; rows for T are positions T-1 .. -(T-1). Only without pooling (T fixed per layer)."""
        assert not self.pools, "pos cache needs an unpooled encoder"
        pe = self.pos_enc.pe                                                     # [1, 2M-1, D]
        z = pe.size(1) // 2                                                      # index of position 0
        assert max_frames <= z + 1, (max_frames, pe.size(1))
        sub = pe[0, z - (max_frames - 1): z + max_frames].float()               # positions max-1 .. -(max-1)
        self.pos_L, self.pos_cache = max_frames, {}
        with torch.no_grad():
            for li, layer in enumerate(self.layers):
                if li in self.skip:
                    continue
                att = layer.self_attn
                w = att.linear_pos.weight.float()
                b = att.linear_pos.bias.float() if att.linear_pos.bias is not None else None
                p = Fn.linear(sub, w, b).view(-1, att.h, att.d_k).permute(1, 0, 2)  # [H, 2L-1, dk]
                c = torch.einsum("hd,hrd->hr", (att.pos_bias_v - att.pos_bias_u).float(), p)
                self.pos_cache[li] = (p.half().contiguous(), c.contiguous())

    def subsample_masked(self, x, lengths):
        """ConvSubsampling exactly as NeMo masks it (MaskedConvSequential), with conv.0, ReLU and conv.2 as one
        SubConv02 op (the plugin applies the same masks in-kernel). The remaining layers re-mask after every conv,
        since their biases make padded rows non-zero."""
        from nemo.collections.asr.parts.submodules.subsampling import calc_length
        pe = self.pre_encode
        out_lengths = calc_length(lengths, all_paddings=pe._left_padding + pe._right_padding,
                                  kernel_size=pe._kernel_size, stride=pe._stride, ceil_mode=pe._ceil_mode,
                                  repeat_num=pe._sampling_num)
        conv = list(pe.conv)
        assert isinstance(conv[1], torch.nn.ReLU) and conv[0].stride == (2, 2) and conv[2].stride == (2, 2)
        if "w" not in _SUB_CONST:                    # plain tensors (module parameters inside a traced Function fail)
            _SUB_CONST["w"] = {k: t.detach().float().clone() for k, t in
                               (("w0", conv[0].weight), ("b0", conv[0].bias), ("w2", conv[2].weight), ("b2", conv[2].bias))}
        L = lengths.to(torch.int32)
        h = _SubConv02Op.apply(x.unsqueeze(1).half(), L).to(x.dtype)
        cur = ((L + 1) // 2 + 1) // 2
        for layer in conv[3:]:
            h = layer(h)
            if isinstance(layer, torch.nn.Conv2d):
                if layer.stride != (1, 1):
                    assert layer.stride == (2, 2) and layer.kernel_size == (3, 3) and layer.padding == (1, 1)
                    cur = (cur + 1) // 2
                h = h * (torch.arange(h.shape[2], device=h.device)[None, :] < cur[:, None])[:, None, :, None].to(h.dtype)
        b, c, t, f = h.size()
        h = pe.out(h.transpose(1, 2).reshape(b, t, -1))
        return h, torch.minimum(out_lengths.to(lengths.dtype), torch.full_like(lengths, t))

    def subsample(self, x, lengths):
        """ConvSubsampling with the input masked once instead of re-masking before every conv layer (NeMo's
        MaskedConvSequential rebuilds and applies a time mask at each of its layers)."""
        from nemo.collections.asr.parts.submodules.subsampling import calc_length
        pe = self.pre_encode
        out_lengths = calc_length(lengths, all_paddings=pe._left_padding + pe._right_padding,
                                  kernel_size=pe._kernel_size, stride=pe._stride, ceil_mode=pe._ceil_mode,
                                  repeat_num=pe._sampling_num)
        T = x.shape[1]
        x = x * (torch.arange(T, device=x.device)[None, :] < lengths[:, None])[..., None].to(x.dtype)
        x = x.unsqueeze(1)
        for layer in pe.conv:
            x = layer(x)
        b, c, t, f = x.size()
        x = pe.out(x.transpose(1, 2).reshape(b, t, -1))
        return x, torch.minimum(out_lengths.to(lengths.dtype), torch.full_like(lengths, t))   # never past the output

    def forward(self, feats: torch.Tensor, lengths: torch.Tensor):
        x = feats.transpose(1, 2)
        if self.sub_masked:
            x, length = self.subsample_masked(x, lengths)
        elif self.premask_once:
            x, length = self.subsample(x, lengths)
        else:
            x, length = self.pre_encode(x=x, lengths=lengths)
        x, pos_emb = self.pos_enc(x=x, cache_len=0)                               # pos_emb [1, 2T-1, D]

        def masks(x, length):
            B, T, _ = x.shape
            valid = torch.arange(T, device=x.device)[None, :] < length[:, None]   # [B, T]
            key_bias = torch.zeros(B, 1, 1, T, device=x.device, dtype=x.dtype).masked_fill(
                ~valid[:, None, None, :], float("-inf"))
            ar = torch.arange(T, device=x.device)
            return valid[..., None].to(x.dtype), key_bias, (T - 1) - ar[:, None] + ar[None, :]   # rel index [T, T]
        B, T, D = x.shape
        vmask, key_bias, rel_idx = masks(x, length)
        self._cur_len = length
        for li, layer in enumerate(self.layers):
            if li - 1 in self.pools:
                x, length = self.pools[li - 1](x, length)
                T = x.shape[1]
                _, pos_emb = self.pos_enc(x=x.new_zeros(1, T, D), cache_len=0)
                vmask, key_bias, rel_idx = masks(x, length)
                self._cur_len = length
            if li in self.skip:                                                   # dropped layer: identity
                continue
            # FF1 (half step)
            res = x
            h = layer.feed_forward1(layer.norm_feed_forward1(x))
            x = res + 0.5 * h
            # relative-position self-attention
            att = layer.self_attn
            h = layer.norm_self_att(x)
            if self.heads_first:
                x = x + self._attn_hb(att, h, pos_emb, key_bias, rel_idx, B, T, D, li)
            else:
                x = x + self._attn(att, h, pos_emb, key_bias, rel_idx, B, T, D)
            # conv module: pointwise as matmul, depthwise with folded batch norm
            c = layer.conv
            h = layer.norm_conv(x)
            h = self.pw1[li](h)                                                     # [B, T, 2D]
            h = Fn.glu(h, dim=-1) * vmask
            if self.dw_shift:
                hp = Fn.pad(h, (0, 0, self.pad, self.pad))                          # [B, T + K - 1, D]
                w = self.dw_w[li][:, 0, :]                                          # [D, K]
                acc = self.dw_b[li] + hp[:, 0:T] * w[:, 0]
                for k in range(1, self.kernel):
                    acc = acc + hp[:, k:k + T] * w[:, k]
                h = c.activation(acc)
            else:
                h = Fn.conv1d(h.transpose(1, 2), self.dw_w[li], self.dw_b[li], padding=self.pad, groups=D)
                h = c.activation(h).transpose(1, 2)
            h = self.pw2[li](h)
            x = x + h
            # FF2 (half step) and output norm
            res = x
            h = layer.feed_forward2(layer.norm_feed_forward2(x))
            x = layer.norm_out(res + 0.5 * h)
        return x.transpose(1, 2), length
