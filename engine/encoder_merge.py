"""Run the FastConformer encoder of a NeMo ASR model layer by layer, with the option of shortening the frame sequence
after a chosen layer (frame merging or dropping). Every layer after the merge point then costs proportionally less.

encode(m, feats, lengths, merge_at=None, merge=None) -> (encoded [B, D, T'], lengths')
  replicates ConformerEncoder.forward_internal for offline use (no streaming caches), so merge_at=None gives the
  model's own encoder output (checked by check_equivalence()). After layer merge_at the sequence is passed through
  merge(x [B, T, D], lengths) -> (x', lengths'), and the relative position encoding and masks are rebuilt for the
  new length, as NeMo does for its own reduction_subsampling.

Merge rules (all keep at least one frame per utterance; ratio = fraction of frames kept, roughly):
  pair_mean(ratio)   average adjacent pairs where needed, evenly spaced (uniform merging)
  sim_merge(ratio)   repeatedly merge the most similar adjacent pair (cosine), weighted by merged size (ToMe-like)
  drop_uniform(ratio)  keep evenly spaced frames
  drop_random(ratio, seed)  keep a random subset, in order
  drop_lownorm(ratio)  keep the highest-norm frames (silence proxy)
merge_at=-1 applies the rule before the first layer (after the 8x subsampling).
"""
from __future__ import annotations

import torch
import torch.nn.functional as Fn


def encode(m, feats: torch.Tensor, lengths: torch.Tensor, merge_at: int | None = None, merge=None):
    enc = m.encoder
    x = torch.transpose(feats, 1, 2)
    x, length = enc.pre_encode(x=x, lengths=lengths)
    length = length.to(torch.int64)
    x, pos_emb = enc.pos_enc(x=x, cache_len=0)
    ctx = enc.att_context_size

    def masks(x, length):
        return enc._create_masks(att_context_size=ctx, padding_length=length, max_audio_length=x.size(1),
                                 offset=None, device=x.device)
    if merge_at == -1:                                # before the first layer (right after the 8x subsampling)
        x, length = merge(x, length)
        _, pos_emb = enc.pos_enc(x=x, cache_len=0)
    pad_mask, att_mask = masks(x, length)
    for lth, layer in enumerate(enc.layers):
        x = layer(x=x, att_mask=att_mask, pos_emb=pos_emb, pad_mask=pad_mask)
        if merge_at is not None and lth == merge_at:
            x, length = merge(x, length)
            _, pos_emb = enc.pos_enc(x=x, cache_len=0)
            pad_mask, att_mask = masks(x, length)
    if enc.out_proj is not None:
        x = enc.out_proj(x)
    return torch.transpose(x, 1, 2), length


def _pack(seqs: list[torch.Tensor], like: torch.Tensor):
    T = max(s.shape[0] for s in seqs)
    out = like.new_zeros(len(seqs), T, like.shape[-1])
    for i, s in enumerate(seqs):
        out[i, :s.shape[0]] = s
    return out, torch.tensor([s.shape[0] for s in seqs], device=like.device, dtype=torch.int64)


def pair_mean(ratio: float):
    def f(x, length):
        seqs = []
        for i in range(x.shape[0]):
            s = x[i, :length[i]]
            n = s.shape[0]
            target = max(1, round(n * ratio))
            merges = n - target                       # number of pairs to merge, evenly spaced
            out, j, done = [], 0, 0
            step = n / max(merges, 1)
            next_at = 0.0
            while j < n:
                if done < merges and j + 1 < n and j >= next_at:
                    out.append((s[j] + s[j + 1]) / 2)
                    j += 2
                    done += 1
                    next_at += step
                else:
                    out.append(s[j])
                    j += 1
            seqs.append(torch.stack(out))
        return _pack(seqs, x)
    return f


def sim_merge(ratio: float):
    def f(x, length):
        seqs = []
        for i in range(x.shape[0]):
            s = x[i, :length[i]].float()
            w = torch.ones(s.shape[0], device=s.device)
            target = max(1, round(s.shape[0] * ratio))
            while s.shape[0] > target:
                # merge in rounds: up to half the remaining excess per round, best non-overlapping pairs
                sim = Fn.cosine_similarity(s[:-1], s[1:], dim=-1)
                k = max(1, min((s.shape[0] - target), s.shape[0] // 2))
                order = torch.argsort(sim, descending=True).tolist()
                used, pairs = set(), []
                for p in order:
                    if p in used or p + 1 in used:
                        continue
                    pairs.append(p)
                    used.update((p, p + 1))
                    if len(pairs) >= k:
                        break
                pairs = set(pairs)
                ns, nw, j = [], [], 0
                while j < s.shape[0]:
                    if j in pairs:
                        tw = w[j] + w[j + 1]
                        ns.append((s[j] * w[j] + s[j + 1] * w[j + 1]) / tw)
                        nw.append(tw)
                        j += 2
                    else:
                        ns.append(s[j])
                        nw.append(w[j])
                        j += 1
                s, w = torch.stack(ns), torch.stack(nw)
            seqs.append(s.to(x.dtype))
        return _pack(seqs, x)
    return f


def drop_uniform(ratio: float):
    def f(x, length):
        seqs = []
        for i in range(x.shape[0]):
            n = int(length[i])
            k = max(1, round(n * ratio))
            idx = torch.linspace(0, n - 1, k, device=x.device).round().long().unique()
            seqs.append(x[i, idx])
        return _pack(seqs, x)
    return f


def drop_random(ratio: float, seed: int = 0):
    g = torch.Generator().manual_seed(seed)

    def f(x, length):
        seqs = []
        for i in range(x.shape[0]):
            n = int(length[i])
            k = max(1, round(n * ratio))
            idx = torch.sort(torch.randperm(n, generator=g)[:k])[0].to(x.device)
            seqs.append(x[i, idx])
        return _pack(seqs, x)
    return f


def drop_lownorm(ratio: float):
    """Keep the frames with the largest activation norm (a silence / low-energy proxy), in order."""
    def f(x, length):
        seqs = []
        for i in range(x.shape[0]):
            n = int(length[i])
            k = max(1, round(n * ratio))
            idx = torch.sort(torch.topk(x[i, :n].float().norm(dim=-1), k).indices)[0]
            seqs.append(x[i, idx])
        return _pack(seqs, x)
    return f


def check_equivalence(m, feats, lengths) -> float:
    """Max abs difference between encode() without merging and the model's own encoder."""
    with torch.inference_mode():
        a, la = m.encoder(audio_signal=feats, length=lengths)
        b, lb = encode(m, feats, lengths)
    assert torch.equal(la.to(torch.int64), lb), (la, lb)
    return float((a.float() - b.float()).abs().max())


def pool2(x: torch.Tensor, length: torch.Tensor):
    """Batched stride-2 average pooling over frames (fully on the GPU, no per-utterance loop). Odd-length sequences
    pair their last frame with itself; padded positions stay zero. Returns (x' [B, ceil(T/2), D], ceil(length/2))."""
    B, T, D = x.shape
    if T % 2:
        x = torch.cat([x, x.new_zeros(B, 1, D)], dim=1)
        T += 1
    idx = torch.arange(T, device=x.device)[None, :, None]
    last = (length - 1).clamp(min=0)[:, None, None]
    odd = (length % 2 == 1)[:, None, None]
    # for odd lengths, the slot after the last frame takes the last frame's value
    fill = torch.gather(x, 1, last.expand(B, 1, D))
    x = torch.where(odd & (idx == last + 1), fill.expand(B, T, D), x)
    y = x.view(B, T // 2, 2, D).mean(2)
    return y, (length + 1) // 2


def encode_multi(m, feats: torch.Tensor, lengths: torch.Tensor, merges: dict | None = None, skip=()):
    """Like encode(), with a merge function applied after each layer in `merges` ({layer: merge}); -1 = before
    the first layer. Layers in `skip` are not run at all (real compute saving; their merge, if any, still applies)."""
    merges = merges or {}
    skip = set(skip)
    enc = m.encoder
    x = torch.transpose(feats, 1, 2)
    x, length = enc.pre_encode(x=x, lengths=lengths)
    length = length.to(torch.int64)
    x, pos_emb = enc.pos_enc(x=x, cache_len=0)
    ctx = enc.att_context_size

    def masks(x, length):
        return enc._create_masks(att_context_size=ctx, padding_length=length, max_audio_length=x.size(1),
                                 offset=None, device=x.device)
    if -1 in merges:
        x, length = merges[-1](x, length)
        _, pos_emb = enc.pos_enc(x=x, cache_len=0)
    pad_mask, att_mask = masks(x, length)
    for lth, layer in enumerate(enc.layers):
        if lth not in skip:
            x = layer(x=x, att_mask=att_mask, pos_emb=pos_emb, pad_mask=pad_mask)
        if lth in merges:
            x, length = merges[lth](x, length)
            _, pos_emb = enc.pos_enc(x=x, cache_len=0)
            pad_mask, att_mask = masks(x, length)
    if enc.out_proj is not None:
        x = enc.out_proj(x)
    return torch.transpose(x, 1, 2), length


def pool43(x: torch.Tensor, length: torch.Tensor):
    """Batched 4 -> 3 frame reduction: in every group of 4 frames the last two are averaged (f1, f2, (f3+f4)/2).
    Sequences are padded to a multiple of 4 by repeating their last valid frame. Returns (x', new lengths)."""
    B, T, D = x.shape
    T4 = T + (-T) % 4
    if T4 > T:
        x = torch.cat([x, x.new_zeros(B, T4 - T, D)], dim=1)
    pos = torch.arange(T4, device=x.device)[None, :]
    last = (length - 1).clamp(min=0)[:, None]
    src = torch.where(pos < length[:, None], pos, last)                 # pad positions copy the last valid frame
    x = torch.gather(x, 1, src[..., None].expand(B, T4, D))
    g = x.view(B, T4 // 4, 4, D)
    y = torch.stack([g[:, :, 0], g[:, :, 1], g[:, :, 2:4].mean(2)], dim=2).reshape(B, (T4 // 4) * 3, D)
    full, r = length // 4, length % 4
    return y, full * 3 + torch.clamp(r, max=3)
