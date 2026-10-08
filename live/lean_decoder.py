# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Nigel Cannings. Part of fastconformer-trt (see LICENSE and NOTICE).
# Uses NVIDIA NeMo's (Apache-2.0) label-looping greedy RNN-T computer and detokeniser unchanged.
"""Lean live RNN-T decoder (custom arm): NeMo's own label-looping greedy computer (CUDA graphs, same kernels and the
same arithmetic as stock), without stock's per-hypothesis Python.

Hypothesis tested: in a live server the stock decoder is bound by host-side Python, not GPU kernels (profile, 5 Oct:
190 to 1,050 ms of host time per step against 12 to 19 ms of kernels at N = 64 to 256), because every step it builds
a Hypothesis object per stream (batched_hyps_to_hypotheses), merges partial hypotheses, splits and re-stacks the
decoder state per stream, and re-detokenises every stream's full transcript. Here the per-stream decoder state lives in
a slot-indexed slab on the GPU (gathered for the call and scattered back, like the encoder caches), the tokens of each
call are appended to a GPU token slab, and text is produced only when a stream ends (or on request). Nothing is
copied to the host per step. Ablate: --decoder stock.

Exactness: the computer is called exactly as stock calls it (encoder output transposed to [B, T, D], lengths, the
previous batched state or None on a stream's first chunk); the state passed in holds the same values stock's
merge_to_batched_state would build, and the final text uses stock's own detokeniser
(decode_tokens_to_str_with_strip_punctuation, including language-tag stripping). Checked by equiv.py.
"""
from __future__ import annotations

import torch


class LeanDecoder:
    def __init__(self, model, n_slots: int, max_tokens: int):
        self.m = model
        gi = model.decoding.decoding                              # GreedyBatchedRNNTInfer
        self.comp = getattr(gi, "decoding_computer", None)
        if self.comp is None:
            raise RuntimeError("lean decoder needs NeMo's label-looping greedy computer (greedy_batch, loop_labels)")
        if getattr(self.comp, "per_stream_biasing_enabled", False):
            raise NotImplementedError("per-stream biasing is not supported by the lean decoder")
        self.blank = model.decoding.blank_id
        self.n_slots, self.cap = n_slots, max_tokens
        self.state = None                                         # slab, allocated from the first returned state
        with torch.inference_mode():
            self.tok = torch.zeros(n_slots, max_tokens, dtype=torch.int32, device="cuda")
            self.tok_len = torch.zeros(n_slots, dtype=torch.int64, device="cuda")

    # state slab: predictor_states (h, c) are [L, B, H] (batch dim 1); the other fields have batch dim 0
    def _alloc(self, st) -> None:
        h, c = st.predictor_states
        n = self.n_slots
        self.state = {
            "h": h.new_zeros(h.shape[0], n, *h.shape[2:]), "c": c.new_zeros(c.shape[0], n, *c.shape[2:]),
            "out": st.predictor_outputs.new_zeros(n, *st.predictor_outputs.shape[1:]),
            "lab": st.labels.new_zeros(n, *st.labels.shape[1:]),
            "dlen": st.decoded_lengths.new_zeros(n, *st.decoded_lengths.shape[1:]),
        }
        if st.fusion_states_list or st.time_jumps is not None:
            raise NotImplementedError("fusion states / time jumps are not used by this model and not supported")

    def _gather(self, slots: torch.Tensor):
        from nemo.collections.asr.parts.submodules.transducer_decoding.label_looping_base import \
            BatchedLabelLoopingState
        s = self.state
        return BatchedLabelLoopingState(
            predictor_states=(s["h"].index_select(1, slots), s["c"].index_select(1, slots)),
            predictor_outputs=s["out"].index_select(0, slots), labels=s["lab"].index_select(0, slots),
            decoded_lengths=s["dlen"].index_select(0, slots), fusion_states_list=[], time_jumps=None)

    def _scatter(self, slots: torch.Tensor, st) -> None:
        """the CUDA-graph computer returns its static buffers, sized for the largest batch it has seen: the first B
        rows are this call's streams."""
        B = slots.numel()
        h, c = st.predictor_states
        s = self.state
        s["h"].index_copy_(1, slots, h[:, :B])
        s["c"].index_copy_(1, slots, c[:, :B])
        s["out"].index_copy_(0, slots, st.predictor_outputs[:B])
        s["lab"].index_copy_(0, slots, st.labels[:B])
        s["dlen"].index_copy_(0, slots, st.decoded_lengths[:B])

    def reset(self, slot: int) -> None:
        with torch.inference_mode():
            self.tok_len[slot] = 0

    def step(self, enc: torch.Tensor, enc_len: torch.Tensor, slots: torch.Tensor, first: bool) -> None:
        """enc [B, D, T] (after the language prompt), enc_len [B]; first = every stream in the call is on its first
        chunk (stock passes no previous state then)."""
        with torch.inference_mode():
            prev = None if first else self._gather(slots)
            hyps, st = self.comp(x=enc.transpose(1, 2), out_len=enc_len, prev_batched_state=prev,
                                 multi_biasing_ids=None)
            if self.state is None:
                self._alloc(st)
            self._scatter(slots, st)
            # append this call's tokens (hyps.transcript[:, :current_lengths]) to the token slab, all on the GPU
            B = slots.numel()
            tr = hyps.transcript[:B]
            L = tr.shape[1]
            pos = torch.arange(L, device="cuda")[None, :]
            n_new = hyps.current_lengths[:B]
            base = self.tok_len.index_select(0, slots)
            mask = pos < n_new[:, None]
            col = (base[:, None] + pos).clamp(max=self.cap - 1)
            row = slots[:, None].expand_as(col)
            self.tok.index_put_((row[mask], col[mask]), tr[mask].to(torch.int32))
            self.tok_len.index_copy_(0, slots, base + n_new)

    def text(self, slot: int) -> tuple[str, list[int]]:
        n = int(self.tok_len[slot])
        if n > self.cap:
            raise RuntimeError(f"token slab overflow: stream in slot {slot} has {n} tokens, capacity {self.cap}")
        ids = [t for t in self.tok[slot, :n].tolist() if t != self.blank]
        return self.m.decoding.decode_tokens_to_str_with_strip_punctuation(ids), ids
