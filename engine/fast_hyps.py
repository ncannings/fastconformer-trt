"""Drop-in replacement for NeMo's rnnt_utils.batched_hyps_to_hypotheses (alignment-free path), same outputs.

NeMo copies the decoder's full preallocated buffers ([B, max_steps], e.g. [128, 3750]) to pageable host memory and
slices GPU tensors per utterance with GPU-resident lengths (a host sync per utterance). Profiled on GB10 at 360 ms of
the decoder's 0.86 s per test-clean pass. Here: one sync for the lengths, copy only the used prefix of each buffer,
slice on the CPU. With alignments requested, the original function is used. install() patches NeMo in place."""
from __future__ import annotations

import torch

_orig = None


def batched_hyps_to_hypotheses(batched_hyps, alignments=None, batch_size=None):
    from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
    if alignments is not None:
        return _orig(batched_hyps, alignments, batch_size)
    assert batch_size is None or batch_size <= batched_hyps.scores.shape[0]
    n = batched_hyps.scores.shape[0] if batch_size is None else batch_size
    lengths = batched_hyps.current_lengths[:n].cpu()
    L = max(1, int(lengths.max())) if n else 1
    scores = batched_hyps.scores[:n].cpu()
    transcript = batched_hyps.transcript[:n, :L].cpu()
    timestamps = batched_hyps.timestamps[:n, :L].cpu()
    durations = batched_hyps.token_durations[:n, :L].cpu() if batched_hyps.is_with_durations else None
    out = []
    for i in range(n):
        k = int(lengths[i])
        out.append(Hypothesis(score=scores[i].item(), y_sequence=transcript[i, :k], timestamp=timestamps[i, :k],
                              token_duration=durations[i, :k] if durations is not None else torch.empty(0),
                              alignments=None, dec_state=None))
    return out


def install() -> None:
    global _orig
    from nemo.collections.asr.parts.utils import rnnt_utils
    if _orig is None:
        _orig = rnnt_utils.batched_hyps_to_hypotheses
        rnnt_utils.batched_hyps_to_hypotheses = batched_hyps_to_hypotheses
