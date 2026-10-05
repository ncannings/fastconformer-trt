"""Greedy cumulative layer dropping on R2: at each round, drop the layer whose removal (with those already dropped)
gives the lowest quick 96-utterance dev WER; candidates are layers 0-20 (21-23 are expensive alone)."""
import random, sys
import jiwer, torch
from whisper_normalizer.english import EnglishTextNormalizer
import finetune_stride as fs
import nemo.collections.asr as nemo_asr
norm = EnglishTextNormalizer()
rows = fs.rows_of("dev_clean"); idx = list(range(len(rows))); random.Random(20261008).shuffle(idx)
dev = sorted([(fs.load_audio(rows[i]), norm(fs.ref_text(rows[i]))) for i in idx[:96]], key=lambda t: len(t[0]))
m, merges, _ = fs.load_variant(nemo_asr, sys.argv[1])
def wer(skip):
    hyps, refs = [], []
    with torch.inference_mode():
        for b in range(0, len(dev), 16):
            ch = dev[b:b + 16]
            sig, sl = fs.batch_audio([w for w, _ in ch])
            hyps += [norm(h) or "<empty>" for h in fs.decode(m, merges, sig, sl, skip)]
            refs += [r for _, r in ch]
    return 100 * jiwer.wer(refs, hyps)
dropped = []
cands = [4, 5, 8, 10, 16, 17, 2, 3, 7, 11, 15, 19, 20, 0, 1, 6, 12, 13]   # ordered by single-drop cost
for rnd in range(8):
    best = min(((wer(dropped + [c]), c) for c in cands if c not in dropped))
    dropped.append(best[1])
    print(f"GREEDY {len(dropped)} dropped {sorted(dropped)}: quick-dev WER {best[0]:.2f}", flush=True)
