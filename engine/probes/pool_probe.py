"""Training-free cost of an extra frame pool on top of R2 (4->3 pool after L9, distilled): quick 96-utterance dev WER
with an added pool43 or pool2 after layer L, to choose where the next pool goes (and estimated encoder work)."""
import random, sys
import jiwer, torch
from whisper_normalizer.english import EnglishTextNormalizer
import finetune_stride as fs, encoder_merge as em
import nemo.collections.asr as nemo_asr
ck = sys.argv[1]
norm = EnglishTextNormalizer()
rows = fs.rows_of("dev_clean"); idx = list(range(len(rows))); random.Random(20261008).shuffle(idx)
dev = sorted([(fs.load_audio(rows[i]), norm(fs.ref_text(rows[i]))) for i in idx[:96]], key=lambda t: len(t[0]))
m, merges, skip = fs.load_variant(nemo_asr, ck)
def work(pools):
    f, w = 1.0, 0.0
    for l in range(24):
        w += f
        if l in pools: f *= 0.75 if pools[l] is em.pool43 else 0.5
    return w / 24
for extra in [None, (4, "pool43"), (6, "pool43"), (12, "pool43"), (14, "pool43"), (14, "pool2"), (16, "pool2"), (18, "pool2")]:
    mg = dict(merges)
    if extra:
        mg[extra[0]] = getattr(em, extra[1])
    hyps, refs = [], []
    with torch.inference_mode():
        for b in range(0, len(dev), 16):
            ch = dev[b:b + 16]
            sig, sl = fs.batch_audio([w for w, _ in ch])
            hyps += [norm(h) or "<empty>" for h in fs.decode(m, mg, sig, sl, skip)]
            refs += [r for _, r in ch]
    print(f"POOL extra {extra}: quick-dev WER {100 * jiwer.wer(refs, hyps):.2f}, encoder work {work(mg):.3f}", flush=True)
