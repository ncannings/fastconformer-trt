"""Training-free cost of skipping one encoder layer entirely on R2 (4->3 pool after L9): quick 96-utterance dev WER per
skipped layer, to pick layers for distilled removal."""
import random, sys
import jiwer, torch
from whisper_normalizer.english import EnglishTextNormalizer
import finetune_stride as fs
import nemo.collections.asr as nemo_asr
norm = EnglishTextNormalizer()
rows = fs.rows_of("dev_clean"); idx = list(range(len(rows))); random.Random(20261008).shuffle(idx)
dev = sorted([(fs.load_audio(rows[i]), norm(fs.ref_text(rows[i]))) for i in idx[:96]], key=lambda t: len(t[0]))
m, merges, _ = fs.load_variant(nemo_asr, sys.argv[1])
for l in range(24):
    hyps, refs = [], []
    with torch.inference_mode():
        for b in range(0, len(dev), 16):
            ch = dev[b:b + 16]
            sig, sl = fs.batch_audio([w for w, _ in ch])
            hyps += [norm(h) or "<empty>" for h in fs.decode(m, merges, sig, sl, [l])]
            refs += [r for _, r in ch]
    print(f"DROP layer {l}: quick-dev WER {100 * jiwer.wer(refs, hyps):.2f}", flush=True)
