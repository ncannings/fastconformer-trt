"""One-shot 2:4 pruning damage before any recovery training: R2 checkpoint (pool 4->3 after L9), masks by magnitude or
Wanda on chosen linear groups, quick 96-utterance dev WER (seed 20261008, as the training quick-dev). Calibration for
Wanda: 128 train-clean-100 utterances. usage: prune_probe.py CKPT VARIANT...   VARIANT = method:groups e.g. wanda:ff"""
import glob, os, random, sys
import jiwer, torch, pyarrow.parquet as pq
from whisper_normalizer.english import EnglishTextNormalizer
import finetune_stride as fs
import nemo.collections.asr as nemo_asr
os.environ.setdefault("POOL_RULE", "pool43")
ck, variants = sys.argv[1], sys.argv[2:]
norm = EnglishTextNormalizer()
rows = fs.rows_of("dev_clean"); idx = list(range(len(rows))); random.Random(20261008).shuffle(idx)
dev = [(fs.load_audio(rows[i]), norm(fs.ref_text(rows[i]))) for i in idx[:96]]
dev.sort(key=lambda t: len(t[0]))
cr = pq.read_table(sorted(glob.glob(os.path.join(fs.DATA, "train100", "*.parquet")))[0]).slice(0, 128)
ws = sorted([w for w in (fs.load_audio(r) for r in cr.to_pylist()) if len(w) <= 16 * 16000], key=len)
calib = [fs.batch_audio(ws[b:b + 16]) for b in range(0, len(ws), 16)]
for v in ["none"] + variants:
    m, merges, skip = fs.load_variant(nemo_asr, ck)
    if v != "none":
        method, groups = v.split(":")
        fs.sparse24_masks(m, groups.replace("+", ","), calib if method == "wanda" else None)
    hyps, refs = [], []
    with torch.inference_mode():
        for b in range(0, len(dev), 16):
            ch = dev[b:b + 16]
            sig, sl = fs.batch_audio([w for w, _ in ch])
            hyps += [norm(h) or "<empty>" for h in fs.decode(m, merges, sig, sl, skip)]
            refs += [r for _, r in ch]
    print(f"PRUNE {v}: quick-dev WER {100 * jiwer.wer(refs, hyps):.2f}", flush=True)
    del m; torch.cuda.empty_cache()
