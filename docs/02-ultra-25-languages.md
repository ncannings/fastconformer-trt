# 2. parakeet-ultra across 25 languages

parakeet-ultra is Moondream's post-trained version of parakeet-tdt-0.6b-v3: same architecture, better weights. Run through the engine of [01-stock-weight-engine.md](01-stock-weight-engine.md) with a few further tunings, it reaches 4,903x real time against 999x for stock NeMo on the same model, at 104 against 406 J per audio hour net. It is also more accurate than stock v3 on all 25 FLEURS languages. This document covers the conversion, the 25-language evaluation, the engine tuning, the NVFP4 experiment and an independent analysis of how much further the FP8 path could go.

## 1. Converting the checkpoint

`moondream/parakeet-ultra` is published as a Hugging Face transformers checkpoint. `engine/ultra_to_nemo.py HF_REPO OUT.nemo` loads the v3 NeMo model and replaces its weights:

- every NeMo parameter and buffer must be filled exactly once with a matching shape, or the script refuses;
- extra transformers tensors are listed and ignored (only Moondream's VAD head is left out).

The result is a `.nemo` file, so the whole pipeline runs it unchanged via `ASR_MODEL=/data/ultra.nemo` (export, engine build, `trt_rtf.py` and the stock arm `stock_rtf.py`).

First result, same engine configuration as the final 0.6B-v3 engine (residual folded into linear_out and pw2), 2,200 MHz cap, batch 32:

| Ultra | test-clean (full) | dev-clean (2,000) | Earnings-22 (2,000) | RTF test-clean |
|---|---|---|---|---|
| Stock NeMo | 1.788 | 1.836 | 9.16 | 970 |
| Engine | 1.788 | 1.826 | 9.31 | **4,622** (4.76x) |

Against v3 on the same engine (1.918 / 1.95 / 10.52 at 4,576x): the same speed, and better on all three English sets.

## 2. The FLEURS evaluation

### Method

- **Sets:** the full FLEURS test split for each of the 25 languages parakeet-tdt-0.6b-v3 supports (`engine/fetch_data.sh fleurs`, converted by `engine/fleurs_to_parquet.py`). The FLEURS dev split is used only for multilingual FP8 calibration, never for scoring.
- **Normalisers:** Whisper's basic normaliser (`whisper_normalizer.basic.BasicTextNormalizer`: lower case, punctuation removed, language-neutral) for every language except English; Whisper's English normaliser for en_us.
- **Arms:** stock NeMo (`engine/stock_rtf.py`) and the engine (`engine/trt_rtf.py`), same audio, same normaliser.

### Why only relative comparisons are claimed

The absolute WERs here sit above NVIDIA's model card, because our normaliser differs from the one behind the card. For example, Moondream's leaderboard-style scoring gives v3 Greek 35.71; ours gives 37.08. Rather than chase a matching normaliser, this repository claims only comparisons made with the same scorer: engine against stock on the same model, and Ultra against v3.

### First check: v3, 12 languages

Stock NeMo against the FP8 engine on v3, first 12 languages: mean 14.86 against 14.89, no language worse by more than 0.20 absolute (et) or 1.1% relative (hr); el and en were slightly better on the engine. The FP8 calibration, done on English only at that point, held on Latin, Cyrillic and Greek scripts.

### Results, all 25 languages

From `results/fleurs/fleurs_wer_25_languages.json` (WER %, full FLEURS test sets). "Engine F" is the final FP8 engine described in section 3.

| Language | v3 stock NeMo | Ultra stock NeMo | Ultra, engine F | Engine minus stock (Ultra) |
|---|---|---|---|---|
| bg_bg | 14.425 | 12.749 | 12.835 | +0.086 |
| cs_cz | 14.036 | 12.873 | 12.821 | -0.052 |
| da_dk | 20.243 | 17.723 | 17.677 | -0.046 |
| de_de | 6.844 | 6.287 | 6.255 | -0.032 |
| el_gr | 37.079 | 34.147 | 34.403 | +0.256 |
| en_us | 6.674 | 6.002 | 6.009 | +0.007 |
| es_419 | 5.135 | 4.498 | 4.463 | -0.035 |
| et_ee | 19.021 | 15.612 | 15.742 | +0.130 |
| fi_fi | 15.365 | 12.907 | 12.913 | +0.006 |
| fr_fr | 6.894 | 6.300 | 6.209 | -0.091 |
| hr_hr | 14.458 | 13.371 | 13.014 | -0.357 |
| hu_hu | 18.111 | 15.156 | 15.109 | -0.047 |
| it_it | 5.085 | 4.416 | 4.455 | +0.039 |
| lt_lt | 24.372 | 19.456 | 19.444 | -0.012 |
| lv_lv | 25.856 | 20.086 | 20.532 | +0.446 |
| mt_mt | 22.386 | 18.117 | 18.204 | +0.087 |
| nl_nl | 9.084 | 8.249 | 8.236 | -0.013 |
| pl_pl | 9.112 | 7.978 | 7.928 | -0.050 |
| pt_br | 6.741 | 6.122 | 6.165 | +0.043 |
| ro_ro | 14.422 | 12.011 | 12.080 | +0.069 |
| ru_ru | 8.663 | 7.782 | 7.796 | +0.014 |
| sk_sk | 11.682 | 9.495 | 9.723 | +0.228 |
| sl_si | 26.240 | 21.015 | 21.411 | +0.396 |
| sv_se | 17.028 | 14.698 | 14.818 | +0.120 |
| uk_ua | 7.902 | 6.717 | 6.710 | -0.007 |
| **Mean (25)** | **14.674** | **12.551** | **12.598** | **+0.047** |

Reading:

- Ultra through the engine beats stock v3 on 25 of 25 languages (mean 12.60 against 14.67).
- Engine F against stock Ultra: +0.05 mean, worse on 14 languages and better on 11. The largest losses are lv (+0.45) and sl (+0.40), both low-resource.
- An earlier engine (before the tuning in section 3, English-only calibration) measured +0.08 mean against stock Ultra (0.6% relative), worst lv +0.31 and sl +0.32, with 20 of 25 languages slightly worse. Its engine RTF on FLEURS was 4,386x (FLEURS clips are shorter than LibriSpeech utterances).

## 3. Engine tuning on Ultra

All at the 2,200 MHz cap, test-clean full, interleaved repeats so that both arms of each A/B see the same machine state. Step labels follow the order in which changes were tried.

| Step | Change | Measured | Decision |
|---|---|---|---|
| A | Frame-budget batching, engine built for batch 128 (`--frame-budget 4800 --max-batch 128`) | 4,757 / 4,769 against 4,608 / 4,620 at batch 32: **+3.2%**; WER 1.818 against 1.814 | Adopted |
| B | Cached relative-position table (`LEAN_POSCACHE=1`): `linear_pos` and the (pos_bias_v - pos_bias_u) term precomputed in full precision for T <= 750 and sliced. Slice check exact (4.8e-4 relative, all layers, T = 40 / 333 / 750). | Encoder 80.6 against 81.9 ms; end to end 4,819 against 4,752: **+1.4%** | Adopted |
| C | Multilingual FP8 calibration (`LEAN_CALIB=multi`): 32 dev-clean utterances plus 4 per FLEURS **dev** language, 132 utterances; test sets never used | FLEURS mean 12.60 against 12.60 for English-only calibration; Earnings-22 9.16 against 9.32 | No measurable gain; adopted as the principled default for 25 languages |
| D | Exact subsampling masking (`LEAN_SUBMASK=1` instead of mask-once) | Engine-vs-NeMo equivalence 0.0015 (was 0.064); Earnings-22 9.157 (equal to stock); FLEURS mean 12.613 against 12.602; 2.8% slower (4,687 / 4,681) | Not adopted: the remaining FLEURS gap to stock is FP8 itself, not masking |
| E | pw1 + GLU fusion (`GLU_FUSE=1`, PwGluFp8 plugin, `engine/plugins/glu_plugin.cpp`): gate GEMM to fp16 scratch, value GEMM with value * sigmoid(gate) in the epilogue; kernel relative error 2e-4 | The depthwise kernel drops from 3.3 to 1.4 ms, but the two GEMMs cost 6.6 against TensorRT's 5.1 ms; end to end 4,800 / 4,773 against 4,824 / 4,822 | Not adopted: it needs a single-GEMM column-pair epilogue to pay |
| F | conv.0 + conv.2 + conv.3 (1x1, tensor cores via WMMA) + ReLU in one SubConv02 kernel (`SUB_PW3=1`) | 3.48 ms against 4.25 for the three steps it replaces (P = 32 positions per block; 128 was 4.93 ms with one block per SM, 64 was 3.59); end to end 4,838 / 4,854 against 4,804 / 4,816: **+0.7%**; WER 1.795 / 1.851 / 9.26 against 1.803 / 1.836 / 9.16 (noise) | Adopted (engine F) |

Decoder group size was rechecked on Ultra: 256 gave 4,821 / 4,778, 512 gave 4,739 / 4,763 and 1,024 gave 4,582 / 4,586. Smaller groups crash in the compiled featurizer (a stride assertion). It stays at 256.

After step C the configuration ran at 4,800 to 4,820x capped, English test-clean 1.80 / dev-clean 1.84 / Earnings-22 9.16 against stock Ultra 1.79 / 1.84 / 9.16 and FLEURS mean 12.60 against stock Ultra 12.55.

After step F, the FP8 path is at its floor on this model: every remaining fusion that was tried is worth less than 1%.

## 4. NVFP4 feed-forward

### The accuracy probe (v3, before any kernel work)

`engine/probes/nvfp4_probe.py`: ModelOpt fake quantisation of the stock v3 encoder's linears, 2,000-utterance samples. Conv modules are not quantised here, unlike in the engine, so compare variants with each other, not with engine WER.

| Variant | dev-clean | Earnings-22 |
|---|---|---|
| FP8 everywhere (deployed scheme) | 1.952 | 10.331 |
| NVFP4 FF, FP8 elsewhere | 2.016 | 10.229 |
| NVFP4 FF except layers 0, 1, 22, 23 | 2.011 | 10.169 |
| NVFP4 FF with AWQ-lite | 1.984 | 10.530 |

NVFP4 in the FF linears cost about +0.03 to +0.06 on dev-clean, and moved Earnings-22 by -0.16 to +0.20 depending on calibration. The expected engine gain was roughly 10 ms of 87 per batch, about 9% end to end.

### The kernel

`engine/plugins/nvfp4_ffn.cu` builds the sm_120 block-scaled CUTLASS GEMMs: NVFP4 x NVFP4; GEMM1 with SiLU and NVFP4 output plus scale factors, using `LinCombPerColBiasEltActBlockScaleFactor` (the only non-pointer-array sm_120 specialisation with an activation); and an activation quantiser that indexes scale factors through CUTLASS's own layout.

Against CUTLASS 4.1 it failed to compile at `can_implement`, an Arguments type mismatch between kernel instantiations in the 4.1 headers, and was parked. It compiles against **CUTLASS 4.8**. All plugins now build against 4.8 (`CUTLASS_DIR`; the `Dockerfile` pins it), and engine F gives identical WER and speed with either build (4,868 / 4,834 against 4,851 / 4,842).

FFNFp8 gained an fp4 mode (`FFN_FP4=1`):

- the fp16 input is quantised in the plugin (global scale s_x / 6);
- GEMM1 with SiLU and NVFP4 output (scale s_h / 6);
- GEMM2 with the residual folded in, as in FP8 mode;
- FP8 weight bytes are re-quantised to NVFP4 at load;
- `FFN_FP4_SKIP` keeps chosen layers in FP8.

Kernel block: 0.688 against 0.772 ms. On random weights the FF output error is 19% against 5.4% for FP8.

### Results on Ultra

| Ultra engine (2,200 MHz cap) | test-clean | dev-clean | Earnings-22 | FLEURS 25 mean | Worst language | RTF |
|---|---|---|---|---|---|---|
| Stock NeMo | 1.788 | 1.836 | 9.16 | 12.55 | | 970 |
| F: FP8 (default) | 1.795 to 1.814 | 1.851 | 9.26 | 12.60 (+0.05) | lv +0.45 | 4,842 |
| G: NVFP4, all FF | 1.827 | 1.967 | 9.28 | 13.30 (+0.74) | lv +1.89 | 5,089 |
| H: NVFP4 FF, layers 0, 1, 22, 23 kept FP8 | 1.812 | 1.960 | 9.21 | 13.14 (+0.59) | lv +1.47 | 4,896 to 5,050 |

Per-language FLEURS numbers for G and H are in the same JSON file.

Verdict: post-training NVFP4 buys about 5% speed for about 5% more errors across the 25 languages, with the low-resource languages worst (lv, sl, lt, da, mt). It is not the default; it is kept as an opt-in switch. Engine G is still better than stock v3 on 24 of 25 languages.

## 5. Headline: stock machine

Controlled run with `engine/stock_vs_final.sh` (driver-default clocks, 300 s heat soak, 120 s re-soak between arms, Tapo P304M wall meter at 1 s, full test-clean, 5 timed passes per arm), engine F with frame-budget batching. From `results/headline/ultra_engineF_stock_machine.json`:

| Ultra | test-clean | RTF | Wall W, mean | J per audio hour, gross | J per audio hour, net of idle | Loaded SM clock, mean | Peak temperature |
|---|---|---|---|---|---|---|---|
| Stock NeMo | 1.803 | 999 | 185.1 | 666.8 | 405.7 | 2,375 MHz | 84 °C |
| **Engine F** | 1.814 | **4,903** | 190.2 | 168.1 | **104.1** | 2,262 MHz | 81 °C |

Idle 72.5 W. **4.91x faster, 3.9x less energy per audio hour.** FLEURS 25-language mean 12.60 against stock Ultra 12.55 and stock v3 14.67.

An earlier stock-machine run with the step C configuration gave 999x against 4,858x, 674 / 416 against 170 / 106 J per audio hour (gross / net of a 71 W idle), with the loaded SM clock at 2,375 MHz for stock and 2,213 MHz for the engine. Unthrottled is only 0.7% above the 2,200 MHz cap: under the engine's load the GPU sits at about 2,210 MHz either way, because it is power-limited.

## 6. How far can the FP8 path go?

After the final v3 engine was built, an independent review re-derived its profile (88.2 ms per 32 x 16 s batch). The same analysis applies to Ultra, which has the same architecture and shapes.

- The encoder's GEMMs are 7.41 TFLOP per batch, running at about 123 TFLOP/s, 85% of the power-limited 142 to 146 TFLOP/s.
- **The FP8 GEMM floor alone caps this model at about 9,600x on this machine**, with zero overhead anywhere else. 10,000x is out of reach at FP8 without changing the model.
- Realistic estimates: about 5,900x with all remaining FP8 work landed, and about 7,000 to 7,500x with NVFP4.
- The layer-norm cost (13.5 ms per batch) is mostly the residual read and write, not the normalisation itself.

With the measured result at 4,903x and every remaining FP8 fusion below 1%, the next large step needs fewer FLOPs: lower precision (section 4, with its accuracy cost), fewer frames or layers ([03-pruning-and-distillation.md](03-pruning-and-distillation.md)) or a smaller model ([04-small-model-110m.md](04-small-model-110m.md)).
