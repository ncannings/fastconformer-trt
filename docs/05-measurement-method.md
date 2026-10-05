# 5. Measurement method

This document describes how every speed, accuracy and energy number in this repository was measured: the machine, its clocks and thermal state, the datasets and exact subsets, the text normalisers and the definitions of real-time factor and energy per audio hour. The short version: one DGX Spark, whole-pipeline timing, interleaved A/B repeats, same-session comparisons only and wall-plug energy net of idle.

## 1. Hardware and software

- **Machine:** one NVIDIA DGX Spark. GB10 superchip: Blackwell GPU of compute capability 12.1 (sm_121) with unified LPDDR5x memory (273 GB/s) and a 20-core Grace CPU.
- **Kernels:** the CUTLASS plugins are compiled for the sm_120 family (`-gencode=arch=compute_120f,code=sm_120f`, `engine/plugins/build.sh`), which runs on GB10.
- **Software:** NVIDIA's NeMo 25.11 container (TensorRT 10.13, CUDA 13, PyTorch, ModelOpt) plus CUTLASS 4.8 headers and the scoring packages; see the `Dockerfile`. All scripts run in it through `engine/run_in_container.sh`.
- **Power:** the GPU is power-limited under this workload. In a sustained FP8 GEMM test at a 2,200 MHz cap it reached 142 TFLOP/s at 81 W peak GPU power. Under the full engine the whole machine draws about 190 W at the wall, and the achievable FP8 rate is about 142 to 146 TFLOP/s.

## 2. Clocks

Three clock states appear in this repository:

| State | How | Where used |
|---|---|---|
| Locked at 1,176 MHz (about 1,050 MHz under load, 34 W) | A system service, present at the start of the work | Early PyTorch and first TensorRT numbers only; see [01-stock-weight-engine.md](01-stock-weight-engine.md) section 2 |
| 2,200 MHz cap | `nvidia-smi -lgc 300,2200` | Most engineering A/B measurements |
| Driver default (2,418 MHz) | `nvidia-smi -rgc` | Headline stock-versus-final runs ("stock machine") |

The cap barely matters under load. Fully unlocked, the stock arm gains about 3% and the engine about 2% over the cap (Ultra: 0.7%). Under load the GPU clock averaged 2,350 to 2,370 MHz in the stock arms and 2,180 to 2,250 MHz in the engine arms: the engine runs the GPU harder and hits the power limit before the clock limit, sitting at about 2,210 MHz either way.

Before trusting any roofline estimate, check the applied clocks (`nvidia-smi -q -d CLOCK`) and measure a sustained GEMM (`engine/gemm_hold.py`). The early ceiling estimate of 2,500 to 2,900x for the 0.6B model was an artefact of the locked clock.

## 3. Heat soak and thermal safety

GB10 throughput depends on temperature, so a cold first arm would flatter whichever configuration runs first. The headline protocol (`engine/stock_vs_final.sh`) therefore:

1. records 60 s of idle wall power;
2. heat-soaks the GPU with a sustained FP8 GEMM (`engine/gemm_hold.py` in a loop) for 300 s;
3. runs the stock arm, re-soaks for 120 s, then runs the engine arm;
4. logs GPU clock, temperature, power and utilisation every 2 s throughout.

`engine/temp_watchdog.sh` stops the run if the GPU reaches 92 °C. It never tripped; the peak was 85 °C. The GPU counters did show brief software and hardware thermal slowdown events during the unlocked runs. An extra fan was fitted to the machine for this work.

## 4. Repeats and noise

- Each timed measurement is the median of several passes after one warm-up pass: 5 passes in the headline protocol, 2 by default in `engine/trt_rtf.py`.
- Engineering A/B comparisons were run as **interleaved repeats** (A, B, A, B in the same session), and both readings of each arm are reported, for example "4,587 / 4,566 against 4,464 / 4,471".
- Only same-session comparisons are claimed. Background load on the machine varied between sessions (visible as idle power between 36.8 W and 75.8 W), so absolute numbers from different sessions should not be compared closely.

Observed noise:

| Quantity | Run-to-run noise |
|---|---|
| RTF | about 1% (for example 4,415 to 4,435 for one engine) |
| test-clean WER | about 0.03 (the same engine scored 2.413 and 2.440 on two runs; see [04-small-model-110m.md](04-small-model-110m.md)) |

The pipeline is not bit-deterministic run to run: engines that should be equivalent differ on a small fraction of transcripts in both directions, from fp16 rounding. Differences of a few hundredths of a WER point are noise unless they repeat.

Sample size matters too. A 96-utterance quick-dev set used during training reads optimistic, and a 400-utterance Earnings-22 sample gave one reading of 11.06 that the 2,000-utterance sample showed to be noise. The later work therefore uses 2,000-utterance samples or full sets.

## 5. Datasets

All fetched by `engine/fetch_data.sh` into `$ASR_DATA_DIR` (default `~/asr_data`).

| Set | Source | What is scored |
|---|---|---|
| LibriSpeech test-clean | `openslr/librispeech_asr`, clean/test | **All 2,620 utterances.** Held out: never used for selection or calibration. Also the speed set for every RTF in the headline tables. |
| LibriSpeech dev-clean | `openslr/librispeech_asr`, clean/validation (2,703 utterances) | A **2,000-utterance seeded sample** (seed 20261005). Early work used a 400-utterance sample with the same seed, and RTF on all 2,703 utterances. |
| Earnings-22 | `distil-whisper/earnings22`, chunked test split, **first shard only** (`test-00000-of-00038`) | A **2,000-utterance seeded sample** (seed 20261005) from that one shard. Rows with empty references are excluded. |
| FLEURS | `google/fleurs`, 25 languages of parakeet-tdt-0.6b-v3 | **Full test split** for each language. The dev split is used only for multilingual FP8 calibration. |

**Important:** the Earnings-22 figures in this repository come from only the first of 38 shards of the chunked test split. They are useful for comparing configurations with each other, not for comparison with published Earnings-22 results on the full test set.

Other fixed subsets:

| Subset | Seed | Use |
|---|---|---|
| FP8 calibration, default: 128 dev-clean utterances | 20261009 | ModelOpt PTQ (`engine/lean_export.py`) |
| FP8 calibration, multilingual (`LEAN_CALIB=multi`): 32 dev-clean + 4 per FLEURS dev language, 132 utterances | 20261009 | ModelOpt PTQ for the 25-language engine |
| Quick-dev: 96 dev-clean utterances | 20261008 | Monitoring during training and training-free probes only |

Training (for [03-pruning-and-distillation.md](03-pruning-and-distillation.md) only) used LibriSpeech train-clean-100 and train-clean-360 plus AMI IHM train, never a test set.

## 6. Text normalisation and WER

- **English sets** (LibriSpeech, Earnings-22, FLEURS en_us): Whisper's English normaliser (`whisper_normalizer.english.EnglishTextNormalizer`), the one used by the Open ASR Leaderboard.
- **Other FLEURS languages:** Whisper's basic normaliser (`whisper_normalizer.basic.BasicTextNormalizer`: lower case, punctuation removed, language-neutral).
- WER is computed with `jiwer` over the whole set (corpus WER), in percent. An empty hypothesis is scored as `<empty>`.

Because the FLEURS normaliser differs from the one behind NVIDIA's published numbers, absolute FLEURS WERs here sit above the model card (for example v3 Greek 37.08 here against 35.71 with leaderboard-style scoring). Only relative comparisons with the same scorer are claimed; see [02-ultra-25-languages.md](02-ultra-25-languages.md).

## 7. Real-time factor

**RTF = total audio seconds of the set / median wall seconds of a timed pass.**

What a timed pass covers:

- Before timing, the whole set is loaded, decoded to 16 kHz waveforms, sorted by length and padded into batches. In the default runners (`engine/stock_rtf.py`, `engine/trt_rtf.py`) these padded batches are placed in GPU memory before the clock starts; with `--cpu-pre` they stay in host memory and preprocessing runs on the CPU.
- The timed pass then runs the whole pipeline: mel preprocessing, encoder, decoding and conversion of the hypotheses to text. Each pass starts and ends with `torch.cuda.synchronize()`.
- One warm-up pass precedes the timed passes.
- Batch 32, sorted by length, for both stock and engine, unless frame-budget batching is stated (`--frame-budget 4800 --max-batch 128`).
- Audio file reading and decoding from disk are not timed. WER is measured in a separate, untimed pass over the scored set.

`engine/trt_rtf.py` also reports a stage breakdown (preprocessing, encoder, decoder) from one extra pass with a synchronisation between stages; that pass is diagnostic and does not enter the RTF.

## 8. Energy

- **Meter:** a Tapo P304M smart power strip at the wall, sampled every 1 s. This is whole-machine power (GPU, CPU, memory, fans, power supply losses), which is the honest unit for a single-box system; it is never mixed with GPU-only power. `engine/stock_vs_final.sh` accepts any logger through `POWER_LOG_CMD` that writes `t_unix,watts` rows.
- **Idle baseline:** 60 s of wall power with the GPU idle, before the soak.
- **Window:** the timed passes of each arm (their start and end times are recorded by the runner).
- **Gross:** mean wall power over the window times the window length, divided by the audio hours processed in it (audio hours per pass times the number of passes). Reported as J per audio hour.
- **Net of idle:** the same with the idle mean subtracted from the mean wall power first. Idle varied between sessions (35.7 W to 75.8 W) because of other background load on the machine, so net is the more comparable figure across sessions, and only same-session stock-versus-final pairs are claimed.

The `results/headline/*.json` files record, per arm: test-clean WER, RTF, mean and maximum wall power, gross and net J per audio hour, the mean SM clock while the GPU was more than 50% utilised and peak temperature, plus the session's idle power and the speed-up.

## Earnings-22 on the full leaderboard test set

The development figures above use 2,000-utterance samples from one shard. For publication, Earnings-22 was also
scored on the full Open ASR Leaderboard test set (`hf-audio/open-asr-leaderboard`, `earnings22/test`, 2,741
utterances, 5.4 hours; set name `earnings22_full`), at the 2,200 MHz cap:

| Model | Stock NeMo WER | Engine WER | Stock NeMo RTF | Engine RTF |
|---|---|---|---|---|
| moondream/parakeet-ultra (engine F) | 10.03% | 10.07% | 937x | 4,671x |
| nvidia/parakeet-tdt-0.6b-v3 (batch-32 engine) | 11.09% | 10.94% | 916x | 4,253x |

For reference, NVIDIA's model card gives 11.42% for v3 on this set and Moondream's leaderboard-pipeline scoring gives
10.75%; the differences come from scoring details. Source: `results/earnings22/earnings22_full.json`.
