# fastconformer-trt

**Multilingual speech recognition at 4,903x real time on one NVIDIA DGX Spark: 4.9x faster than stock NeMo, at the
same accuracy, on a quarter of the energy.**

fastconformer-trt turns an NVIDIA NeMo FastConformer checkpoint (Parakeet TDT and CTC models) into an FP8 TensorRT
engine with hand-written CUDA/CUTLASS plugins, then runs it in a pipelined batch transcriber. Nothing is retrained:
the weights are the published ones, quantised after training. Every optimisation was kept only if it held accuracy,
in English and across 25 European languages.

The headline was measured on a single DGX Spark (GB10 Grace Blackwell, sm_121, 128 GB unified memory); results on
GH200, H100 and B200 datacentre GPUs follow below.

## Headline

[parakeet-ultra](https://huggingface.co/moondream/parakeet-ultra) (Moondream's post-trained parakeet-tdt-0.6b-v3:
same architecture and tokenizer, 25 languages), LibriSpeech test-clean (2,620 utterances, 5.4 hours). Timed from
16 kHz audio already in GPU memory to text out: mel features, encoder, decoder and text (see measurement notes). Stock machine: GPU clocks at driver default, 300 s heat soak, both arms in the same
session, wall-plug energy.

| parakeet-ultra | test-clean WER | Real-time factor | Energy per audio hour (wall, net of idle) |
|---|---|---|---|
| Stock NeMo (bf16, CUDA-graph greedy decoder, batch 32) | 1.803% | 999x | 406 J |
| **fastconformer-trt** | **1.814%** | **4,903x** | **104 J** |

One hour of audio is transcribed in 0.73 seconds of GPU pipeline time (audio already decoded; see measurement notes).

### Accuracy across 25 languages

FLEURS test sets, all 25 languages the model supports, full sets (about 20,000 utterances):

| | FLEURS mean WER, 25 languages | Worst language vs stock Ultra |
|---|---|---|
| Stock NeMo, nvidia/parakeet-tdt-0.6b-v3 | 14.67% | |
| Stock NeMo, moondream/parakeet-ultra | 12.55% | |
| **fastconformer-trt, parakeet-ultra** | **12.60%** | Latvian +0.45 |

fastconformer-trt running Ultra beats stock NeMo running NVIDIA's v3 on **every one of the 25 languages**, and stays
within 0.4% (relative) of stock Ultra. Per-language results: [docs/02-ultra-25-languages.md](docs/02-ultra-25-languages.md).

### Earnings calls (English, full test set)

Earnings-22, the Open ASR Leaderboard test set (2,741 utterances, 5.4 hours), scored in full:

| | Stock NeMo | fastconformer-trt |
|---|---|---|
| moondream/parakeet-ultra | 10.03% | 10.07% |
| nvidia/parakeet-tdt-0.6b-v3 | 11.09% | 10.94% |

### Other models, same engine

Controlled stock-machine runs, test-clean, stock weights:

| Model | Languages | Stock NeMo (WER / RTF) | fastconformer-trt (WER / RTF) | Speed-up |
|---|---|---|---|---|
| moondream/parakeet-ultra | 25 | 1.803 / 999x | 1.814 / **4,903x** | **4.91x** |
| nvidia/parakeet-tdt-0.6b-v3 | 25 | 1.931 / 994x | 1.910 / 4,872x | 4.90x |
| nvidia/parakeet-tdt-1.1b | English | 1.386 / 684x | 1.392 / 2,991x | 4.37x |
| nvidia/parakeet-ctc-1.1b | English | 1.844 / 727x | 1.833 / 3,094x | 4.26x |
| nvidia/parakeet-tdt_ctc-110m | English | 2.433 / 2,397x | 2.453 / **12,109x** | 5.05x |

The 1.1B rows were measured before the last round of optimisations (frame-budget batching, cached position table,
residual folding, fused subsampling), which added about 6% on the 0.6B models. The 110M is a much smaller,
English-only model; it is fast but about 25% worse on LibriSpeech than the 0.6B models.

### On datacentre GPUs: GH200, H100 and B200

The same engine on rented datacentre GPUs, Ultra, same test sets and runner (default clocks, throughput figures):

| parakeet-ultra | Stock NeMo (test-clean / Earnings-22 / RTF) | fastconformer-trt (test-clean / Earnings-22 / RTF) | Speed-up |
|---|---|---|---|
| NVIDIA GH200 (Hopper) | 1.801 / 10.03 / 3,873x | 1.820 / 10.04 / **22,776x** | 5.9x |
| NVIDIA H100 SXM (Hopper) | 1.795 / 10.06 / 4,693x | 1.841 / 10.09 / **20,667x** | 4.4x |
| NVIDIA B200 (Blackwell) | 1.808 / 10.05 / 5,743x | 1.816 / 10.05 / **25,073x** | 4.4x |

On all three, TensorRT's own FP8 GEMMs beat our Spark-tuned CUTLASS kernels for the large matrix multiplies, while the
fused attention and fused subsampling plugins remain essential (without the subsampling plugin the GH200 runs at
1,216x). On datacentre GPUs the TDT decoder becomes the main limit. See [docs/06-gh200.md](docs/06-gh200.md),
[docs/08-h100.md](docs/08-h100.md) and [docs/07-b200.md](docs/07-b200.md).

## Choosing a speed / accuracy point

| Configuration | Speed (Ultra) | Accuracy cost vs stock Ultra | Retraining |
|---|---|---|---|
| **Default: FP8 engine** | **4,903x** | FLEURS +0.4%, test-clean +0.6% (within run-to-run noise) | none |
| NVFP4 feed-forward, first and last two layers FP8 (`FFN_FP4=1 FFN_FP4_SKIP=0,1,22,23`) | about +3% | FLEURS +4.7%, dev-clean +6% | none |
| NVFP4 feed-forward, all layers (`FFN_FP4=1`) | +5% (5,089x at a 2,200 MHz cap) | FLEURS +6%, dev-clean +6%; low-resource languages worst (Latvian +1.9) | none |
| Frame pooling after layer 9, distilled (v3, English only) | +13% | test-clean +2%, dev-clean +2.6%, Earnings-22 2% better | 9.5 h |
| + 4 layers dropped / + 2:4 sparsity (v3, English only) | +30% / +36% | test-clean +19% / +35% | +1.5 h / +2 h |

The retrained rows were distilled on English data from v3 and never checked on the other 24 languages. The full
story, including the ideas that did not work, is in [docs/](docs/).

## How it works

A FastConformer encoder is about 90% of the work. Per batch of 32 sixteen-second utterances, the final engine spends
about 80 ms in the encoder:

| Part | ms | What we did |
|---|---|---|
| Feed-forward blocks (48) | 39.3 | **FFNFp8 plugin**: each block is two CUTLASS FP8 GEMMs; SiLU and the FP8 re-quantise run in GEMM1's epilogue, the half-step residual add in GEMM2's. The 4x-wide hidden tensor never leaves FP8. |
| Q, K, V projection | 9.1 | **QKVHeads plugin**: one batched FP8 GEMM per part that writes heads-first `[H, B*T, dk]` directly, biases (and `pos_bias_u`) in the epilogue, so attention needs no transpose. |
| LayerNorm + FP8 quantise | 6.1 | Left to TensorRT; the residual adds that used to sit in these kernels moved into GEMM epilogues. |
| Conv module (pw1, GLU + depthwise, pw2) | 11.9 | Depthwise conv as 9 shifted multiply-adds; batch norm folded; pw2's residual add in its GEMM epilogue (**SpLinearFp8**, dense mode). |
| Attention | 3.1 + 3.7 | **RelPosAttn plugin**: fused relative-position attention (a Triton kernel shipped as a cubin). The position term is computed on chip: no `[T, 2T-1]` score tensor, padded keys skipped. The projected position table is precomputed once per layer and sliced. Output projection carries the residual add. |
| Subsampling | 3.5 + 2.3 | **SubConv02 plugin**: the first three 8x-subsampling convolutions in one kernel, the 1x1 conv on tensor cores (WMMA), channels-last output, so the 210 MB intermediate is never written. |

Around the encoder:

- **Export-friendly rewrite** of the NeMo encoder (`lean_encoder.py`): masks built once, no fp32 round trips, heads-first attention, equivalence-checked against NeMo before every export.
- **FP8 post-training quantisation** with NVIDIA ModelOpt, calibrated on English and all 25 FLEURS development sets (never the test sets).
- **Decoder**: NeMo's CUDA-graph TDT greedy decoder in bf16, with a fused Triton joint + argmax kernel (FP8 joint weights) and leaner hypothesis handling.
- **Pipeline**: encoder and decoder on separate CUDA streams; `torch.compile`d mel front end; utterances packed to a frame budget (up to 128 short utterances per encoder batch); every batch trimmed to its own longest utterance.

The [measurement method](docs/05-measurement-method.md) and the [stock-weight engine notes](docs/01-stock-weight-engine.md)
give every step with its measured gain.

### What limits it

An independent review of the profile put the FP8 floor for this model at about 9,600x on this machine with zero
overhead: the encoder's 7.4 TFLOP per batch at the GB10's power-limited ~146 TFLOP/s. The GEMMs already run at about
85% of that. What is left is memory traffic in the non-GEMM layers, the decoder and preprocessing, all of which run
on the same GPU. Under this load the GB10 is power-limited, not clock-limited: it averages about 2,260 MHz whether
or not the clock is capped at 2,200 MHz.

## Quick start

Requirements: a DGX Spark or another sm_120-family Blackwell GPU, Docker with the NVIDIA container toolkit, and
access to NGC (`nvcr.io/nvidia/nemo:25.11`). The plugins are compiled for `compute_120f`. Hopper (GH200,
H100) builds with `FC_SM=90` and datacentre Blackwell (B200, GB200) with `FC_SM=100`; both are measured (see above)
and carry the FP8 paths only (the sparse and NVFP4 options refuse with an error). On machines that are themselves the
NeMo container (no Docker, e.g. RunPod), use `NATIVE=1` builds and `engine/cloud_run_native.sh`.

```bash
# 1. image: NeMo 25.11 + scoring packages + CUTLASS 4.8
docker build -t fastconformer-trt:25.11 .

# 2. plugins (CUDA/CUTLASS kernels + TensorRT plugins -> engine/plugins/libffn_fp8.so)
cd engine
docker run --rm -v $PWD:/w -w /w/plugins fastconformer-trt:25.11 bash build.sh

# 3. evaluation data into ~/asr_data (LibriSpeech, Earnings-22 shard, FLEURS test + dev)
./fetch_data.sh

# 4. parakeet-ultra into NeMo format (or use any NeMo hub model name directly as ASR_MODEL)
./run_in_container.sh ultra_to_nemo.py moondream/parakeet-ultra /data/ultra.nemo

# 5. engine (the default configuration)
ASR_MODEL=/data/ultra.nemo MAXB=128 FFN_RESIDUAL=1 LEAN_PREMASK_ONCE=1 LEAN_DWSHIFT=1 LEAN_RELSHIFT=1 \
  LEAN_QKV_PLUGIN=1 LEAN_ATTN_PLUGIN=1 LEAN_HB=1 SP_LINEAR=1 SP_DENSE=1 LEAN_POSCACHE=1 LEAN_CALIB=multi SUB_PW3=1 \
  ./build_variant.sh ultra_f

# 6. transcribe and score (WER on test-clean, speed over 5 passes)
ASR_MODEL=/data/ultra.nemo FAST_JOINT_FP8=1 FFN_PLUGIN_LIB=/w/plugins/libffn_fp8.so ./run_in_container.sh \
  trt_rtf.py /data/ultra_f/engine.plan /data/ultra_f/result.json --batch 32 --frame-budget 4800 --max-batch 128 \
  --dec-batch 256 --dec-bf16 --pipeline --pre-compile --fast-joint --fast-hyps \
  --wer-sets test_clean --wer-n 3000 --rtf-set test_clean --reps 5

# 7. controlled stock-versus-engine comparison (optional wall-plug meter via POWER_LOG_CMD)
FINAL_FLAGS="--frame-budget 4800 --max-batch 128" ./stock_vs_final.sh /data/ultra.nemo /data/ultra_f/engine.plan cmp 300
```

`ASR_DATA_DIR` (default `~/asr_data`) and `ASR_IMAGE` (default `fastconformer-trt:25.11`) override the data
directory and image. Engines are specific to the GPU and TensorRT version, so build them on the machine that runs them.

### Build switches

| Switch | Effect |
|---|---|
| `LEAN_PREMASK_ONCE`, `LEAN_DWSHIFT`, `LEAN_RELSHIFT`, `LEAN_HB` | Export-friendly encoder rewrites (all exact). |
| `LEAN_QKV_PLUGIN`, `LEAN_ATTN_PLUGIN` | QKVHeads and RelPosAttn plugins (head dim 128 or 64). |
| `FFN_RESIDUAL` | Feed-forward plugin with the residual add folded in. |
| `SP_LINEAR=1 SP_DENSE=1` | Attention output and conv pw2 as plugins with the residual add folded in. |
| `SUB_PW3` | Subsampling conv.0 to conv.3 fused (tensor cores). |
| `LEAN_SUBMASK` | Subsampling with NeMo's exact per-layer masking (needed for models whose subsampling convs leak bias into padding, such as the 110M). |
| `LEAN_POSCACHE` | Precomputed relative-position table (exactness checked at export). |
| `LEAN_CALIB=multi` | FP8 calibration on English plus all FLEURS dev sets. |
| `MAXB=128` | Engine profile up to 128 utterances, for `--frame-budget` batching. |
| `FFN_FP4`, `FFN_FP4_SKIP` | NVFP4 feed-forward (opt-in, costs accuracy; see above). |
| `FFN_SPARSE`, `QKV_SPARSE` | 2:4 sparse kernels for pruned checkpoints (`LEAN_CKPT=...`). |
| `FFN_SKIP`, `LEAN_QKV=1` with `LEAN_QKV_PLUGIN=0` | Leave the feed-forward and QKV GEMMs to TensorRT (the Hopper configuration). |
| `FC_SM=90` / `FC_SM=100` (plugin build) | Hopper / datacentre Blackwell builds of the plugins. |

## Repository layout

```
Dockerfile               NeMo 25.11 + scoring packages + CUTLASS 4.8
engine/                  everything that runs (mounted as /w in the container)
  lean_encoder.py        export-friendly FastConformer encoder
  lean_export.py         equivalence check, FP8 calibration, ONNX export
  build_variant.sh       export -> ONNX surgery -> TensorRT engine -> profile
  trt_rtf.py             pipelined transcriber, WER and speed
  stock_rtf.py           the stock NeMo baseline
  stock_vs_final.sh      controlled comparison (heat soak, clocks, temperature, optional wall power)
  ultra_to_nemo.py       transformers parakeet_tdt checkpoint -> NeMo
  fetch_data.sh          evaluation data
  finetune_stride.py     data helpers, and the distillation used for the pruning experiments
  plugins/               CUDA/CUTLASS kernels, TensorRT plugins, ONNX surgery scripts, unit tests
  probes/                accuracy probes used to choose pruning targets
docs/                    the engineering notes: every step, its measured effect, and the dead ends
results/                 headline summaries and per-language FLEURS results
```

## Measurement notes

- **Real-time factor** is audio seconds per wall second: total audio over total wall time across five timed passes.
  The clock covers mel features, encoder, decoder and conversion to text, starting from 16 kHz samples already padded,
  length-sorted and resident on the GPU. Reading and decoding audio files, and the host-to-GPU copy, are **not**
  timed, for both the stock and the engine arm. At these speeds file decoding is a real cost (several CPU seconds for
  5.4 hours of FLAC), so a production pipeline needs parallel decoding ahead of the GPU.
- **WER** uses the Whisper English normaliser for English sets and the Whisper basic normaliser for the other FLEURS
  languages. Our absolute FLEURS numbers differ from NVIDIA's model card (scoring differs), so only comparisons made
  under the same scorer are claimed.
- **Earnings-22** headline figures use the full Open ASR Leaderboard test set (2,741 utterances). Figures in the
  development notes are 2,000-utterance seeded samples from the first of the 38 shards of `distil-whisper/earnings22`
  and are labelled as such.
- **Run-to-run noise**: about 1% in speed and about 0.03 in test-clean WER for the same engine; differences smaller
  than that were judged on repeated, interleaved runs.
- **Energy** is wall-plug power sampled every second, net of the machine's idle draw.

## Status and limitations

- Offline batch transcription. Streaming is not addressed.
- Utterances up to 60 s per segment (the engine's input profile); longer audio needs segmenting.
- Measured on the DGX Spark (sm_121), a GH200 and an H100 (sm_90) and a B200 (sm_100).
  The engine build is per machine.
- The pipeline is not bit-for-bit deterministic run to run (differences at the 0.03 WER level on test-clean).

## Licence and acknowledgements

Apache-2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)). Built on NVIDIA NeMo, TensorRT, TensorRT Model Optimizer
and CUTLASS. Models by NVIDIA (Parakeet) and Moondream (parakeet-ultra), all CC-BY-4.0, downloaded from their owners.
Evaluation data: LibriSpeech, Earnings-22 and FLEURS.
