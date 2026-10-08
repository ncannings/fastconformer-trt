# 9. Live streaming: concurrent real-time streams per GPU

The rest of this repository measures batch throughput: how fast recorded audio is transcribed. For live use (contact
centres, captioning, voice agents) the question is different: **how many simultaneous real-time audio streams can one
GPU serve, with every stream kept up to date and its final words out quickly?** This page answers it for NVIDIA's
cache-aware streaming model, `nvidia/nemotron-3.5-asr-streaming-0.6b`, first for stock NVIDIA NeMo, then with our
encoder engine and decoder, under measurement rules that were written down and frozen before the final runs.

| Concurrent live streams, C120 (rule in section 3) | Chunk | Stock NeMo + gc.freeze | Ours, FP16 engine | Ours, FP8 engine |
|---|---|---|---|---|
| NVIDIA DGX Spark (GB10) | 80 ms | 58 | **194** (3.3x) | **239** (4.1x) |
| NVIDIA DGX Spark (GB10) | 320 ms | 181 | **658** (3.6x) | **743** (4.1x) |
| NVIDIA DGX Spark (GB10) | 1.12 s | 326 | **1,086** (3.3x) | **1,397** (4.3x) |
| NVIDIA H100 80GB HBM3 (SXM) | 80 ms | 105 | **902** (8.6x) | **956** (9.1x) |
| NVIDIA H100 80GB HBM3 (SXM) | 1.12 s | 1,507 | **5,625** (3.7x) | **5,937** (3.9x) |

Ours = TensorRT encoder engine + our fused RNN-T decoder (+ the mel front end as one CUDA graph at 80 ms). Accuracy:
every arm is within 2% (relative) of stock NeMo's LibriSpeech test-clean WER at the same chunk size on the same GPU;
the FP16 engines used in the sweeps are at most 0.37% above stock (an engine rebuilt on a second H100 pod, +0.69%) and FP8 up to 1.74% above (sections 6 and 7).
Single runs; Earnings-22 calls as the live load; limitations in section 10. Summaries: [results/live/](../results/live/).

NVIDIA's model card quotes 2,400 streams at 1.12 s and 240 at 80 ms on one H100, by its own method (median
final-token latency). Those figures are not comparable with the table above; section 8 compares like with like.

## 1. What live capacity means

A live server receives audio from N callers in real time. The model works in chunks: with 1.12 s chunks a stream has
a new chunk ready every 1.12 s, with 80 ms chunks every 80 ms. The server keeps up if every chunk is processed soon
after it is ready, for every stream, for as long as the streams last. Two things therefore matter:

- **Falling behind.** If a step takes longer than a chunk period while chunks keep arriving, the queue grows and
  transcripts drift away from real time. We count a chunk as late when it waits more than one chunk duration.
- **Latency.** The time from the end of a stream's audio to its final transcript (final-token latency). The chunk
  size itself adds an algorithmic delay (you cannot transcribe a chunk before it has arrived); we report latency
  **excluding** that delay so that it measures the server, and we bound its p95, not just its median.

Capacity is the largest N at which both hold. It depends on the chunk size: small chunks give low latency but many
more, smaller steps, so per-step overheads dominate; large chunks are more efficient per second of audio.

## 2. The model and the software

- **Model:** `nvidia/nemotron-3.5-asr-streaming-0.6b` (0.6B parameters): a cache-aware FastConformer encoder
  (24 layers, d = 1,024) with an RNN-T decoder and a language-ID prompt, punctuation and capitals, 40 locales.
  Chunk sizes are set by `att_context_size = [56, r]`: 80 ms (r = 0), 320 ms (r = 3) and 1.12 s (r = 13) are
  measured here. Every stream here is English (`en-US`). The weights are NVIDIA's, under the OpenMDW-1.1 licence;
  this repository contains no weights and no engines.
- **Runtime:** NeMo 3.1.0 from `nvcr.io/nvidia/nemo:26.06` with the ASR extras installed and nothing in NeMo changed
  (`live/Dockerfile`), TensorRT 10.16, TensorRT Model Optimizer 0.44 for FP8.
- **Harness** (`live/harness.py`): one server loop serves N simulated callers. The stock step is NeMo's own
  `conformer_stream_step` split at its seams (encoder step, language prompt, RNN-T decoding), so that the encoder
  step and the decoder can be swapped. Per-stream encoder caches live in a slot-indexed slab on the GPU; streams join
  and leave at any step. Mel features are computed per chunk on the GPU from the audio received so far, as a live
  server must. Batching is continuous: a step starts as soon as any stream has a complete chunk and takes every
  complete chunk. On the same audio, the harness's stock transcripts were identical to NeMo's own streaming script
  (8 of 8 Earnings-22 streams at 320 ms, `live/equiv.py`).

## 3. The measurement rules

The rules were frozen on 6 October 2026, before any 80 ms or H100 number existed, and amended once (Amendment 1,
below) after a check showed an artefact. Every arm, stock included, is measured under the same rules.

| Rule | Why |
|---|---|
| **Load:** N streams cut from Earnings-22 earnings calls (the Open ASR Leaderboard test set; the harness groups its segments into 6 calls, 5.4 hours), each starting at a seeded random point, arrivals staggered uniformly over 10 s, audio fed in real time. | Long-form, conversational, multi-speaker audio is what live servers carry. Staggered arrivals are what a server sees; all streams starting on the same tick (section 8) flatter the server, because every step is then full. |
| **Late chunk:** a chunk whose step starts more than one chunk duration after the chunk became ready. | A server that is not keeping up shows chunks waiting longer and longer. See Amendment 1. |
| **Trial passes** if it is not aborted (backlog over 3 s), at most 0.1% of its chunks are late and the p95 final-token latency, excluding the chunk delay, is at most 1,000 ms. | 0.1% tolerates isolated hiccups without letting a server that is behind pass. A p95 bound, not a median, because a live user notices the slow tail. 1,000 ms was chosen as a generous bound; it is a choice (section 10). |
| **C120 (headline):** the largest N at which every trial at that N passed, including at least one 120 s trial. | Short trials hide slow drift: in our runs several N passed 60 s and failed 120 s. |
| **Search:** N grows by 1.25x per 60 s trial until one fails, bisects to 5%, then 120 s confirmation trials step N down 10% at a time (at most 5 steps). Also reported: A (largest N passing a 60 s trial with zero late chunks) and B (the same for 120 s). | A and B show how close the strict zero-late figure is. The frozen text said to double N; every final sweep used the finer 1.25x step, for every arm. |
| **Fresh process per trial**, model loaded and warmed up each time (Amendment 1). | See below. |
| **Accuracy gate:** LibriSpeech test-clean, full set (2,620 utterances), streamed through the same harness at the same chunk size (every utterance a stream, 64 live at a time); a custom arm must be within 2% relative of stock on the same GPU. | Speed at a different accuracy is not the same product. |
| **Quiet machine:** one arm on the GPU at a time, nothing else on it (each sweep's start and end records list no other GPU process). | Shared GPUs give unrepeatable timings. |

### Corrections made on the way (each applied to every arm, stock included)

1. **Decoder graph re-capture stalls.** NeMo's label-looping RNN-T decoder captures its CUDA graphs for the largest
   batch seen so far; under continuous batching, every new maximum triggered a re-capture of 100 to 400 ms, which
   stalls every live stream at once. Fix: a server warm-up that runs the decoder once at the full slot count before
   audio arrives (`Server.prewarm`). Every reported stock trial records zero re-captures.
2. **Python garbage-collection stalls.** Stock NeMo creates per-stream hypothesis objects every step; full
   (generation 2) collections of about 0.4 s coincided with stock's remaining stalls. `gc.collect()` then
   `gc.freeze()` after model load and warm-up removes them without changing NeMo. On 6 October (same-process
   trials) it raised stock's C120 at 320 ms on the Spark from 78 to 172. **Stock + gc.freeze is the stock baseline
   throughout**; it is the strongest stock configuration we found, and it changes nothing numerically.
3. **End-of-stream late-chunk artefact.** The original rule counted a chunk late if its step started after the
   stream's next chunk was complete. Every stream in a trial has the same length, so every stream ends on the same
   short final remainder; the second-to-last chunk was then flagged whenever the remainder became ready first. A
   check at 1.12 s showed that below capacity every chunk the original rule counted late was a second-to-last chunk
   and none had waited more than one chunk duration (stock at 320 streams: 21 of 21; our FP8 arm with an earlier
   decoder at 800: 44 of 44). Amendment 1 (6 October, approved before the final runs) redefined late as waiting more
   than one chunk duration. Every trial
   file carries both counts.
4. **Long-process slowdown.** A trial run late in a long sweep process was slower than the same trial in a fresh
   process (state left by earlier trials, including aborted ones at very large N), which understated every arm.
   Amendment 1 runs every trial in its own process.
5. **Start-up warm-up memory.** On the H100 on 7 October our arms ran out of GPU memory at 2,598 to 3,500 streams, in
   the server's warm-up rather than while serving: the warm-up built a second full set of caches and left freed
   blocks of every size behind. The warm-up now runs largest size first on persistent buffers. Peak reserved memory
   at 2,048 slots on the Spark fell from 53.7 GB to 37.2 GB with this fix and to 24.2 GB with the one-copy cache step
   (item 4 of section 5), with identical transcripts ([results/live/memory.json](../results/live/memory.json)).
6. **The stock path's memory, fixed for stock too.** Stock's encoder step gathered all of a step's caches into a
   full-batch copy and returned another full-batch copy. With `HARNESS_STOCK_SLAB_STEP=1` the harness calls NeMo's
   own encoder step on pieces of at most 512 streams straight from the float32 cache slab (same NeMo modules,
   float32 caches as shipped). Peak per step at 640 rows: 12.5 GB as one call, 6.8 GB in pieces. Up to 512 rows a
   piece is bit-identical to one call; above that NeMo's own output already depends on the batch size (cuDNN in the
   subsampling convolution), so larger calls are not bit-identical to one full-batch call. All final stock runs use
   it. On the H100 at 1.12 s stock's C120 went from 1,294 (6 October, without this fix and without the encoder
   warm-up) to 1,507 (7 October, with both).

## 4. The stock baseline

**Stock** is NeMo 3.1.0 as shipped: its own encoder step in float32 with TF32 matmuls (the settings of NeMo's
streaming script), float32 caches, its CUDA-graph label-looping greedy RNN-T decoder and the mel front end per
chunk. On top: gc.freeze, the server warm-up, the piece-wise cache step, pinned index uploads and a fresh process per
trial. None of these changes NeMo's numerics, apart from piece boundaries above 512 rows (item 6 above). Our arms get
the same warm-up, cache-step, upload and fresh-process treatment and run with Python's default garbage collection.

Where stock runs into NeMo's limits:

- **H100, decoder pre-warm.** On the H100, warming NeMo's decoder at 384 slots or more crashed its CUDA graphs
  ("illegal memory access", reproduced with stock and TensorRT encoders). The H100 stock runs at 1.12 s therefore
  ran with the decoder pre-warm off (the encoder warm-up stayed on); those trials recorded no decoder re-captures.
- **H100, 80 ms.** In the first stock sweep (7 October) stock passed 60 s trials up to 245 streams, but every 120 s
  confirmation failed, from 245 down to 160 (0.29% late at 160), and the search ran out of time before 144. A second
  sweep on 8 October, starting from 100 streams, confirmed C120 = 105 (section 7). In the same runs the decoder part of the step
  (NeMo's decoder and the language prompt) took 7 to 20 ms on average in the 60 s trials and 18 to 27 ms in the 120 s
  trials, higher in the longer trials even at similar or smaller batch sizes, while GPU utilisation stayed at 35 to
  44%: the decoder's cost grows as the streams run on and the GPU is mostly idle, which points to host-side work in
  NeMo's decoder. The 8 October sweep shows the same pattern (8.8 to 21.5 ms at 60 s, 15.0 to 28.5 ms at 120 s, GPU
  utilisation 29 to 35%). We have not traced the exact mechanism.
- **Spark.** The Spark runs used a container limited to 2.5 CPUs for every arm (`live/run_live.sh`). Stock does more
  host work per step than our arms; whether it gains from more CPUs was not tested.

## 5. What we changed

1. **TensorRT cache-aware encoder engine.** NeMo's `encoder.cache_aware_stream_step` exported to ONNX as two graphs
   per chunk size (a stream's first chunk and every later chunk), with a dynamic stream count and the caches as
   inputs and outputs, built with TensorRT in FP16, or in FP8 after ModelOpt post-training quantisation of the 216
   linear layers (convolutions, subsampling and LayerNorm inputs not quantised), calibrated by streaming 64
   LibriSpeech dev-clean utterances with live caches (never a test set). Caches are float16 at the engine boundary.
   Engine calls are capped at 512 rows (`TRT_MAX_CALL_B=512`). Code: `live/trt/`.
2. **Our fused RNN-T decoder** (`live/fused_decoder.py`): NeMo's greedy label-looping semantics (blank advances the
   frame, a label updates the prediction network, at most 10 symbols per frame) reimplemented as one flat masked
   iteration per step, a fused Triton joint + argmax kernel that never writes the logits and a fused bf16 LSTM cell.
   Its CUDA graphs have fixed shapes (one per batch bucket up to the slot count) and are captured once at start-up;
   nothing is captured while serving. Rows still emitting after the fixed iteration count continue as a compacted
   call of only those rows, so the result is exact for any burst length. With its float32 PyTorch path the decoder is
   token-for-token identical to NeMo's at 80 ms, 320 ms and 1.12 s (8 of 8 streams, decoder isolated,
   `live/dec_equiv.py`); with the Triton kernels used in every reported run, 1 or 2 of 8 streams differed at
   near-ties. The WER gate covers this.
3. **Mel front end as one CUDA graph** at 80 ms, where its launch overhead matters (`--mel-graph`).
4. **One cache copy per stream.** The engine reads and writes the cache slab piece by piece instead of gathering a
   full-batch copy and receiving another back (`HARNESS_SLAB_STEP=1`), with float16 caches and pinned asynchronous
   index uploads. This mattered for memory at thousands of streams, not for speed on its own.

The GPU stays busier: at C120 in the 120 s trials our arms averaged 89 to 95% GPU utilisation against 66% for stock
on the H100 at 1.12 s (`gpu_util_mean_pct` in the summaries).

## 6. Results on the DGX Spark

Quiet window, 7 to 8 October 2026 (22:36 to 00:31 BST), one sweep per arm and chunk size, all the fixes above, seed
20261005 (the load for each N is seeded by 20261005 + N). Files: `results/live/raw/spark_20261007/`, summarised in
[results/live/c120_spark.json](../results/live/c120_spark.json).

| Chunk | Arm | C120 | vs stock | Late chunks in the 120 s trial at C120 | Final-token p50 / p95 (ms) | A | B |
|---|---|---|---|---|---|---|---|
| 80 ms | stock + gc.freeze | 58 | | 0 of 87,058 | 46 / 82 | 58 | 58 |
| 80 ms | FP16 + fused + mel graph | **194** | **3.3x** | 16 of 291,194 | 25 / 66 | 179 | none |
| 80 ms | FP8 + fused + mel graph | **239** | **4.1x** | 0 of 358,739 | 18 / 72 | none | 239 |
| 320 ms | stock + gc.freeze | 181 | | 0 of 68,056 | 64 / 330 | 192 | 181 |
| 320 ms | FP16 + fused | **658** | **3.6x** | 24 of 247,408 | 39 / 507 | 658 | none |
| 320 ms | FP8 + fused | **743** | **4.1x** | 0 of 279,368 | 27 / 294 | 743 | 743 |
| 1.12 s | stock + gc.freeze | 326 | | 0 of 35,208 | 146 / 755 | 363 | 326 |
| 1.12 s | FP16 + fused | **1,086** | **3.3x** | 0 of 117,288 | 56 / 488 | 1,086 | 1,086 |
| 1.12 s | FP8 + fused | **1,397** | **4.3x** | 0 of 150,876 | 48 / 544 | 1,397 | 1,397 |

"none" for A or B means no trial of that length had zero late chunks, although the arm passed the 0.1% rule.

Accuracy, LibriSpeech test-clean through the harness (gate: within 2% relative of stock),
[results/live/wer_gates.json](../results/live/wer_gates.json):

| Chunk | Stock NeMo | FP16 + fused | FP8 + fused | Limit |
|---|---|---|---|---|
| 80 ms | 3.796 | 3.807 (+0.30%, with mel graph) | 3.822 (+0.70%) | 3.872 |
| 320 ms | 3.321 | 3.294 (-0.80%) | 3.300 (-0.62%) | 3.387 |
| 1.12 s | 3.027 | 3.023 (-0.12%) | 3.079 (+1.74%) | 3.087 |

The FP8 80 ms and 1.12 s gates and the stock 80 ms and 1.12 s references were run on 6 and 7 October with the same
engines as the capacity runs; the rest on 7 October. The FP8 80 ms gate ran without the mel graph; on 6 October the
mel graph left the FP8 WER unchanged (3.828 with and without, previous decoder).

## 7. Results on the H100

RunPod, one NVIDIA H100 80GB HBM3 (SXM, 700 W limit, driver 580.126.09), the same NeMo 26.06 software (x86) run
natively on the pod, engines built on the pod, every trial in a fresh process under the same rules. Two pods on
7 October, v4 (our arms, `results/live/raw/h100_20261007_v4/`) and v5 (stock with the memory fix,
`results/live/raw/h100_20261007_v5/`), a third on 8 October, v6 (stock at 80 ms,
`results/live/raw/h100_20261008_v6/`), and a fourth the same day, v7 (FP8 at 80 ms, the same settings as the
FP16 80 ms arm, `results/live/raw/h100_20261008_v7/`), summarised in [results/live/c120_h100.json](../results/live/c120_h100.json).

| Chunk | Arm | C120 | vs stock | Late chunks in the 120 s trial at C120 | Final-token p50 / p95 (ms) | A | B |
|---|---|---|---|---|---|---|---|
| 1.12 s | stock + gc.freeze | 1,507 | | 0 of 162,756 | 117 / 770 | 1,675 | 1,507 |
| 1.12 s | FP16 + fused | **5,625** | **3.7x** | 0 of 607,500 | 226 / 768 | 5,625 | 5,625 |
| 1.12 s | FP8 + fused | **5,937** | **3.9x** | 0 of 641,196 | 156 / 577 | 5,937 | 5,937 |
| 80 ms | stock + gc.freeze (8 Oct) | 105 | | 103 of 157,605 | 61 / 105 | 100 | none |
| 80 ms | FP16 + fused + mel graph | **902** | **8.6x** | 41 of 1,353,902 | 20 / 81 | 902 | none |
| 80 ms | FP8 + fused + mel graph (8 Oct) | **956** | **9.1x** | 0 of 1,434,956 | 19 / 56 | 956 | 956 |

Stock at 80 ms needed two sweeps. The first (v5, 7 October, from 112 streams) found no C120: 60 s trials passed up
to 245, but every 120 s confirmation failed (245, 220, 198, 178 and 160) before the time ran out. The second (v6,
8 October, a third pod, decoder pre-warm on, from 100 streams, `results/live/raw/h100_20261008_v6/`) passed 60 s
trials up to 180 and confirmed 105 over 120 s after 120 s failures at 180, 162, 145, 130 and 117. The 8.6x and 9.1x at 80 ms
therefore compare runs on different pods of the same type; both are large mainly because stock's 120 s trials degrade
(section 4).

Memory is not the limit at these N on the H100: our arms served 12,000 streams at 1.12 s for 20 s with at most 73 GB
reserved while serving, though the start-up warm-up peaked at 79 to 80 GB of the card's 80 GB (FP16 and FP8 alike);
stock with the memory fix served 2,400 with at most 37.5 GB reserved (`results/live/memory.json`).

Accuracy on the H100, test-clean through the harness (gate: within 2% of the H100 stock run):

| Chunk | Stock NeMo | FP16 + fused (engine used in the sweep) | Same arm, engine rebuilt on the second pod | FP8 + fused | Limit |
|---|---|---|---|---|---|
| 80 ms | 3.794 | 3.790 (-0.10%, with mel graph) | 3.798 (+0.10%) | 3.821 (+0.70%, with mel graph, 8 Oct) | 3.870 |
| 1.12 s | 3.025 | 3.036 (+0.37%) | 3.046 (+0.69%) | 3.062 (+1.25%) | 3.085 |

TensorRT engine builds are not bit-for-bit repeatable (tactic selection), which is why the two FP16 engines built on
two pods score slightly differently. Both pass.

## 8. NVIDIA's published figures and their method

NVIDIA's model card for `nvidia/nemotron-3.5-asr-streaming-0.6b` (Hugging Face, "Throughput & Efficiency") states,
for a single H100: 240 concurrent streams at 80 ms and 2,400 at 1,120 ms (against 14 and 400 for buffered
Parakeet RNNT 1.1B), where "throughput is the number of real-time streams sustainable in parallel; latency is the
median final-token latency at a given level of concurrency". These are NVIDIA's figures, measured by NVIDIA; the card
does not describe the load in more detail.

To check our harness against them we ran NVIDIA's definition as we read it: all N streams start together, 60 s
trials, median final-token latency excluding the chunk delay and "keeps up" meaning no chunk waited more than one
chunk duration. These runs (6 and 7 October) predate some of the fixes above: our arm was the FP8 engine with
earlier decoders (not the final configuration), and NeMo's decoder pre-warm was off in the stock runs (section 4).
[results/live/nvidia_method_h100.json](../results/live/nvidia_method_h100.json).

| H100, all streams start together | N | Stock NeMo + gc.freeze | Ours (FP8 engine) |
|---|---|---|---|
| 80 ms | 160 | 57 ms, keeps up | 21 ms, keeps up (lean decoder) |
| 80 ms | 240 | **94 ms, keeps up** | 26 ms, keeps up (lean decoder) |
| 80 ms | 320 | behind (35% of chunks late) | 30 ms, keeps up (lean decoder) |
| 80 ms | 480 | | 40 ms, keeps up (lean decoder) |
| 80 ms | 960 | | fused decoder: 88 ms, keeps up in one run; 48% late in a second run |
| 80 ms | 1,200 | | behind (both runs) |
| 1.12 s | 1,600 | 839 ms, keeps up | 260 ms, keeps up (lean decoder) |
| 1.12 s | 2,000 | 2,476 ms, 2,000 late chunks | 325 ms, keeps up (lean decoder) |
| 1.12 s | 2,400 | **1,579 ms, keeps up** | 380 ms (lean decoder); 366 and 371 ms (fused decoder, two runs); all keep up |
| 1.12 s | 3,200 | | 479 and 485 ms, keeps up (fused decoder, two runs) |

What this shows:

1. **Our harness reproduces NVIDIA's capacity figures for stock NeMo** under their kind of method: stock keeps up at
   240 streams at 80 ms (and falls behind at 320) and at 2,400 at 1.12 s. Our stock latencies at those points are
   higher than NVIDIA's chart shows (read from the card's latency chart: roughly 60 to 70 ms at 240 streams at
   80 ms and about 340 ms at 1,600 streams at 1.12 s, against our 94 ms and 839 ms), so NVIDIA's own serving path is
   leaner than NeMo's Python loop in our harness.
2. **The same stock NeMo gets 1,507 streams at 1.12 s under our rule** (section 7). Staggered arrivals, a p95
   latency bound and 120 s trials are stricter than aligned starts with a median.
3. Under aligned starts our arm keeps up at 3,200 streams at 1.12 s, the largest N tried with the fused decoder.

## 9. Videos

Terminal recordings of the live server, made with `live/demo_live.py`. Each act is one process serving N Earnings-22
calls in real time, with the clock, step time, late chunks and four transcripts on screen. Pauses between acts (model
loading and warm-up) are shortened in the recording; the acts play at recorded speed. A demo act is not a capacity
trial: the dashboard adds host work. Per-act counts: [results/live/videos.json](../results/live/videos.json).

- **DGX Spark** ([GIF](media/live_spark.gif), [MP4](media/live_spark.mp4), 8 October): stock + gc.freeze against
  the FP16 arm at 85% of the FP16 arm's C120 (165, 560 and 923 streams at 80 ms, 320 ms and 1.12 s). Ours: 0 late
  chunks in all three acts. Stock falls behind in all three (it is far above stock's own C120).
- **H100** ([GIF](media/live_h100.gif), [MP4](media/live_h100.mp4), 7 October): at 80 ms, 760 streams each (stock
  falls behind; ours 0 late chunks); at 1.12 s, stock at 2,400 streams (NVIDIA's published figure; it kept up for
  the 45 s act, 0 late chunks) and ours at 5,625 (our FP16 C120; 0 late chunks).

## 10. Limitations

- **FP8's accuracy margin at 1.12 s is thin**: +1.74% on the Spark (3.079 against a limit of 3.087) and +1.25% on
  the H100. Other calibration recipes (percentile ranges, more and broader calibration data, the two most sensitive
  layers in FP16) did not give a real margin. The FP16 engine is the arm with a clear margin and is the one we would
  deploy by default; FP8 adds 6 to 29% more streams.
- **Stock at 80 ms on the H100** degrades over 120 s trials (section 4): it passes 60 s trials up to 180 to 245
  streams but confirms only 105 over 120 s. The 8.6x and 9.1x ratios at 80 ms rest on that behaviour of NeMo's decoder in
  our harness; under NVIDIA's aligned-start method stock keeps up at 240 (section 8).
- **Single runs.** Each C120 comes from one sweep. Nearby N pass and fail non-monotonically near the edge, and the
  two aligned-start 80 ms runs at 960 streams disagreed. Treat differences of a few percent as noise.
- **The load is Earnings-22 English.** Streams are cut from 6 calls (5.4 hours) at seeded random offsets, so many
  streams carry the same audio from different points. Other audio (silence-heavy calls, other languages, the
  language-ID prompt in auto mode) was not measured. Accuracy was gated on LibriSpeech test-clean only.
- **The latency bound is a choice.** C120 uses p95 final-token latency at most 1,000 ms, excluding the chunk delay.
  A tighter bound lowers every arm's C120; the per-trial files contain the percentiles to apply another bound.
- **Different GPUs, different software paths.** The Spark runs were in a 2.5-CPU container; the H100 runs were native
  on the pod. Comparisons are only made between arms on the same machine.
- **What is not measured:** energy per stream, the 160 and 560 ms chunk sizes, GH200 and B200, network transport,
  voice activity detection with endpointing and multi-GPU serving.

## 11. Reproducing

Code: [live/](../live/), with [live/README.md](../live/README.md) for the container, data, engines and sweep commands.
The tables on this page are rebuilt from the raw per-trial lines by `python results/live/summarise.py`, which also
checks each C120 against the sweep's own result line.
