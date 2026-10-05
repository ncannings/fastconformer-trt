# 1. The stock-weight engine for parakeet-tdt-0.6b-v3

This document describes how NVIDIA's parakeet-tdt-0.6b-v3 (FastConformer encoder, TDT decoder) was made about 4.5 times faster than stock NeMo on a single DGX Spark (GB10), with the original weights and no retraining. Each step is given with the hypothesis behind it, what was built and the measured gain, followed by the ideas that did not pay and the controlled stock-versus-final comparison on three models. The goal throughout was simple: as fast as possible at equal accuracy.

All speeds are real-time factors (RTF: audio seconds transcribed per wall second, whole pipeline) and all accuracies are WER in percent. The measurement method is described in [05-measurement-method.md](05-measurement-method.md).

## Result at a glance

Controlled run, driver-default clocks, 300 s heat soak, wall-plug meter, LibriSpeech test-clean (all 2,620 utterances), batch 32 (`engine/stock_vs_final.sh`):

| Model | Stock WER | Final WER | Stock RTF | Final RTF | Speed-up | J per audio hour, gross | J per audio hour, net of idle |
|---|---|---|---|---|---|---|---|
| parakeet-tdt-0.6b-v3 | 1.931 | 1.918 | 1,008 | 4,580 | **4.54x** | 670 to 181 | 414 to 113 |
| parakeet-tdt-1.1b | 1.386 | 1.392 | 684 | 2,991 | **4.37x** | 993 to 279 | 594 to 168 |
| parakeet-ctc-1.1b | 1.844 | 1.833 | 727 | 3,094 | **4.26x** | 948 to 267 | 573 to 161 |

The per-run JSON summaries are in `results/headline/`. The post-trained parakeet-ultra variant of the 0.6B model reaches 4,903x on the same engine ([02-ultra-25-languages.md](02-ultra-25-languages.md)).

## 1. Baseline and measurement

### Stock NeMo

The reference is the unmodified model in NeMo, bf16, batch 32 with utterances sorted by length, CUDA-graph batched greedy decoding. Several stock figures appear in this document, and they do not all come from the same harness:

| Baseline | What it is | test-clean RTF |
|---|---|---|
| Early research harness | fp32 weights with autocast, as first set up | 693 (one early measurement on dev-clean gave 627 and did not reproduce) |
| First bar measurement | NeMo bf16, batch 32, 3 repeats, median | 760 (quoted as 769 in a later note; WER 1.92%) |
| Best PyTorch | bf16 encoder, SDPA attention, FP8 encoder linears via Transformer Engine, `torch.compile` | 870 to 881 |
| Stock arm of the final protocol (`engine/stock_rtf.py`) | NeMo bf16 autocast, CUDA-graph batched greedy decoder | 977 at a 2,200 MHz clock cap; 1,008 at driver-default clocks |

The headline speed-ups always use the last row, measured in the same session as the final engine.

### Where the time went in PyTorch

A profile of the best PyTorch configuration (12 dev-clean batches of 32) showed a GPU that was busy but inefficient:

- GPU busy 85 to 87% of wall time (union of kernel intervals);
- about 1,640 to 1,905 kernels per batch;
- achieved encoder arithmetic of 11 to 14 TFLOP/s;
- bound by memory traffic (273 GB/s on this machine): about 8,000 copies and casts, unfused elementwise operations, fp32 attention with a materialised relative-position matrix, 1x1 convolutions running as cuDNN convolutions and FP8 weight re-casts on every call (5,184 kernels).

CUDA-graph capture of the PyTorch encoder failed with an illegal memory access, with and without FP8, and was not pursued. The conclusion was to leave PyTorch for TensorRT and to remove memory traffic rather than arithmetic.

### First TensorRT engines (NeMo's own ONNX export)

Encoder exported with NeMo's ONNX export, built with `trtexec` (dynamic batch 1 to 32, up to 3,600 mel frames), decoder still in NeMo. Measured at the locked 1,176 MHz clock described in the next section.

| Encoder engine | dev-clean | Earnings-22 | RTF dev-clean |
|---|---|---|---|
| PyTorch, best | 1.70 | 10.86 | 876 |
| TensorRT bf16 | 1.78 | 10.82 | 1,109 |
| TensorRT fp16 | 1.67 | 10.89 | 1,118 |
| **TensorRT FP8** (ModelOpt PTQ on the linear layers, 128 dev-clean utterances calibration, convolutions fp16) | **1.69** | **10.85** | **1,339** |

On held-out test-clean the FP8 engine scored 1.96% WER (unmodified model 1.94 to 1.96%) at 1,332x, 1.53x the best PyTorch configuration.

## 2. The GPU clock lesson

For the first part of this work the GPU clock on the test machine was locked at 1,176 MHz by a system service. Under load the GPU held about 1,050 MHz at 34 W. Every number measured in that state was internally consistent, so nothing looked wrong.

It mattered because it produced a false ceiling. At the locked clock, cuBLAS FP8 GEMMs at the encoder's shapes (M = 6,400 frames) reached 75 to 86 TFLOP/s, TensorRT's own FP8 GEMMs about 89 TFLOP/s, bf16 30 and NVFP4 130. From that, the encoder floor was estimated at about 140 ms per 32 x 16 s batch and the pipeline ceiling at 2,500 to 2,900x. That estimate was wrong: it was a ceiling of the clock, not of the chip.

After the lock was replaced by a 2,200 MHz cap (`nvidia-smi -lgc 300,2200`) and an extra fan was fitted:

- sustained FP8 GEMM throughput rose from 84 to 142 TFLOP/s, at about 1,970 to 2,170 MHz, 81 W peak GPU power and 72 °C peak;
- the same engine went from 1,884x to 2,566 to 2,581x on test-clean.

Lesson: check `nvidia-smi -q -d CLOCK` and measure a sustained GEMM before trusting any roofline. `engine/gemm_hold.py` runs a 25 s FP8 GEMM load for exactly this purpose. Fully unlocked clocks (driver default) gain little over the 2,200 MHz cap under this workload, because the GPU becomes power-limited first; see [05-measurement-method.md](05-measurement-method.md).

## 3. The lean, export-friendly encoder

**Hypothesis:** NeMo's encoder forward is written for training and streaming, and much of its memory traffic exists only because of how it is expressed. A re-implementation with the same weights, written for export, would let TensorRT fuse far more.

`engine/lean_encoder.py` re-implements the FastConformer forward with identical weights:

- no fp32 casts (NeMo forces fp32 attention);
- the relative-position term as a single gather with an index matrix built once per batch, instead of NeMo's pad, reshape and slice in every layer;
- the key-padding bias and frame mask built once per batch;
- pointwise convolutions as `nn.Linear` on [B, T, C], so FP8 quantisation sees them and no transposes surround them;
- inference batch norm folded into the depthwise convolution;
- subsampling masked once at the input instead of before each of its 7 conv layers (`LEAN_PREMASK_ONCE=1`);
- encoded lengths capped at the output length.

`engine/lean_export.py` checks the lean forward against NeMo's encoder, then exports ONNX with ModelOpt FP8 PTQ on all linears. The engine was built for up to 60 s per utterance and the runner decoded 128 utterances per call. `engine/lean_equiv_probe.py` reports, per option, the maximum difference from NeMo's encoder output for any model.

| Configuration (locked clock) | test-clean | dev-clean | Earnings-22 | RTF test-clean |
|---|---|---|---|---|
| Best PyTorch | 1.96 | 1.70 | 10.86 | 870 to 881 |
| NeMo export, FP8 TensorRT | 1.95 | 1.69 | 10.85 | 1,397 |
| **Lean, FP8 TensorRT** | **1.93** | **1.67** | **10.78** | **1,748 to 1,752** |

The lean FP8 engine is 1.99x the best PyTorch configuration. Per dev-clean pass: preprocessing 0.50 s, encoder 8.97 s, decoder 1.44 s.

Two further lean-encoder options followed:

- **Depthwise convolution as 9 shifted multiply-adds** on [B, T, C] (`LEAN_DWSHIFT=1`), so TensorRT can fuse it with the GLU, the mask and the activation: 211 to 203.5 ms per batch, dev-clean RTF 1,839.
- **Heads-first attention** (`LEAN_HB=1`): q, k and v laid out [H, B, T, dk], so the relative-position term is one [H, B*T, dk] @ [H, dk, 2T-1] matmul per head against the shared position projection. TensorRT had been replicating that projection across the batch (about 10 ms per batch). Encoder 144.2 to 130.2 ms per batch; adding NeMo's rel_shift form (`LEAN_RELSHIFT=1`) instead of the index gather, 126.0 ms. End to end 3,053x on test-clean, WER unchanged.

`engine/build_variant.sh` runs the export, all plugin surgeries, the TensorRT build and a `trtexec` profile for any combination of these switches.

## 4. FP8 post-training quantisation

All encoder linears (feed-forward, attention projections, pointwise convolutions) are FP8 with ModelOpt PTQ. Calibration uses 128 dev-clean utterances by default; the depthwise and subsampling convolutions stay fp16. No calibration or selection uses a test set.

On this model FP8 PTQ is accuracy-neutral throughout: test-clean stays at 1.90 to 1.94% against 1.93 to 1.96% for the unmodified model. The one place it was not neutral was a shared scale, described under QKVHeads below.

Profile of the lean FP8 engine per 32 x 16 s batch (locked clock, 211 ms in total), which set the order of the plugin work:

| Component | ms |
|---|---|
| FF GEMMs | 58 |
| SiLU and cast | 18 |
| Attention | 14.5 |
| Subsampling convolutions | 13.4 |
| Pointwise GEMMs | 12.7 |
| Relative-position gather | 10 |
| Position and score matmuls | 9.8 |
| Layer norms | about 25 |
| Transposes and moves | about 15 |

## 5. TensorRT plugins

All plugins are TensorRT IPluginV3 implementations in `engine/plugins/`, built into one library (`libffn_fp8.so`, `engine/plugins/build.sh`) from CUTLASS sm_120-family kernels, Triton cubins and hand-written CUDA. Each plugin has a graph-surgery script that swaps it into the ModelOpt ONNX and refuses on any unexpected pattern. Timings below are `trtexec` per 32 x 16 s batch at the 2,200 MHz cap unless stated.

### FFNFp8: fused feed-forward

**Hypothesis:** TensorRT runs the FF block as GEMM, SiLU, cast, GEMM, writing and re-reading the 4x-wide intermediate. Putting SiLU and the FP8 quantise in the first GEMM's epilogue removes that traffic.

- `ffn_fp8.cu`: CUTLASS 4 FP8 kernels, GEMM1 with SiLU and FP8 quantise in the epilogue (EVT), GEMM2 with fp16 output.
- `ffn_surgery.py` swaps the plugin into all 48 FF blocks.
- Relative error 2e-4 against a torch FP8 reference.
- TensorRT never fuses SiLU into an FP8 GEMM epilogue, whatever the ONNX form (checked on isolated FF blocks).

| Engine | ms per batch | FF blocks |
|---|---|---|
| Lean FP8 | 159.9 | 51.5 (34.7 GEMMs + 16.8 SiLU and quantise) |
| + FFN plugin | 150.7 | 36.2 |

End to end: 2,566 to 2,581x became **2,750x** (test-clean 1.87 on a 400-utterance sample). Part of the saving was lost to a residual add that could no longer fuse into the next layer norm; that is recovered by residual folding below.

### SubConv02: fused subsampling front end

**Hypothesis:** the first subsampling convolution (1 to 256 channels, 3x3, stride 2) writes an 840 MB output per batch that the next layer immediately reads back.

- `sub_plugin.cu`, `sub_surgery.py`: conv.0 + ReLU + depthwise conv.2 (3x3, stride 2) computed from a shared-memory input tile, so the conv.0 output is never written. fp16, channels-last vector stores.
- 9.1 ms down to 1.96 ms, plus a 2 ms layout move that TensorRT still inserted. Encoder 150.7 to 144.2 ms.
- End to end with both plugins: **2,835x**, test-clean 1.81 / dev-clean 1.69 / Earnings-22 10.88.

**NHWC output** (`SUB_NHWC=1`): one thread per channel, a 4 x 8 position tile and the 7 x 7 patch in registers, so each position's 256 channels are one coalesced 512-byte write. This removes the layout move: plugin 1.74 ms against 1.94 + 1.97 ms before, encoder 89.9 to **87.2 ms**, WER identical.

### QKVHeads: fused, heads-first Q/K/V projection

**Hypothesis:** with heads-first attention, the cost is no longer the q/k/v GEMMs but the layout change and the bias adds around them.

A plain fused QKV linear in TensorRT (`LEAN_QKV=1`) gave nothing (126.2 against 126.0 ms): the three GEMMs merged (7.8 to 7.6 ms) but the permute to heads-first plus the bias adds cost about 10.5 ms per batch at memory bandwidth either way.

`qkv_heads.cu` (`qkv_surgery.py`) is an FP8 batched CUTLASS GEMM per q/k/v over the 8 head blocks, with the shared input at batch stride 0. The projection bias and `pos_bias_u` go in the epilogue, and the output is written directly as [H, B*T, dk]. The second query projection is folded away: bd = q_u.p + (pos_bias_v - pos_bias_u).p.

- 0.364 ms against 0.741 ms for GEMM + permute (the plain GEMM alone is 0.349 ms).
- On its own the gain disappeared, because TensorRT's attention kernel copied q and v into its own buffers (a new 7.9 ms move). It pays only together with RelPosAttn.
- **One FP8 weight scale shared by q, k and v cost accuracy** (400-utterance samples: test-clean 1.94 / dev-clean 1.78 / Earnings-22 10.93). With a per-part scale (each GEMM launch has its own alpha): 1.86 / 1.67 / 10.92, back at baseline.

### RelPosAttn: fused relative-position attention

**Hypothesis:** relative-position attention in TensorRT is five separate memory-bound steps; one kernel can do all of them.

`engine/relpos_attn.py` is a Triton kernel, compiled to a cubin by `engine/relpos_cubin.py` and run from `relpos_plugin.cpp`. One kernel does the q.k tile, the q.p band, the relative shift with an in-register `tl.gather`, the key mask, online softmax and the output in [B, T, H*dk]. Keys beyond each utterance's length are skipped.

- Relative error 3.4e-4 against the fp32 reference.
- 0.33 ms per layer at 32 x 200 (best of a sweep: BM = BN = 32, 4 warps, 2 stages).
- Replaces the position-score matmul (6.3 ms), shift (5.75), attention (10.4), copies (7.9) and output permute (2.9).

With QKVHeads, encoder 126.0 to **97.6 ms** per batch:

| Configuration | test-clean (full 2,620) | RTF test-clean |
|---|---|---|
| Lean FP8 + FFN + SubConv02 + heads-first + compiled preprocessing | 1.90 | 3,171 to 3,189 |
| **+ QKVHeads + RelPosAttn** | **1.94** | **3,871 to 3,873** |

### Residual folding into FFNFp8

**Hypothesis:** the FF output is written, then read back for the residual add. CUTLASS can do the add in GEMM2's epilogue.

`FFN_RESIDUAL=1` in `ffn_surgery.py`: GEMM2's epilogue computes residual + 0.5 * FF as a linear combination with the residual as the source operand.

- With TensorRT feeding an fp16 residual: encoder 97.6 to **89.9 ms**. Restricting to an fp32 residual gives 103.9 ms.
- WER on 2,000-utterance samples is unchanged:

| Variant | Earnings-22 | dev-clean |
|---|---|---|
| fp16 residual (fold) | 10.56 | 1.95 |
| fp32 residual (fold) | 10.52 | 1.95 |
| No fold | 10.58 | 1.95 |

Full test-clean: **1.90% at 4,075x**.

### SpLinearFp8 dense mode: residual folding into linear_out and pw2

**Hypothesis:** the same fold applies to the two other GEMMs that feed a residual add, the attention output projection (`linear_out`) and the conv module's second pointwise linear (`pw2`).

The SpLinearFp8 plugin (`splinear_plugin.cpp`, `sparse_fp8.cu`) was written for 2:4 sparse weights ([03-pruning-and-distillation.md](03-pruning-and-distillation.md)). Its dense mode (`SP_LINEAR=1 SP_DENSE=1`, `sparse_surgery.py`) runs a CUTLASS FP8 GEMM with C = residual on the unpruned stock weights.

- Encoder 82.5 ms against 87.2 (`trtexec`).
- End to end 4,587 / 4,566 against 4,464 / 4,471 in interleaved runs: **+2.4%**.
- test-clean 1.929 against 1.918 (about 6 words, identical across both runs). Adopted.

## 6. Decoder and runner

Once the encoder was fast, the decoder and the Python runner (`engine/trt_rtf.py`) became visible. The decoder is NeMo's TDT label-looping greedy decoder with CUDA graphs; the changes below keep its algorithm.

| Change | Runner flag | Measured effect |
|---|---|---|
| Decoder and joint weights in bf16. About 60% of decoder GPU time was fp32 joint GEMMs (640 to 8,198 at every step); NeMo disables autocast inside decoding, so the weights are cast instead. | `--dec-bf16` | Decoder 1.45 to 1.16 s per dev-clean pass; dev-clean RTF 1,839 to 1,882 (locked clock) |
| Producer/consumer pipeline over two streams | `--pipeline` | dev-clean RTF 1,882 to 1,916, about 2%: the decoder's kernels compete with the encoder rather than filling idle time |
| `torch.compile` of the mel featurizer | `--pre-compile` | 3,053 to 3,189x; preprocessing 0.49 s to 0.18 s per test-clean pass |
| Fused TDT joint + argmax (`engine/fast_joint.py`): one Triton kernel does the frame gather, add, ReLU, the 640 to 8,198 GEMM and a per-tile max/argmax, with logits rounded to bf16 as torch rounds them and ties to the lowest index; a second kernel reduces the tiles. Labels, durations and scores identical to NeMo on random states; 64/64 texts identical. | `--fast-joint` | Decoder 0.75 to 0.685 s per test-clean pass; 4,075 to 4,142x |
| Sub-batch trim (bug fix): each 32-utterance encoder sub-batch was sliced from the decoder group without trimming, so the encoder processed padding up to the longest utterance of the whole group | (always on) | Encoder stage 3.93 to 3.70 s per test-clean pass |
| Decoder groups of 256 utterances | `--dec-batch 256` | With the trim: 4,375x (group size table below) |
| Fast hypotheses (`engine/fast_hyps.py`): replaces NeMo's `batched_hyps_to_hypotheses`, which copies full preallocated buffers to pageable host memory and syncs once per utterance; here one sync per batch and only the used prefix is copied. Same outputs. | `--fast-hyps` | With NHWC SubConv02: 4,460 to 4,481x |
| FP8 joint weights, per-output-column scale dequantised in registers | `FAST_JOINT_FP8=1` | Decoder 0.56 to 0.51 s per pass; **4,508x**; WER unchanged (2,000 samples: Earnings-22 10.52 / dev-clean 1.95 against 10.52 / 1.96 in bf16) |
| Frame-budget batching: sub-batches packed to a padded-frame budget instead of a fixed 32 utterances (engine profile up to 128 x 6,000) | `--frame-budget 4800 --max-batch 128` | 4,576x against 4,463x, +2.5%, WER unchanged |

Decoder group size (test-clean 400-utterance sample, after the trim fix):

| Decoder group | RTF |
|---|---|
| 128 | 4,311 |
| 256 | 4,435 |
| 512 | 4,055 (too few groups to overlap) |

Why frame budgeting helps: per-frame encoder cost is 15.3 us at 1,600 frames per batch and 13.6 to 13.9 us from 3,200 frames up, and 40% of test-clean's audio sat in fixed-32 batches under 3,200 frames.

Robustness notes: decoder CUDA graphs are captured before the pipeline starts (a capture concurrent with `torch.compile` in the producer thread crashed at group size 512); the producer stops if a consumer dies instead of deadlocking on a full queue; each encoder sub-batch is kept at least 1 s long, because the engine profile's minimum is 100 frames and trimming had exposed sub-batches of very short Earnings-22 clips.

## 7. Speed progression

test-clean, same machine, stock weights throughout:

| Configuration | RTF |
|---|---|
| Early research harness | 693 |
| Best PyTorch | 870 to 881 |
| Lean FP8 TensorRT (clock locked at 1,176 MHz) | 1,746 |
| + shifted depthwise conv, bf16 decoder, pipeline | 1,884 |
| GPU clock cap 1,176 to 2,200 MHz | 2,566 |
| FFN plugin | 2,750 |
| Subsampling plugin | 2,835 |
| Heads-first attention | 3,053 |
| Compiled preprocessing | 3,189 |
| QKVHeads + RelPosAttn | 3,873 |
| Residual fold | 4,075 |
| Fused joint | 4,142 |
| Sub-batch trim + decoder group 256 | 4,375 |
| NHWC subsampling + fast hyps | 4,460 to 4,481 |
| FP8 joint weights | 4,508 |
| Frame-budget batching | 4,576 |
| Residual fold into linear_out and pw2 (+2.4% in interleaved A/B) | 4,587 / 4,566 against 4,464 / 4,471 |

The full engine recipe is the lean FP8 export with `LEAN_PREMASK_ONCE`, `LEAN_DWSHIFT`, `LEAN_RELSHIFT`, `LEAN_HB`, `LEAN_QKV_PLUGIN` and `LEAN_ATTN_PLUGIN`, plus the FFN with residual, SubConv02 (NHWC), QKVHeads, RelPosAttn and SpLinearFp8 dense plugins. The runner flags are `--batch 32 --dec-batch 256 --dec-bf16 --pipeline --pre-compile --fast-joint --fast-hyps`, with `FAST_JOINT_FP8=1` and optionally `--frame-budget 4800 --max-batch 128` on an engine built for batch 128.

### Where the time goes now

Per 32 x 16 s batch, encoder about 87 ms before the last residual fold:

- GEMMs about 60 ms, near the FP8 peak (142 TFLOP/s at 2,200 MHz);
- memory-bound work about 28 ms: layer norms 13.4, subsampling remainder 5.4, attention 3.2, GLU and depthwise conv 3.1.

Per test-clean pass: encoder 3.62 s, decoder 0.55 s, preprocessing 0.18 s. The stages barely overlap, because the decoder is latency- and weight-bandwidth-bound and is starved of SMs while the encoder runs. The pipeline is encoder-bound: about 13.6 us per frame against about 7.9 us of pure FP8 arithmetic at the power-limited 146 TFLOP/s. An independent review of the final profile, and the resulting ceiling for this model, are in [02-ultra-25-languages.md](02-ultra-25-languages.md).

## 8. Dead ends

Every row below was measured; none was adopted.

### Training-free frame merging (token merging)

**Hypothesis:** speech has near-duplicate frames, so a ToMe-style merge, compute, un-merge inside each encoder layer (weights unchanged) could skip redundant work. The residual stream, conv module and decoder keep the full 80 ms grid.

Accuracy was fine (dev-clean / Earnings-22, 400 samples, base 1.69 / 10.88):

- FF merge (anchored blocks of 2, cosine at least 0.95, group-mean input, from layer 6): 1.67 / 10.78 with 33% of FF evaluations skipped. Chain grouping drifts: at 0.95 it gave 9.2.
- Attention merge (proportional attention, relative positions gathered at the anchors): FF + attention 1.64 / 10.85, but only about 9% of attention inputs merge.
- Conv-module pointwise merge is destructive: 3.24 / 18.5.
- The rewritten layer with merging off reproduces NeMo's encoder output to 2.3e-4.

Speed was not (PyTorch, dev-clean, batch 32, bf16 + FP8 + compile):

| Configuration | RTF |
|---|---|
| Standard NeMo path | 876 |
| Rewritten layer, no merge | 799 |
| Rewritten layer, FF + attention merged | 821 |
| Rewritten layer, FF merged | 840 |
| Packed FF merge, no merge / merged | 759 / 804 |

The redundancy is real but bounded and in the wrong place for this GPU: gather/scatter, a host sync per grouping and compile graph breaks cost more than the FLOPs saved, and with FF at about 34% of encoder time, skipping a third of it is worth about 11% at best. The same merge on the Grace CPU (20 threads, PyTorch fp32) gave 0.90x real time against 0.92x unmerged. The frame-merging encoder (`engine/encoder_merge.py`, which also provides the 4-to-3 pool used in [03-pruning-and-distillation.md](03-pruning-and-distillation.md)) is included; the merge evaluation and packed-merge scripts live in the original research repository and are not included.

### Untrained frame pooling

A fixed 4-to-3 frame pool (`LEAN_POOL43`), with no retraining (dev-clean / Earnings-22, 400 samples):

| Pool | dev-clean | Earnings-22 | RTF dev-clean |
|---|---|---|---|
| After layer 11 | 1.90 | 11.44 | 2,051 |
| After layer 8 | 1.95 | 12.26 | 2,127 |

Against 1,916x for the same engine without pooling at the time: 7 to 11% speed for a clear WER loss.

### Engine-level tweaks with no gain

| Change | Result |
|---|---|
| Shape tuning (opt 800, batch 64, decode 256) | 1,664 to 1,775 dev-clean RTF; no gain |
| fp16 strongly typed engine | 1,763; no gain |
| `builderOptimizationLevel 5` | no gain |
| rel_shift instead of the index gather, alone (`LEAN_RELSHIFT=1`) | 203.5 to 200.4 ms in `trtexec`, no end-to-end gain (1,870); it paid only later with heads-first |
| Fused QKV linear in TensorRT (`LEAN_QKV=1`) | 126.2 against 126.0 ms |

### NVFP4 through TensorRT's native PTQ

ModelOpt NVFP4 PTQ on all linears: dev-clean 2.31, Earnings-22 11.86, 1,871x. Rejected: WER worse, and the dynamic activation quantisation ate most of the GEMM gain. The feed-forward-only NVFP4 route with a custom CUTLASS kernel is in [02-ultra-25-languages.md](02-ultra-25-languages.md).

### Decoder on a green context

**Hypothesis:** give the decoder its own SM partition (CUDA driver green contexts wrapped as torch external streams, `engine/green_streams.py`, `--green N`) so it runs alongside the encoder instead of queueing behind it.

| Partition | Encoder | Decoder |
|---|---|---|
| Full GPU | 368 ms | 53 ms |
| 40 SMs (encoder) / 8 SMs (decoder) | 392 ms (+6.6%) | 55 ms (+5%) |

The partitions worked, but the pipeline got slower: 4,275 to 4,282x with 6 or 8 SMs, 3,854x with 12. A pipeline trace (`PIPE_TRACE=1`) showed why: each decoder group (30 to 100 ms) already runs while the next group encodes, so wall time is the encoder chain and partitioning only slows the encoder.

### Other runner ideas

| Idea | Result |
|---|---|
| Decoder stream at high CUDA priority (`--dec-priority`), three interleaved pairs | 4,460 against 4,458; no effect |
| Mel features on the CPU in a worker thread (`--cpu-pre`) | 6.5 to 6.8 s per pass against 0.18 s on the GPU (about 1,340x); dead |
| Pipeline overlap | about 2% only (see above) |
| PyTorch encoder on the Grace CPU | about 1x real time; not viable for this model |

### A hand-written runtime

Estimated, not built: a hand-written runtime (fused layer norm, residual and quantise, SiLU in the GEMM epilogue) would leave about 55 ms of unavoidable memory traffic against about 70 ms at the time it was estimated, a 10 to 15% further encoder gain for a large effort. Most of that was later captured by the plugins above.

## 9. Controlled stock versus final, three models

### Protocol

`engine/stock_vs_final.sh MODEL ENGINE OUT_DIR [SOAK_S]`:

1. 60 s idle baseline on the wall meter.
2. GEMM heat soak (`engine/gemm_hold.py` in a loop): 300 s in the headline runs, 30 s in the earlier dry runs.
3. Stock arm: unmodified NeMo (`engine/stock_rtf.py`), bf16 autocast, CUDA-graph batched greedy decoder, batch 32 sorted. Full test-clean for WER, then 5 timed passes.
4. 120 s re-soak.
5. Final arm: the engine through `engine/trt_rtf.py` with the flags above. Full test-clean for WER, then 5 timed passes.
6. Throughout: wall power at 1 s, GPU clock, temperature and power every 2 s. `engine/temp_watchdog.sh` stops the run at 92 °C (it never tripped).

Both arms use batched greedy decoding. parakeet-tdt-1.1b ships the per-utterance `greedy` strategy (215x as shipped); `ASR_DECODING=greedy_batch` (the default in `engine/run_in_container.sh`) gives it the same batched decoder as the others, so the stock arm is not a straw man.

Porting the engine to the 1.1B models needed three changes: biases in the FFN plugin (their FF linears have them), the feature count from the model config (80 mel, not 128) and CTC decoding in the runner. Everything else, all plugins included, carried over unchanged.

### Dry runs at the 2,200 MHz cap (30 s soak)

| Model | Stock WER | Final WER | Stock RTF | Final RTF | Speed-up | J per audio hour gross, stock to final | Peak temperature |
|---|---|---|---|---|---|---|---|
| parakeet-tdt-0.6b-v3 | 1.931 | 1.918 | 977 | 4,488 | **4.59x** | 572 to 166 | 79 °C |
| parakeet-tdt-1.1b | 1.386 | 1.392 | 667 | 2,970 | **4.45x** | 855 to 260 | 82 °C |
| parakeet-ctc-1.1b | 1.844 | 1.833 | 706 | 3,072 | **4.35x** | 826 to 252 | 83 °C |

### Headline: driver-default clocks (300 s soak)

| Model | Stock WER | Final WER | Stock RTF | Final RTF | Speed-up | J/audio h gross | J/audio h net | Peak temperature |
|---|---|---|---|---|---|---|---|---|
| parakeet-tdt-0.6b-v3 | 1.931 | 1.918 | 1,008 | 4,580 | **4.54x** | 670 to 181 | 414 to 113 | 85 °C |
| parakeet-tdt-1.1b | 1.386 | 1.392 | 684 | 2,991 | **4.37x** | 993 to 279 | 594 to 168 | 84 °C |
| parakeet-ctc-1.1b | 1.844 | 1.833 | 727 | 3,094 | **4.26x** | 948 to 267 | 573 to 161 | 84 °C |

- Idle was 71.5 to 75.8 W in this session.
- Under load the GPU clock averaged 2,350 to 2,370 MHz in the stock arms and 2,180 to 2,250 MHz in the final arms. The final arms run harder and are power-limited at about 190 W at the wall.
- The GPU counters showed brief software and hardware thermal slowdown events during the run.
- Unlocking gains little over the 2,200 MHz cap: about +2% for the final arm and +3% for stock.

The speed-up is slightly lower unlocked than capped because stock benefits more from the extra clock headroom: the engine is already power-limited.

## 10. Energy

Energy is whole-machine wall power from a Tapo P304M smart power strip at 1 s, integrated over the timed window and divided by audio hours, gross and net of an idle baseline ([05-measurement-method.md](05-measurement-method.md)). Idle drifted between sessions (36.8 W, then 62 to 65 W, then 71.5 to 75.8 W) because of other background load on the machine, so gross figures from different sessions are not comparable; net figures are the better guide, and the same-session stock-versus-final pairs are the only claims.

| Configuration | Session idle | J per audio hour, gross | J per audio hour, net |
|---|---|---|---|
| First bar, NeMo bf16 batch 32 | 35.7 W | 346 | 176 |
| Early research harness | 36.8 W | 426 | 234 |
| Best PyTorch | 36.8 W | 328 | 176 |
| FP8 TensorRT, NeMo export | 62.3 W | 275 (not comparable) | 107 |
| Lean FP8 TensorRT (best PyTorch 150 net in the same session) | 64.9 W | | 100 |
| + FFN, SubConv02, heads-first, QKVHeads, RelPosAttn (mean 154 W, peak 164 W) | 63 W | 197 | 116 |
| Stock-weight engine at the end of the plugin work (mean 167 W, peak 181 W) | 65 W | 148 | 90.5 |
| **Headline: stock arm** | 71.5 W | **670** | **414** |
| **Headline: final arm** | 71.5 W | **181** | **113** |

The 90.5 J net figure and the 113 J headline figure come from different sessions with different idle, clock policy and soak; the later controlled pair (414 to 113 J net) is the one to quote. The stock arm of the controlled run also uses more energy than the first bar measurement (414 against 176 J net). The two were not measured under the same conditions: the first bar ran while the GPU clock was still locked at 1,176 MHz, with a much lower idle, and without the 300 s heat soak.
