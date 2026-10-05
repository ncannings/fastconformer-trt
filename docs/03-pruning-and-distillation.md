# 3. Pruning and distillation: making the 0.6B model smaller

With the stock-weight engine at its FP8 floor, the next question was whether changing the model itself (fewer frames, fewer layers, 2:4 sparse weights, recovered by self-distillation) could push parakeet-tdt-0.6b-v3 towards 10,000x with minimal accuracy loss. It bought up to 6,087x against 4,467x for the stock weights in the same session, at a clear cost in WER, and a smaller stock model, parakeet-tdt_ctc-110m, then beat every pruned 0.6B candidate on both speed and accuracy. All training here used English data only (LibriSpeech and AMI) on v3; no multilingual evaluation of the trained models is reported.

## 1. Rules

These rules were fixed before any retrained number was seen.

- **Training:** self-distillation from the frozen stock model only: targets are its own greedy transcripts plus its encoder outputs, on training audio, never on a test set. Trainer: `engine/finetune_stride.py`.
- **Training data:** LibriSpeech train-clean-100 and train-clean-360 plus AMI IHM train.
- **Engine:** every retrained model runs through the same engine path as the stock-weight final (`LEAN_CKPT=... engine/build_variant.sh` plus all plugins) and the same runner flags. `trt_rtf.py --ckpt` loads the fine-tuned decoder and joint as well as the encoder.
- **Accuracy bars**, against the stock-weight final engine on the same samples:

| Set | Reference | Bar |
|---|---|---|
| test-clean, full 2,620 (held out, never used for selection) | 1.90 to 1.92 | <= 1.97 |
| Earnings-22, 2,000-utterance sample (seed 20261005) | 10.52 to 10.56 | <= 10.70 |
| dev-clean, 2,000-utterance sample | 1.95 | <= 2.00 |

- A candidate passes only if it meets all three. Selection uses dev-clean and Earnings-22 only; test-clean is read once per finalist.
- **Stop rule:** if the main candidate misses, report the measured frontier and stop; no bar is re-opened.

## 2. Background: the same idea in PyTorch

Before the TensorRT engine existed, the same levers were tried in PyTorch (bf16, SDPA, FP8 linears, `torch.compile`; 400-utterance dev-clean and Earnings-22 samples):

| Candidate | dev / Earnings-22 WER | Speed against the optimised unmodified model |
|---|---|---|
| Unmodified | 1.70 / 10.86 | 1.00x |
| P1: 4-to-3 frame pool after layer 9 + skip layers 0, 2, 16, 17 | 2.13 / 12.19 | 1.28x |
| P2: 4-to-3 frame pool after layer 9, all layers distilled 2.5 h | 1.89 / 10.92 | 1.13x |
| Frame halving (several variants) | 3.1 to 3.4 / 15 to 15.4 | about 1.4 to 1.5x |
| 8 layers removed (step 1,000) | 4.02 / 28.6 | n/a |

The pool in P2 replaces every 4 frames by 3 after layer 9, averaging the last two (`encoder_merge.pool43` in `engine/encoder_merge.py`), so later layers and the decoder see 25% fewer frames. P2 scored 2.07 on held-out test-clean against 1.94 to 1.96 for the unmodified model. With up to about 10 h of distillation per candidate, every 1.5x reduction cost at least 1.4 points of WER, and every reduction within the accuracy bars gave at most about 1.13 to 1.28x.

Method lessons that carried forward:

- freeze batch norm during fine-tuning;
- train all layers, not only those after the cut;
- take distillation targets from a frozen teacher;
- untrained variants understate trained speed, because wrong transcripts double the decoding time;
- a 96-utterance quick-dev set reads optimistic; monitor on a 400-utterance set or larger.

## 3. R1 and R2: the frame pool through the engine

| Step | What | Training |
|---|---|---|
| R1 | The P2 checkpoint (pool after layer 9; 2.5 h) through the engine | none beyond P2 |
| R2 | P2 continued for 7 h with linear learning-rate decay: 18,449 steps, 253,637 utterances | 7 h |

Export equivalence against the training-time reference: max difference 0.051 (R1) and 0.047 (R2), lengths equal. Encoder per 32 x 16 s batch: 75.5 ms (R1) and 74.0 ms (R2), against 87.2 ms for stock weights.

A runner fix was needed on the way: the first R1 reading (dev-clean 7.7%) paired the fine-tuned encoder with the stock decoder and joint, whereas P2 had fine-tuned those too.

| Check (bar) | R1 | R2 | Stock-weight final |
|---|---|---|---|
| test-clean, full 2,620 (<= 1.97) | 2.02 (2,000-utterance sample) | **1.957** pass | 1.90 to 1.92 |
| Earnings-22, 2,000 (<= 10.70) | 10.38 pass | **10.29** pass | 10.52 |
| dev-clean, 2,000 (<= 2.00) | 2.08 fail | **2.0015** fail by 0.0015 | 1.95 |
| RTF test-clean, same session | about 4,955 (1.11x) | **5,035** (1.13x) | 4,459 to 4,476 |

R2 is not a pass under the pre-registration, which requires all three bars; the bar was not rounded. The dev-clean miss is about one word in 2,000 utterances. Against the stock-weight final, R2 is better on Earnings-22 and worse by 0.04 to 0.06 on the LibriSpeech sets. R3 (a deeper cut) was not started, because its precondition, R2 passing with margin, was not met.

## 4. Track B: towards 10,000x

**Goal:** 10,000x on test-clean on this machine with minimal degradation, documenting the speed and accuracy cost of every step so readers can choose where to stop. Reference: the stock-weight engine at 4,576x, test-clean 1.90 to 1.92%.

Levers: frame pooling after earlier layers, layer dropping, 2:4 structured sparsity on the encoder linears (sparse FP8 tensor cores) and NVFP4. Encoder work against stock, by pooling:

| Pools | Encoder work |
|---|---|
| 4 to 3 after L9 | 0.85 |
| + 4 to 3 after L4 | 0.69 |
| + halving after L14 | about 0.59 |

### 2:4 sparse GEMMs on sm_121

- PyTorch's semi-structured sparse path (cuSPARSELt) ran 4 to 14x *slower* than dense on this GPU; its CUTLASS path is sm_8x only.
- CUTLASS 4.1's sm_120 sparse FP8 kernels work: `engine/plugins/sparse_fp8.cu`, relative error 2e-4.
- A coarse tile sweep (`engine/plugins/sparse_sweep.cu`, `sweep_sparse.py`), measured with background load on the GPU: 1.15 to 2.5x against dense cuBLAS depending on shape.
- Only one kernel schedule (cooperative) is available; valid tiles are 128x128x128, 128x128x256 and 128x256x128.

### The pruning criterion matters enormously

R2 checkpoint, 2:4 on all FF, attention and pointwise linears, no recovery training, 96-utterance quick-dev WER (`engine/probes/prune_probe.py`):

| Criterion | Quick-dev WER |
|---|---|
| Unpruned | 1.86 |
| Magnitude | 88.53 |
| Wanda (\|W\| * \|\|x_in\|\|) | 9.42 |
| Wanda, FF only | 3.95 |
| Wanda, FF + pointwise | 7.09 |

Recovery by distillation (Wanda, all groups, from R2), quick-dev WER:

| Step | 0 | 500 | 1,000 |
|---|---|---|---|
| T1w | 13.3 | 3.38 | 2.62 |

### Where to cut: training-free probes

On R2, quick-dev WER, baseline 1.86.

Extra frame pool (`engine/probes/pool_probe.py`):

| Extra pool | Quick-dev WER | Encoder work |
|---|---|---|
| 4 to 3 after L4 | 3.95 | 0.693 |
| 4 to 3 after L6 | 3.24 | 0.714 |
| 4 to 3 after L12 | 2.90 | 0.768 |
| 4 to 3 after L14 | 3.09 | 0.784 |
| Halving after L14 | 23.5 | |
| Halving after L16 | 28.9 | |
| Halving after L18 | 36.8 | |

Halving the frame rate needs far more training than a few hours.

Single layer dropped (`engine/probes/drop_probe.py`): layers 0 to 20 each cost 0 to 0.2; layer 21 gives 2.38, layer 22 gives 4.14 and layer 23 gives 98.5.

Greedy cumulative drops (`engine/probes/drop_greedy.py`):

| Layers dropped | Quick-dev WER |
|---|---|
| 4 | 1.90 |
| 6 | 2.24 |
| 8 (1, 3, 5, 8, 11, 13, 16, 20) | 3.57 |

Layer dropping is far cheaper than early pooling.

### Combined runs stopped on plateaus

| Run | Configuration (from R2 unless stated) | Quick-dev WER trajectory |
|---|---|---|
| T2 | Pools after L4 and L9 + all-linear 2:4, from T1w | 7.33, 4.24, 4.05, 3.57, 3.38 at 0 to 2,000 steps; 3.38 to 3.71 at 2,000 to 2,500 |
| T3 | Pools after L4 and L9 + FF-only 2:4 | 15.2, 3.52, 3.33, 3.05 at 0 to 1,500 steps |
| T4 | Drop 8 layers + pools after L9 and L12 + Wanda 2:4 on FF | 51.7, 8.47, 5.81, 4.95, 4.90 at 0 to 2,000 steps |

The early pool after L4 dominated the cost of T2 and T3. T4 was designed as the 10,000x candidate (encoder work about 0.50 of stock before sparsity; the decoder sees about 0.56 of the frames), but the cuts compound far worse together than separately (each alone: 2.90 to 3.95). This mirrors how R2 was reached: one cut, then recovery.

The T4 architecture was still built as an engine, to measure its speed: 32 sparse FF plugin blocks, 16 QKVHeads, 16 RelPosAttn, dropped layers as identities (as in training, and skipped during QKV plugin calibration). Export matched its reference to 0.013. Encoder **51.1 ms** per 32 x 16 s batch against 87.2 for stock weights, measured while training was running on the same GPU: a speed-up of at least 1.70x before sparse tile tuning.

### Staged schedule

Each stage starts from the previous stage's weights:

| Stage | Adds | Training |
|---|---|---|
| S1 | Drop 4 free layers (1, 3, 8, 13) from R2 | 1.5 h, 4,025 steps |
| S2 | Wanda 2:4 on FF | 2 h, 5,559 steps |
| S3 | 4-to-3 pool after L12 | 2 h (not run) |
| S4 | Drop 4 more layers (5, 11, 16, 20) | 3 h (not run) |

PyTorch bf16, 400-utterance seeded samples (stock 1.70 dev-clean / 10.86 Earnings-22; R2 1.82 / 10.92):

| Stage | dev-clean | Earnings-22 | Last quick-dev |
|---|---|---|---|
| S1 | 2.24 | 11.93 | 1.71 |
| S2 | 2.69 | 12.96 | 2.14 |

S1 cost +0.42 dev-clean and +1.0 Earnings-22 against R2: the "free" untrained drops were free on the 96-utterance quick-dev only, and the 400-utterance sets showed the cost. The chain was stopped after S2, which was already well past minimal degradation.

## 5. The sparse plugin bug

The first sparse S2 engine gave 97.7% WER. `FFNFp8::Weights::upload()` compressed the 2:4 weights before copying them to the device, so it compressed uninitialised memory. The unit test compressed buffers that were already filled, so it never saw this. The fix is to copy, then compress.

The T4-architecture timing above is unaffected, and according to the research record no earlier accuracy number used the sparse FF path. (The research record also notes an earlier check in which dense and sparse T1w engines agreed on dev-clean, 2.837 against 2.849; how that engine avoided the bug is not recorded.)

Other safeguards in the sparse path: `engine/plugins/sparse_surgery.py` quantises each weight exactly as TensorRT would and refuses any weight that is not exactly 2:4 along K after FP8 quantisation. Sparse variants exist for FFNFp8 (`FFN_SPARSE=1`), QKVHeads (`QKV_SPARSE=1`, per-head compressed, batched) and SpLinearFp8 (pw1, pw2 and linear_out with the residual folded).

## 6. Results

Same session, test-clean full, 2,200 MHz cap, frame budget off, batch 32:

| Engine | test-clean WER | RTF | Encoder per 32 x 16 s batch |
|---|---|---|---|
| Stock weights | 1.918 | 4,467 | 87.2 ms |
| R2 | 1.957 | 5,035 | 74.0 ms |
| S1 | 2.280 | 5,810 | 62.3 ms |
| S2, dense kernels | 2.604 | 5,774 | 62.3 ms |
| S2, sparse kernels (after the fix) | 2.59 | 6,087 | 59.6 ms |

The S2 sparse row was recorded in a later comparison ("S2 at 6,087x and 2.59"), not in the same session table; it is given here for completeness.

## 7. Conclusion: switching model beat pruning

parakeet-tdt_ctc-110m, unmodified, through stock NeMo:

| Model | test-clean | dev-clean (2,000) | Earnings-22 (2,000) | Stock NeMo RTF |
|---|---|---|---|---|
| parakeet-tdt-0.6b-v3 | 1.931 | 1.95 | 10.52 | 1,008 |
| parakeet-tdt_ctc-110m | 2.415 | 2.45 | 10.81 | 2,293 |

With no retraining it is more accurate than S1 and S2 on test-clean, level on Earnings-22 and starts 2.3x faster. Through the engine it reaches 11,052x at 2.44 test-clean, against 6,087x at 2.59 for the pruned 0.6B S2 ([04-small-model-110m.md](04-small-model-110m.md)). The smaller stock model dominates the pruning route on both axes.

The best same-model figures therefore stay at 4,580x with stock weights and 6,087x pruned (+0.67 test-clean). Two caveats apply to everything in this document:

- all training used English data only (LibriSpeech and AMI) on v3, which is a 25-language model; the effect of pruning on its other 24 languages was not measured;
- the training-free probes and quick-dev numbers use a 96-utterance set (seed 20261008) that reads optimistic, as S1 showed.

Not included in this repository: the staged training driver (`stages.sh`), the sparse GEMM micro-benchmarks and the other research scripts referred to above live in the original research repository.
