# 4. The small model: parakeet-tdt_ctc-110m

parakeet-tdt_ctc-110m is a different, smaller FastConformer model: English only and less accurate than the 0.6B, but much cheaper. With no retraining and only general changes to the engine, it runs at 11,052x real time at a 2,200 MHz clock cap and 11,222x at driver-default clocks, 4.73x stock NeMo on the same model, at 39 against 133 J per audio hour net. It shows that the engine transfers to another model size, and it sits above every pruned 0.6B model on the speed and accuracy frontier.

## 1. What this model is, and is not

- A different model, not a compressed 0.6B. It is English only; parakeet-tdt-0.6b-v3 covers 25 languages.
- About 25% worse on LibriSpeech: 2.44 against 1.92 test-clean through the engine. Earnings-22 is 0.3 worse.
- It is **not** 10,000x at 0.6B accuracy. The best same-model figures for the 0.6B stay at 4,580x (stock weights) and 6,087x (pruned, +0.67 test-clean); see [03-pruning-and-distillation.md](03-pruning-and-distillation.md).

Stock NeMo, unmodified models:

| Model | test-clean | dev-clean (2,000) | Earnings-22 (2,000) | Stock NeMo RTF |
|---|---|---|---|---|
| parakeet-tdt-0.6b-v3 | 1.931 | 1.95 | 10.52 | 1,008 |
| parakeet-tdt_ctc-110m | 2.415 | 2.45 | 10.81 | 2,293 |

## 2. Changes needed

No retraining. All changes are general; the 0.6B path is unchanged by them.

- **Head dim 64.** QKVHeads and RelPosAttn now accept head dim 64. In QKVHeads the CUTLASS tile N equals dk. The RelPosAttn Triton kernel is compiled per head dim (`RELPOS_DK=64 engine/relpos_cubin.py` produces `engine/plugins/relpos64.cubin`), and the plugin checks the cubin's dk at run time.
- **Exact subsampling masking.** The 110m's subsampling convolutions leak their biases into padded frames, so the mask-once shortcut used for the 0.6B (`LEAN_PREMASK_ONCE`) is not equivalent for it (max difference 1.32). The new `LEAN_SUBMASK=1` mode makes SubConv02 take the lengths and zero rows in-kernel exactly as NeMo's `MaskedConvSequential` does; the rest of the subsampling re-masks after every conv. Equivalence 0.032 (fp16 rounding). `engine/lean_equiv_probe.py` is the tool that finds which lean-encoder option breaks equivalence for a given model.
- **Runner:** the decoder warm-up had a hard-coded width of 1,024.

Unfused, the subsampling was 17 of the 41 ms encoder time per 32 x 16 s batch (the 840 MB conv.0 output written, masked and read back), which is why the masked SubConv02 matters far more here than on the 0.6B.

## 3. Results at the 2,200 MHz cap

Batch 32:

| 110m | test-clean (full) | dev-clean (2,000) | Earnings-22 (2,000) | RTF test-clean | Encoder per 32 x 16 s |
|---|---|---|---|---|---|
| Stock NeMo | 2.415 | 2.450 | 10.81 | 2,293 | |
| Engine, subsampling unfused | 2.413 | 2.448 | 10.85 | 8,536 | 41.7 ms |
| Engine, masked SubConv02 | 2.442 | 2.467 | 10.81 | **11,052** | 28.3 ms |

The residual fold into linear_out and pw2, worth +2.4% on the 0.6B, was tried here too: 10,760 against 10,851, which is within noise, so no gain on the small model.

## 4. Controlled run at driver-default clocks

`engine/stock_vs_final.sh`, driver-default clocks (2,418 MHz), 300 s heat soak, Tapo P304M wall meter, same session. From `results/headline/parakeet-tdt_ctc-110m_stock_machine.json`:

| 110m | test-clean | RTF | Wall W, mean | J per audio hour, gross | J per audio hour, net of idle | Loaded SM clock, mean | Peak temperature |
|---|---|---|---|---|---|---|---|
| Stock NeMo | 2.433 | 2,374 | 162.2 | 245.9 | 133.2 | 2,393 MHz | 79 °C |
| **Engine** | 2.465 | **11,222** | 175.1 | 67.4 | **38.8** | 2,337 MHz | 76 °C |

Idle 74.3 W. **4.73x faster.** A first run gave 2,378 against 11,349x, but the wall meter dropped its connection during the second arm, so that run has no energy figure.

## 5. Accuracy against the 0.6B

| Model, through the engine | test-clean | RTF (2,200 MHz cap) |
|---|---|---|
| parakeet-tdt-0.6b-v3, stock weights | 1.918 | 4,576 |
| parakeet-tdt-0.6b-v3, pruned and distilled (S2) | 2.59 | 6,087 |
| **parakeet-tdt_ctc-110m, stock weights** | **2.44** | **11,052** |

Against the pruned 0.6B, the 110m is both faster and more accurate on English. Against the unpruned 0.6B, it trades test-clean accuracy (2.44 against 1.92) and the 24 non-English languages for speed.

## 6. Run-to-run nondeterminism

The pipeline is not bit-deterministic run to run at the 0.03 WER level:

- the fused and unfused 110m engines differ on 107 of 2,620 test-clean transcripts, in both directions, from fp16 rounding;
- the unfused engine itself scored 2.413 and 2.440 on two runs.

Differences of a few hundredths of a WER point between configurations should therefore be read as noise unless they repeat. See [05-measurement-method.md](05-measurement-method.md).

## Final configuration (5 October)

With the later optimisations (frame-budget batching, cached position table, residual folding into the attention
output and pw2 GEMMs, including their biases), the 110m was re-measured on the stock machine (driver-default clocks,
300 s heat soak, wall meter): stock NeMo 2,397x at test-clean 2.433%, 137.7 J per audio hour net; engine **12,109x** at
2.453%, **39.1 J** per audio hour net: **5.05x**. Encoder 26.7 ms per batch of 32 x 16 s (28.3 ms before). The 110m keeps
the exactly masked SubConv02 (`LEAN_SUBMASK=1`); the tensor-core conv.3 fusion only exists for the mask-once path.
