# live/: concurrent real-time streams per GPU

Code for [docs/09-live-streaming.md](../docs/09-live-streaming.md): a multi-stream live server harness for NVIDIA's
cache-aware streaming model `nvidia/nemotron-3.5-asr-streaming-0.6b`, the concurrency sweep with the frozen C120 rule,
a TensorRT cache-aware encoder step (FP16 or FP8) and our fused RNN-T decoder. Stock NeMo runs in the same harness
unchanged, so both arms are measured the same way. No model weights, ONNX graphs or engines are included; you build
them from NVIDIA's published checkpoint (OpenMDW-1.1) on the machine that runs them.

## Files

| File | What it does |
|---|---|
| `harness.py` | The server: N simulated real-time streams, slot-indexed cache slab, mel per chunk on the GPU, continuous batching, per-chunk timing, late-chunk and latency statistics. Stock step = NeMo's `conformer_stream_step` split at its seams; the encoder step and the decoder are pluggable. |
| `sweep.py` | Concurrency search and the pass rule (at most 0.1% of chunks late, p95 final-token latency at most 1,000 ms, 120 s confirmation), one fresh process per trial with `--fresh-process`. |
| `loadgen.py` | Seeded load: Earnings-22 calls as long-form sources, seeded start points, staggered arrivals. |
| `fused_decoder.py` | Our RNN-T greedy decoder (`--decoder fused`): fixed-shape CUDA graphs captured once at start-up, Triton joint + argmax and LSTM kernels (`FUSED_JOINT`, `FUSED_LSTM`: `triton` as measured, `torch`/`cudnn` for the NeMo-identical path). |
| `lean_decoder.py` | NeMo's own decoder computer with decoder state in GPU slabs (`--decoder lean`; an earlier arm). |
| `trt/export_step.py`, `trt/recipe.py`, `trt/common.py` | Export of the encoder streaming step to ONNX (first and steady chunks), FP8 post-training quantisation with TensorRT Model Optimizer. |
| `trt/build_engine.sh` | `trtexec` engine build for one exported graph. |
| `trt/trt_step.py` | The TensorRT step as a drop-in for the harness (`--encoder /w/trt/trt_step.py:build --engine ...`). |
| `harness_wer.py`, `score.py` | Accuracy gate: LibriSpeech test-clean through the harness, every utterance one stream. |
| `demo_live.py`, `video_acts.sh` | The terminal dashboard used for the videos. |
| `equiv.py`, `dec_equiv.py`, `trt/equiv_step.py`, `check_stock_slab.py`, `trt/check_slab_step.py`, `trt/check_call_path.py`, `recapture_check.py`, `late_check.py`, `mem_probe.py` | Correctness and method checks (harness against NeMo's streaming script, decoder against NeMo's, engine against the PyTorch step, piece-wise cache steps bit-identical, zero CUDA-graph re-captures, the late-chunk split, memory peaks). |
| `p0_stock.py`, `make_manifests.py`, `trt/wer_trt.py` | Stock WER through NeMo's own streaming script, unmodified. |
| `bench_decoder.py`, `trt/bench_step.py`, `profile_step.py`, `trt/fq_step.py`, `trt/sensitivity.py`, `env_check.py` | Development tools (microbenchmarks, profile, FP8 sensitivity). |

## 1. Container, data and model

```bash
# NeMo 26.06 (NeMo 3.1.0) with the ASR extras and the scorer's normalisers; NeMo itself unchanged
docker build -t nemo-live:26.06 live

# data into ~/asr_data: test_clean, dev_clean and earnings22_full (the Open ASR Leaderboard Earnings-22 test set)
engine/fetch_data.sh

# the model into the Hugging Face cache (run_live.sh mounts it read-only and runs offline)
python -c "from huggingface_hub import hf_hub_download as h; h('nvidia/nemotron-3.5-asr-streaming-0.6b', 'nemotron-3.5-asr-streaming-0.6b.nemo')"
```

`live/run_live.sh SCRIPT.py ARGS` runs a script in the container as the calling user, with `live/` as `/w`, the
repository's `engine/` (data loading and the scorer) as `/speech`, `~/asr_data` read-only as `/data` and
`~/asr_data/live` as `/out` for everything written. It limits the container to `LIVE_CPUS` CPUs (default 2.5, as in
our Spark runs) and runs at nice 19. `LIVE_PY=0 live/run_live.sh CMD ...` runs a command instead of Python. On a
machine that is itself the NeMo container (cloud pods), install the same extras (`live/Dockerfile`) and run the
scripts directly with `PYTHONPATH=live:engine ASR_DATA=~/asr_data LIVE_OUT=~/asr_data/live`, adjusting the `/w` and
`/out` paths in the commands below.

## 2. Engines (per chunk size: r = 0, 3, 13 for 80 ms, 320 ms, 1.12 s)

```bash
cd live
# ONNX: the float graph (for FP16 engines) and the FP8 Q/DQ graph (ModelOpt, 64 dev-clean utterances, seed 20261005)
./run_live.sh trt/export_step.py --r 13 --quant none
./run_live.sh trt/export_step.py --r 13 --quant fp8
# engines: first-chunk and steady-chunk graphs; B up to 1,024 streams per call profile (calls are capped at 512 rows)
for k in first steady; do
  MAXB=1024 OPTB=512 TMIN=1 trt/build_engine.sh ml 13 $k fp16
  MAXB=1024 OPTB=512 TMIN=1 trt/build_engine.sh ml 13 $k fp8
done
```

Engines land in `~/asr_data/live/trt/ml_r13/` (`steady_fp16_b1024.plan`, `first_fp16_b1024.plan` and so on). The
harness is given the steady engine; the first-chunk engine is found next to it. Engines are specific to the GPU and
TensorRT version. On the H100 we built the first-chunk engines (and the 80 ms steady engine) with `MAXB=2048`; the
1.12 s steady engine failed to build at 2,048 there (6 October) and uses 1,024.

## 3. Accuracy gate

```bash
FX="env TRT_MAX_CALL_B=512 FUSED_JOINT=triton FUSED_LSTM=triton HARNESS_SLAB_STEP=1 HARNESS_PINNED_H2D=1"
# stock reference
./run_live.sh harness_wer.py /out/wer_r13_stock.json --set test_clean --r 13 --decoder stock
# ours (add --mel-graph at r = 0)
LIVE_PY=0 ./run_live.sh $FX python harness_wer.py /out/wer_r13_fp16.json --set test_clean --r 13 \
  --encoder /w/trt/trt_step.py:build --engine /out/trt/ml_r13/steady_fp16_b1024.plan --cache-dtype float16 --decoder fused
```

The gate is the arm's WER at most 2% (relative) above stock's at the same chunk size on the same GPU.

## 4. Capacity sweeps (C120)

Run on a quiet GPU, one arm at a time. These are the settings of the reported runs:

```bash
COMMON="--fresh-process --late-rule wait --dur-s 60 --confirm-dur-s 120 --tick-ms 0 --growth 1.25 --crash-as-fail"
# stock NeMo + gc.freeze (the piece-wise stock cache step and pinned uploads, as in every reported stock run)
LIVE_PY=0 ./run_live.sh env HARNESS_GC=freeze HARNESS_STOCK_SLAB_STEP=1 HARNESS_PINNED_H2D=1 HARNESS_UTIL_S=1 \
  python sweep.py /out/sweeps/stock_r13 --r 13 --n-start 315 $COMMON
# ours: FP16 (or FP8) engine + fused decoder; add --mel-graph at r = 0
LIVE_PY=0 ./run_live.sh env HARNESS_GC=default TRT_MAX_CALL_B=512 FUSED_JOINT=triton FUSED_LSTM=triton \
  HARNESS_SLAB_STEP=1 HARNESS_PINNED_H2D=1 HARNESS_UTIL_S=1 \
  python sweep.py /out/sweeps/fp16_r13 --r 13 --n-start 1242 $COMMON \
  --encoder /w/trt/trt_step.py:build --engine /out/trt/ml_r13/steady_fp16_b1024.plan --cache-dtype float16 --decoder fused
```

Each trial is one line of `OUT/results.jsonl` with the full harness summary; the last line per chunk size is the
result (`confirmed_max_streams` is C120). Our Spark start points were 58, 171 and 315 (stock) and 239, 680 and 1,242
(ours) at r = 0, 3 and 13. On the H100, NeMo's decoder pre-warm crashed at 384 slots or more; set `HARNESS_PREWARM=0`
for stock sweeps that go above that (the encoder warm-up stays on). `--deadline-epoch` stops starting new trials at a
given time. To score the raw files of the published runs: `python results/live/summarise.py`.

## 5. The dashboard

```bash
LIVE_TTY=1 LIVE_PY=0 ./run_live.sh env TRT_MAX_CALL_B=512 FUSED_JOINT=triton FUSED_LSTM=triton HARNESS_SLAB_STEP=1 \
  HARNESS_STOCK_SLAB_STEP=1 python demo_live.py --acts stock:13:300,ours:13:300 \
  --engines 13=/out/trt/ml_r13/steady_fp16_b1024.plan --ours-label "OURS: FP16 + fused decoder" --log /out/demo.log
```

## Environment switches

| Switch | Effect |
|---|---|
| `HARNESS_GC=freeze` | `gc.collect()` then `gc.freeze()` after load and warm-up (stock arm). |
| `HARNESS_PREWARM=0` | Skip NeMo's decoder warm-up at the full slot count (on by default). |
| `HARNESS_WARM_ENC=0` | Skip the encoder and mel warm-up at every reachable batch size (on by default). |
| `HARNESS_SLAB_STEP=1` | TensorRT step reads and writes the cache slab piece by piece (one cache copy per stream). |
| `HARNESS_STOCK_SLAB_STEP=1` | The same for NeMo's own encoder step (pieces of `HARNESS_STOCK_PIECE`, default 512, float32 caches). |
| `HARNESS_PINNED_H2D=1` | Per-step index uploads through pinned buffers, asynchronously. |
| `HARNESS_UTIL_S` | GPU utilisation sampling period in seconds (NVML; 0 = off). |
| `TRT_MAX_CALL_B` | Largest number of rows per TensorRT call; larger batches run in pieces. |
| `FUSED_JOINT`, `FUSED_LSTM` | Fused decoder kernels: `triton` (measured) or `torch` / `cudnn` (NeMo-identical). |
| `RESERVE_HOST_GB`, `RESERVE_CUDA_GB` | Start-up memory reservation (`gpu_reserve.py`; for unified-memory machines shared with other jobs). |

The harness and checks were written on a shared DGX Spark: several docstrings say that timings from a shared GPU are
not reportable. Only quiet-machine runs are reported in docs/09.
