#!/bin/bash
# Runs ON a cloud GPU machine that is itself the NeMo 25.11 container, as root with no Docker (e.g. a RunPod pod):
# kernel-variant sweep, fused vs datacentre engine race, then a controlled stock-vs-engine headline with GPU energy.
# Copy this repository to /root/fct first (git archive HEAD | ssh ... "tar -x -C /root/fct"), then:
#   ssh ... "bash -s" < engine/cloud_run_native.sh
export HOME=/root; R=/root/asr_data/results_box; mkdir -p $R
# SSH sessions do not inherit the container's environment (PATH to nvcc, CUDA library paths): take PID 1's
while IFS='=' read -r k v; do case "$k" in PATH|LD_LIBRARY_PATH|CUDA_HOME|CUDA_PATH|LIBRARY_PATH|CPATH|C_INCLUDE_PATH|TRITON_*|NVIDIA_*|CUDNN_*|TRT_*|PYTHONIOENCODING|LANG) export "$k=$v";; esac; done < <(tr '\0' '\n' < /proc/1/environ)
command -v nvcc >/dev/null || export PATH=/usr/local/cuda/bin:$PATH
export C_INCLUDE_PATH=/usr/local/cuda/include${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}   # Triton helper needs cuda.h
log() { echo "$(date '+%H:%M:%S') $*" | tee -a $R/steps.log; }
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,power.limit,memory.total,compute_cap --format=csv | tee $R/gpu.csv
SM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.'); FCSM=120; case "$SM" in 90) FCSM=90;; 100) FCSM=100;; esac
log "compute capability $SM -> FC_SM=$FCSM"
pip install -q jiwer whisper-normalizer soundfile scipy pyarrow >> $R/steps.log 2>&1
[ -d /opt/cutlass ] || { git clone -q https://github.com/NVIDIA/cutlass.git /opt/cutlass && git -C /opt/cutlass checkout -q 0b55a2f; }
ln -sfn /root/fct/engine /w; ln -sfn /root/asr_data /data; ln -sfn /root/fct/engine/plugins /plug
export ASR_DATA_DIR=/data ASR_DATA=/data NATIVE=1
cd /root/fct/engine/plugins && log "plugins" && FC_SM=$FCSM bash build.sh 2>&1 | grep -E " error|libffn" | tee -a $R/steps.log
cd /root/fct/engine
log "attention cubin"; ./run_native.sh relpos_cubin.py plugins/relpos 2>&1 | grep -E '"DK"|Error' | tee -a $R/steps.log
for w in ffn lin qkv; do ./run_native.sh plugins/tune_kernels.py $w 2>&1 | tail -1 | tee -a $R/unit.log; done
log "data"; ./fetch_data.sh english >> $R/steps.log 2>&1
log "convert ultra"; [ -f /data/ultra.nemo ] || ./run_native.sh ultra_to_nemo.py moondream/parakeet-ultra /data/ultra.nemo 2>&1 | grep -E "filled|saved|Error" | tee -a $R/steps.log

# 1. kernel sweep: fastest correct variant of each production kernel on this GPU
cd /w/plugins; V="0 1 2 3 4 5 6 7"; [ "$FCSM" = 90 ] && V="0 1 2 3 4 5 6 7 8 9 10 11 12 13"
log "kernel sweep ($V)"
for v in $V; do FFN_V1=$v python tune_kernels.py ffn 2>/dev/null | tail -1; done > $R/sweep_v1.jsonl
for v in -1 $V; do FFN_VR=$v python tune_kernels.py ffn 2>/dev/null | tail -1; done > $R/sweep_vr.jsonl
for v in -1 $V; do SPL_VR=$v python tune_kernels.py lin 2>/dev/null | tail -1; done > $R/sweep_spl.jsonl
for v in 0 1 2 3 4 5 6 7 8 9; do QKV_V=$v python tune_kernels.py qkv 2>/dev/null | tail -1; done > $R/sweep_qkv.jsonl
best() { python3 -c "
import json,sys
rows=[json.loads(l) for l in open('$R/'+sys.argv[1]) if l.startswith('{')]
ok=[r for r in rows if r.get('rc')==0 and r.get('rel_err',1)<1e-3 and 'ms' in r]
b=min(ok,key=lambda r:r['ms']) if ok else {}
print(b.get(sys.argv[2]) or sys.argv[3])" "$@"; }
export FFN_V1=$(best sweep_v1.jsonl FFN_V1 1); export FFN_VR=$(FFN_V1=$FFN_V1 best sweep_vr.jsonl FFN_VR -1)
export SPL_VR=$(best sweep_spl.jsonl SPL_VR -1); export QKV_V=$(best sweep_qkv.jsonl QKV_V 0)
log "best variants FFN_V1=$FFN_V1 FFN_VR=$FFN_VR SPL_VR=$SPL_VR QKV_V=$QKV_V"
FFN_V1=$FFN_V1 FFN_VR=$FFN_VR python tune_kernels.py ffn 2>/dev/null | tail -1 | tee -a $R/sweep_best.jsonl
cd /w

# 2. engines: our fused plugins with the tuned variants vs the datacentre configuration
export ASR_MODEL=/data/ultra.nemo FAST_JOINT_FP8=1 FFN_PLUGIN_LIB=/w/plugins/libffn_fp8.so
log "engine fused (tuned)"; MAXB=128 FFN_RESIDUAL=1 LEAN_PREMASK_ONCE=1 LEAN_DWSHIFT=1 LEAN_RELSHIFT=1 LEAN_QKV_PLUGIN=1 LEAN_ATTN_PLUGIN=1 LEAN_HB=1 SP_LINEAR=1 SP_DENSE=1 LEAN_POSCACHE=1 ./build_variant.sh ultra_fused 2>&1 | tee $R/build_fused.log | grep -E "EQUIV|Engine built|GPU Compute|Traceback"
log "engine datacentre"; MAXB=128 FFN_SKIP=1 LEAN_PREMASK_ONCE=1 LEAN_DWSHIFT=1 LEAN_RELSHIFT=1 LEAN_QKV=1 LEAN_ATTN_PLUGIN=1 LEAN_HB=1 LEAN_POSCACHE=1 ./build_variant.sh ultra_dc 2>&1 | tee $R/build_dc.log | grep -E "EQUIV|Engine built|GPU Compute|Traceback"
cp /data/ultra_fused/prof.json $R/prof_fused.json 2>/dev/null; cp /data/ultra_dc/prof.json $R/prof_dc.json 2>/dev/null
F="--batch 32 --frame-budget 19200 --max-batch 128 --dec-batch 256 --dec-bf16 --pipeline --pre-compile --fast-joint --fast-hyps"
for e in fused dc fused dc; do log "race $e"; ./run_native.sh trt_rtf.py /data/ultra_$e/engine.plan /data/results_box/race_$e.json $F --wer-sets test_clean --wer-n 3000 --rtf-set test_clean --reps 3 2>&1 | grep -E '^\{"|Traceback' | cut -c1-200 | tee -a $R/steps.log; done
WIN=$(python3 -c "
import json
r={e:json.load(open('$R/race_'+e+'.json'))['rtf'] for e in ('fused','dc')}
print(max(r,key=r.get))"); log "winner $WIN"

# 3. controlled headline: idle, 300 s soak, stock arm, 120 s re-soak, engine arm; GPU power every second
plog() { nvidia-smi --query-gpu=timestamp,power.draw,clocks.sm,temperature.gpu,utilization.gpu --format=csv,noheader,nounits -lms 1000 > $R/power_$1.csv & echo $!; }
soak() { local end=$((SECONDS + $1)); while [ $SECONDS -lt $end ]; do ./run_native.sh gemm_hold.py > /dev/null 2>&1; done; }
P=$(plog idle); sleep 60; kill $P
log "soak 300 s"; soak 300
W="--wer-sets test_clean,earnings22_full --wer-n 100000 --rtf-set test_clean --reps 5"
P=$(plog stock); log "headline stock"; ./run_native.sh stock_rtf.py /data/results_box/headline_stock.json $W 2>&1 | grep -E '^\{"|Traceback' | cut -c1-300 | tee -a $R/steps.log; kill $P
soak 120
P=$(plog engine); log "headline engine ($WIN)"; ./run_native.sh trt_rtf.py /data/ultra_$WIN/engine.plan /data/results_box/headline_engine.json $F $W 2>&1 | grep -E '^\{"|Traceback' | cut -c1-300 | tee -a $R/steps.log; kill $P
python3 - <<'PY' | tee -a $R/steps.log
import csv, json, statistics, datetime
R = "/root/asr_data/results_box"
def rows(n):
    out = []
    for r in csv.reader(open(f"{R}/power_{n}.csv")):
        try: out.append((datetime.datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp(), float(r[1]), float(r[2]), float(r[3])))
        except Exception: pass
    return out
idle = statistics.mean(w for _, w, _, _ in rows("idle"))
out = {"gpu_idle_w": round(idle, 1), "energy_note": "GPU power from nvidia-smi only (no wall meter)"}
for arm in ("stock", "engine"):
    r = json.load(open(f"{R}/headline_{arm}.json")); t0, t1 = r["timed_window"]
    win = [x for x in rows(arm) if t0 <= x[0] <= t1]; w = statistics.mean(x[1] for x in win); h = r["audio_s"] / 3600 * len(r["walls_s"])
    out[arm] = {"rtf": round(r["rtf"]), "test_clean": round(r["test_clean"], 3), "earnings22_full": round(r["earnings22_full"], 3),
                "gpu_w_mean": round(w, 1), "gpu_J_per_audio_h": round(w * (t1 - t0) / h, 1),
                "gpu_J_per_audio_h_net": round((w - idle) * (t1 - t0) / h, 1),
                "sm_mhz_mean": round(statistics.mean(x[2] for x in win)), "temp_max_c": max(x[3] for x in win)}
out["speedup"] = round(out["engine"]["rtf"] / out["stock"]["rtf"], 2)
json.dump(out, open(f"{R}/headline_summary.json", "w"), indent=1); print(json.dumps(out))
PY
log "TUNE AND HEADLINE DONE"
