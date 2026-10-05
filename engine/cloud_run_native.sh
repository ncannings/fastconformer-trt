#!/bin/bash
# Runs ON a cloud GPU machine that is itself the NeMo 25.11 container, as root with no Docker (e.g. a RunPod pod):
# Ultra, stock NeMo vs engine in the Spark and datacentre configurations. Results in /root/asr_data/results_box.
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
log "convert ultra"; ./run_native.sh ultra_to_nemo.py moondream/parakeet-ultra /data/ultra.nemo 2>&1 | grep -E "filled|saved|Error" | tee -a $R/steps.log
export ASR_MODEL=/data/ultra.nemo FAST_JOINT_FP8=1 FFN_PLUGIN_LIB=/w/plugins/libffn_fp8.so
log "engine: Spark configuration"; MAXB=128 FFN_RESIDUAL=1 LEAN_PREMASK_ONCE=1 LEAN_DWSHIFT=1 LEAN_RELSHIFT=1 LEAN_QKV_PLUGIN=1 LEAN_ATTN_PLUGIN=1 LEAN_HB=1 SP_LINEAR=1 SP_DENSE=1 LEAN_POSCACHE=1 SUB_PW3=1 ./build_variant.sh ultra_f 2>&1 | tee $R/build_f.log | grep -E "EQUIV|Engine built|GPU Compute|Traceback"
log "engine: datacentre configuration"; MAXB=256 FFN_SKIP=1 LEAN_PREMASK_ONCE=1 LEAN_DWSHIFT=1 LEAN_RELSHIFT=1 LEAN_QKV=1 LEAN_ATTN_PLUGIN=1 LEAN_HB=1 LEAN_POSCACHE=1 ./build_variant.sh ultra_dc 2>&1 | tee $R/build_dc.log | grep -E "EQUIV|Engine built|GPU Compute|Traceback"
cp /data/ultra_f/prof.json $R/prof_f.json 2>/dev/null; cp /data/ultra_dc/prof.json $R/prof_dc.json 2>/dev/null
W="--wer-sets test_clean,earnings22_full --wer-n 100000 --rtf-set test_clean --reps 5"
F="--batch 32 --dec-batch 256 --dec-bf16 --pipeline --pre-compile --fast-joint --fast-hyps"
log "stock arm"; ./run_native.sh stock_rtf.py /data/results_box/stock.json $W 2>&1 | grep -E '^\{"|Traceback|Error' | cut -c1-300 | tee -a $R/steps.log
for v in "f_fb9600 ultra_f 9600" "dc_fb9600 ultra_dc 9600" "dc_fb19200 ultra_dc 19200"; do set -- $v
  log "engine $1"; ./run_native.sh trt_rtf.py /data/$2/engine.plan /data/results_box/engine_$1.json --frame-budget $3 --max-batch 128 $F $W 2>&1 | grep -E '^\{"|Traceback|Error' | cut -c1-300 | tee -a $R/steps.log
done
nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu --format=csv >> $R/gpu.csv
log "BOX RUN DONE"
