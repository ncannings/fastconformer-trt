#!/bin/bash
# Export a lean-encoder variant (LEAN_* env), swap in the FFN and subsampling plugins, build the TensorRT engine and
# profile one 32 x 16 s batch. usage: LEAN_...=1 build_variant.sh OUT_DIR   (OUT_DIR under ~/asr_data, e.g. lean11)
set -e
R=${RUNNER:-./run_in_container.sh}        # RUNNER=./run_nolock.sh while a training run holds the GPU lock
LK="flock /tmp/fastconformer-trt.lock"; [ -n "$NOLOCK" ] && LK=""
D=$1; cd $(dirname $0)
TAG=$(python3 -c "import os;print(('_pm1' if os.environ.get('LEAN_PREMASK_ONCE')=='1' else '')+('_h' if os.environ.get('LEAN_HALF')=='1' else '')+('_dws' if os.environ.get('LEAN_DWSHIFT')=='1' else '')+('_rs' if os.environ.get('LEAN_RELSHIFT')=='1' else '')+('_hb' if os.environ.get('LEAN_HB')=='1' else '')+('_qkv' if os.environ.get('LEAN_QKV')=='1' or os.environ.get('LEAN_QKV_PLUGIN')=='1' else '')+('p' if os.environ.get('LEAN_QKV_PLUGIN')=='1' else '')+('_attn' if os.environ.get('LEAN_ATTN_PLUGIN')=='1' else ''))")
$R lean_export.py /data/$D 2>&1 | grep -E "EQUIV|POSCACHE|position table|calibration:|Traceback|Error" | head -6
FIN=/data/$D/ffn.onnx
if [ "${FFN_SKIP:-0}" = 1 ]; then FIN=/data/$D/$(basename $(ls -t ${ASR_DATA_DIR:-$HOME/asr_data}/$D/lean*_fp8.onnx | head -1)); echo "FF blocks left to TensorRT (FFN_SKIP=1)"; else
FFN_RESIDUAL=${FFN_RESIDUAL:-0} FFN_SPARSE=${FFN_SPARSE:-0} $R plugins/ffn_surgery.py /data/$D/$(basename $(ls -t ${ASR_DATA_DIR:-$HOME/asr_data}/$D/lean*_fp8.onnx | head -1)) /data/$D/ffn.onnx 2>&1 | grep -E "replaced|Traceback" | head -2
fi
NPZ=$(ls -t ${ASR_DATA_DIR:-$HOME/asr_data}/$D/lean*_fp8.qkv.npz 2>/dev/null | head -1)   # side data only with LEAN_QKV_PLUGIN
QKV_SPARSE=${QKV_SPARSE:-0} $R plugins/qkv_surgery.py $FIN /data/$D/ffn_q.onnx ${NPZ:+/data/$D/$(basename $NPZ)} 2>&1 | grep -E "rewired|attached|Traceback" | head -3
Q=ffn_q; if [ "${SP_LINEAR:-0}" = 1 ]; then $R plugins/sparse_surgery.py /data/$D/ffn_q.onnx /data/$D/ffn_qs.onnx 2>&1 | grep -E "sparse:|dense:|Traceback|refusing|SP_DENSE" | head -2; Q=ffn_qs; fi
if [ "${GLU_FUSE:-0}" = 1 ]; then $R plugins/glu_surgery.py /data/$D/$Q.onnx /data/$D/${Q}g.onnx 2>&1 | grep -E "glu:|Traceback|expected|not found" | head -2; Q=${Q}g; fi
# SubConv02 fuses the subsampling as masked once (LEAN_PREMASK_ONCE=1); NeMo's per-layer masking is left unfused
S=$Q; if [ "${LEAN_PREMASK_ONCE:-0}" = 1 ]; then $R plugins/sub_surgery.py /data/$D/$Q.onnx /data/$D/ffn_sub.onnx 2>&1 | grep -E "replaced|Traceback" | head -2; S=ffn_sub; else echo "subsampling not fused (LEAN_PREMASK_ONCE unset)"; fi
if [[ "${ASR_MODEL:-}" == *.nemo ]]; then NF=$(tar -xOf "${ASR_MODEL/#\/data/${ASR_DATA_DIR:-$HOME/asr_data}}" ./model_config.yaml | python3 -c "import sys,yaml;print(yaml.safe_load(sys.stdin)['preprocessor']['features'])"); else
NF=$(ASR_MODEL=${ASR_MODEL:-nvidia/parakeet-tdt-0.6b-v3} docker run --rm -v $HOME/.cache/huggingface:/root/.cache/huggingface -e ASR_MODEL ${ASR_IMAGE:-fastconformer-trt:25.11} python -c "
import os, tarfile, glob, yaml
from huggingface_hub import hf_hub_download, list_repo_files
r = os.environ['ASR_MODEL']; f = [x for x in list_repo_files(r) if x.endswith('.nemo')][0]
t = tarfile.open(hf_hub_download(r, f)); c = [n for n in t.getnames() if n.endswith('model_config.yaml')][0]
print('NFEAT', yaml.safe_load(t.extractfile(c))['preprocessor']['features'])" 2>/dev/null | grep NFEAT | cut -d' ' -f2); fi
echo "features: $NF"
TE="docker run --rm --gpus all --ipc=host -v ${ASR_DATA_DIR:-$HOME/asr_data}:/data -v $PWD/plugins:/plug ${ASR_IMAGE:-fastconformer-trt:25.11} /usr/src/tensorrt/bin/trtexec"
PREC="--fp16 --fp8"; [ "$LEAN_HALF" = 1 ] && PREC="--stronglyTyped"     # fp16 export: types fixed by the graph
$LK $TE --onnx=/data/$D/$S.onnx --staticPlugins=/plug/libffn_fp8.so --saveEngine=/data/$D/engine.plan $PREC --minShapes=audio_signal:1x${NF}x100,length:1 --optShapes=audio_signal:32x${NF}x1600,length:32 --maxShapes=audio_signal:${MAXB:-32}x${NF}x6000,length:${MAXB:-32} --memPoolSize=workspace:16384 --skipInference 2>&1 | grep -E "Engine built|\[E\]" | head -3
$LK $TE --loadEngine=/data/$D/engine.plan --staticPlugins=/plug/libffn_fp8.so --shapes=audio_signal:32x${NF}x1600,length:32 --iterations=30 --warmUp=1000 --dumpProfile --separateProfileRun --exportProfile=/data/$D/prof.json 2>&1 | grep -E "GPU Compute Time:" | cut -c1-120
