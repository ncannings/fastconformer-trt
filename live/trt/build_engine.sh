#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Nigel Cannings. Part of fastconformer-trt (see LICENSE and NOTICE).
# Build a TensorRT engine for one exported streaming-step graph (export_step.py) with trtexec in nemo-live:26.06.
# One optimisation profile: B (streams) from 1 to MAXB (default 512; the step wrapper splits larger batches), opt OPTB;
# T from TMIN to the chunk width (the last chunk of an utterance can be shorter in NeMo's stock script).
# usage: trt/build_engine.sh MODEL_r R KIND PREC        e.g. trt/build_engine.sh ml 13 steady fp8
#   KIND first|steady, PREC fp16 (from the fp32 graph, --fp16) | fp8 (from the FP8 Q/DQ graph, --fp16 --fp8)
#   env: MAXB, OPTB, PLUGIN=/out/trt/plugins/libffn_fp8.so ONNX_IN=name.onnx TAG=suffix
set -e
LIVE=$(cd $(dirname $0)/.. && pwd)
M=$1; R=$2; KIND=$3; PREC=$4
D=/out/trt/${M}_r${R}
G=$HOME/asr_data/live/trt/${M}_r${R}/geometry.json
W=$(python3 -c "import json;g=json.load(open('$G'));print(g['${KIND}_w'])")
MAXB=${MAXB:-512}; OPTB=${OPTB:-128}; TMIN=${TMIN:-9}
case $PREC in
  fp16) SRC=${ONNX_IN:-${KIND}_fp32.onnx}; FLAGS="--fp16";;
  fp8)  SRC=${ONNX_IN:-${KIND}_fp8.onnx};  FLAGS="--fp16 --fp8";;
  fp32) SRC=${ONNX_IN:-${KIND}_fp32.onnx}; FLAGS="";;
esac
OUT=${KIND}_${PREC}${TAG}_b${MAXB}.plan
sh() { echo "audio_signal:$1x128x$2,length:$1,cache_last_channel:24x$1x56x1024,cache_last_time:24x$1x1024x8,cache_last_channel_len:$1"; }
PL=""; [ -n "$PLUGIN" ] && PL="--staticPlugins=$PLUGIN"
echo "build $D/$OUT from $SRC flags '$FLAGS' B 1..$MAXB opt $OPTB T $TMIN..$W $PL"
cd $LIVE
LIVE_PY=0 LIVE_NAME=trt-build-${M}${R}${KIND}${PREC} ./run_live.sh trtexec --onnx=$D/$SRC --saveEngine=$D/$OUT $FLAGS $PL \
  --minShapes=$(sh 1 $TMIN) --optShapes=$(sh $OPTB $W) --maxShapes=$(sh $MAXB $W) \
  --memPoolSize=workspace:${WS:-4096} --skipInference --builderOptimizationLevel=${OPTLEVEL:-3} 2>&1 \
  | grep -vE "strongly typed network is rec|\[V\]" | tail -${TAILN:-6} | cut -c1-500
ls -la $HOME/asr_data/live/trt/${M}_r${R}/$OUT
