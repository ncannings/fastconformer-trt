#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Nigel Cannings. Part of fastconformer-trt (see LICENSE and NOTICE).
# Run a script from live/ inside the NeMo 26.06 image (nemo-live:26.06, see Dockerfile). No flock: this must never
# wait on, or block, the offline project's lock. Runs as the calling user so outputs stay user-owned.
# Shared-machine limits: LIVE_CPUS CPUs (default 2.5), 2 OpenMP threads, nice 19 inside the container, data read-only.
# Outputs go to ~/asr_data/live (/out in the container).
# usage: ./run_live.sh script.py [args...]      (LIVE_PY=0 ./run_live.sh cmd ... runs a command instead of python)
LIVE=$(cd $(dirname $0) && pwd)
SPEECH=$(cd $LIVE/../engine && pwd)   # engine/finetune_stride.py: data loading and the scorer's normalisers
mkdir -p $HOME/asr_data/live
if [ "${LIVE_PY:-1}" = 1 ]; then CMD="python"; else CMD=""; fi
exec docker run --rm ${LIVE_TTY:+-t -e TERM=${TERM:-xterm-256color} -e COLUMNS=$(tput cols 2>/dev/null || echo 120) -e LINES=$(tput lines 2>/dev/null || echo 40)} ${LIVE_NAME:+--name $LIVE_NAME} --gpus all --ipc=host --ulimit memlock=-1 --cpus=${LIVE_CPUS:-2.5} \
  --user $(id -u):$(id -g) -e HOME=/tmp/home -e HF_HOME=/hf -e NUMBA_CACHE_DIR=/tmp/numba \
  -v $HOME/.cache/huggingface:/hf:ro \
  -v ${ASR_DATA_DIR:-$HOME/asr_data}:/data:ro -v $HOME/asr_data/live:/out \
  -v $LIVE:/w -v $SPEECH:/speech:ro -w /w -e PYTHONPATH=/w:/speech \
  -e ASR_DATA=/data -e LIVE_OUT=/out -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 -e TOKENIZERS_PARALLELISM=false \
  -e HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} -e RESERVE_HOST_GB=${RESERVE_HOST_GB:-6} -e RESERVE_CUDA_GB=${RESERVE_CUDA_GB:-4} \
  ${LIVE_IMAGE:-nemo-live:26.06} nice -n 19 $CMD "$@"
