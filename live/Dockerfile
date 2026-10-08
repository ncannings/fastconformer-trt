# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Nigel Cannings. Part of fastconformer-trt (see LICENSE and NOTICE).
# nemo-live:26.06: nvcr.io/nvidia/nemo:26.06 with its bundled NeMo source (/opt/NeMo, NeMo 3.1.0) installed with the
# ASR extras, plus the offline scorer's text normalisers. The 26.06 container ships /opt/NeMo but not the ASR
# collection's Python dependencies (import nemo.collections.asr fails on `lightning`). The container's own torch,
# torchvision, torchaudio, triton and numpy are pinned so pip cannot replace them. NeMo itself is not changed.
# build: docker build -t nemo-live:26.06 live
FROM nvcr.io/nvidia/nemo:26.06
RUN pip list --format=freeze | grep -iE '^(torch|torchvision|torchaudio|triton|numpy|pytorch-triton)==' > /tmp/pins.txt \
 && cat /tmp/pins.txt \
 && MAX_JOBS=4 pip install --no-cache-dir -c /tmp/pins.txt -e "/opt/NeMo[asr]" whisper_normalizer jiwer \
 && pip list --format=freeze | grep -iE '^(torch|torchvision|torchaudio|triton|numpy)==' > /tmp/after.txt \
 && diff /tmp/pins.txt /tmp/after.txt
# Some directories in the base image lack the execute bit for other users (e.g. /opt/Megatron-Bridge/3rdparty), which
# breaks `import nemo` when the container runs as the calling user (run_live.sh); make directories traversable.
RUN find /opt -type d ! -perm -o+x -exec chmod a+rx {} +
