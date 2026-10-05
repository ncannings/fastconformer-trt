#!/bin/bash
# Sweep kernel variants with tune_kernels.py (one process per setting); prints one JSON line each.
# usage (in the container, from plugins/): bash sweep_kernels.sh "0 1 3 4 8 9 10 11 12 13"
V=${1:-"0 1 2 3 4 5 6 7"}
for v1 in $V; do FFN_V1=$v1 python tune_kernels.py ffn 2>/dev/null | tail -1; done                 # GEMM1 (default GEMM2)
for vr in -1 $V; do FFN_VR=$vr python tune_kernels.py ffn 2>/dev/null | tail -1; done               # residual GEMM2
for vr in -1 $V; do SPL_VR=$vr python tune_kernels.py lin 2>/dev/null | tail -1; done               # dense residual linear
for q in 0 1 2 3 4 5 6 7 8 9; do QKV_V=$q python tune_kernels.py qkv 2>/dev/null | tail -1; done   # QKV heads-first
