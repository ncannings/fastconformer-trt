#!/bin/bash
# Build libffn_fp8.so (CUTLASS kernels + TensorRT plugin) inside the NeMo container. sm_120 family code runs on GB10.
set -e
C=${CUTLASS_DIR:-/opt/cutlass}   # CUTLASS 4.8 (the image pins it); NVFP4 needs >= 4.8
if [ "${FC_SM:-120}" = 90 ]; then      # Hopper (GH200, H100): FP8 paths only; sparse and NVFP4 are stubs that refuse
  ARCH="-gencode=arch=compute_90a,code=sm_90a -DFC_SM90"
  SRC="ffn_fp8.cu ffn_plugin.cpp sub_plugin.cu qkv_heads.cu relpos_plugin.cpp splinear_plugin.cpp glu_plugin.cpp sm90_stubs.cu"
else                                   # Blackwell sm_120 family (DGX Spark GB10)
  ARCH=${FFN_ARCH:-"-gencode=arch=compute_120f,code=sm_120f"}
  SRC="ffn_fp8.cu ffn_plugin.cpp sub_plugin.cu qkv_heads.cu relpos_plugin.cpp sparse_fp8.cu splinear_plugin.cpp glu_plugin.cpp nvfp4_ffn.cu"
fi
nvcc -std=c++17 -O3 $ARCH --expt-relaxed-constexpr -I$C/include -I$C/tools/util/include -I/usr/include/aarch64-linux-gnu \
  -Xcompiler -fPIC -shared -o libffn_fp8.so $SRC -lnvinfer -lcuda 2>&1 | grep -v "warning #" | grep -E "error|Error|warning" | head -30
ls -la libffn_fp8.so
