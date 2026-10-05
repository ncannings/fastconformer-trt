# fastconformer-trt build and run image: NVIDIA's NeMo 25.11 container (TensorRT 10.13, CUDA 13, PyTorch, ModelOpt)
# plus the scoring packages and CUTLASS 4.8 headers (the plugins' CUTLASS kernels; NVFP4 needs >= 4.8).
#   docker build -t fastconformer-trt:25.11 .
FROM nvcr.io/nvidia/nemo:25.11
RUN pip install -q jiwer whisper-normalizer soundfile scipy pyarrow
# CUTLASS 4.8 (header-only), pinned to the commit the kernels were validated against
RUN git clone -q https://github.com/NVIDIA/cutlass.git /opt/cutlass && git -C /opt/cutlass checkout -q 0b55a2f
