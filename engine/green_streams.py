"""CUDA green contexts for PyTorch: partition the GPU's SMs and get a torch stream bound to each partition.

Hypothesis: on GB10 the TDT decoder (many tiny, weight-bandwidth-bound kernels) and the encoder (compute-bound GEMMs)
serialise, because the encoder's kernels occupy every SM. Giving the decoder a few SMs of its own lets it run
concurrently; the encoder loses those SMs (less than proportionally, since about a third of its time is memory-bound).
make_streams(n_small) -> (big_stream, small_stream) as torch.cuda.ExternalStream objects, via cuda-python's driver API.
Run as a script: a GEMM timed on the full GPU, the big partition and the small partition.
"""
from __future__ import annotations

import torch
from cuda.bindings import driver as cu

_keep = []        # green contexts must outlive their streams


def _ok(r):
    """cuda-python returns (CUresult, values...): check, return the values (one value unwrapped)."""
    if not isinstance(r, tuple):
        r = (r,)
    if r[0] != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver error {r[0]}")
    vals = r[1:]
    return vals[0] if len(vals) == 1 else vals


def make_streams(n_small: int, device: int = 0):
    torch.cuda.init()
    _ok(cu.cuInit(0))
    dev = _ok(cu.cuDeviceGet(device))
    sm = _ok(cu.cuDeviceGetDevResource(dev, cu.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM))
    total = sm.sm.smCount
    groups, n_groups, rest = _ok(cu.cuDevSmResourceSplitByCount(1, sm, 0, n_small))   # n_small SMs + remainder
    small_res = groups[0] if isinstance(groups, (list, tuple)) else groups
    streams = []
    for res in (rest, small_res):
        desc = _ok(cu.cuDevResourceGenerateDesc([res], 1))
        g = _ok(cu.cuGreenCtxCreate(desc, dev, cu.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM))
        st = _ok(cu.cuGreenCtxStreamCreate(g, cu.CUstream_flags.CU_STREAM_NON_BLOCKING, 0))
        _keep.append((g, st))
        streams.append(torch.cuda.ExternalStream(int(st)))
    return streams[0], streams[1], total, (rest.sm.smCount, small_res.sm.smCount)


if __name__ == "__main__":
    import time
    big, small, total, split = make_streams(8)
    a = (torch.randn(8192, 8192, device="cuda") * 0.1).to(torch.float8_e4m3fn)
    one = torch.ones((), device="cuda")
    f = lambda: torch._scaled_mm(a, a.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16)

    def bench(stream, n=20):
        with torch.cuda.stream(stream):
            for _ in range(3): f()
            stream.synchronize(); t = time.time()
            for _ in range(n): f()
            stream.synchronize()
        return (time.time() - t) / n * 1e3
    full = torch.cuda.Stream()
    print(f"SMs {total} split {split}: full {bench(full):.2f} ms | big partition {bench(big):.2f} ms | small (8 SMs) {bench(small):.2f} ms")
