"""CUTLASS 2:4 sparse FP8 GEMM on GB10: correctness against a dense GEMM with the same pruned weights, and speed
against dense FP8 (cuBLAS and our CUTLASS FFN kernels) at the encoder's shapes."""
import ctypes, os, time
import torch
d = os.path.dirname(os.path.abspath(__file__))
sp = ctypes.CDLL(os.path.join(d, "libffn_fp8.so"))
sp.sparse_fp8_prepare.restype = ctypes.c_void_p
sp.sparse_fp8_prepare.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
sp.sparse_fp8_run.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_float, ctypes.c_void_p]
st = torch.cuda.current_stream().cuda_stream
def bench(f, n=50):
    for _ in range(5): f()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.time() - t) / n * 1e3
one = torch.ones((), device="cuda")
for M, K, N in ((6400, 1024, 4096), (6400, 4096, 1024), (6400, 1024, 3072)):
    w = torch.randn(N, K, device="cuda")
    g = w.abs().view(N, K // 4, 4)                       # magnitude 2:4: keep the 2 largest of every 4
    keep = torch.zeros_like(g, dtype=torch.bool).scatter_(2, g.topk(2, dim=2).indices, True).view(N, K)
    w8 = (w * keep * 20).to(torch.float8_e4m3fn).contiguous()
    x8 = (torch.randn(M, K, device="cuda") * 2).to(torch.float8_e4m3fn).contiguous()
    h = sp.sparse_fp8_prepare(w8.data_ptr(), N, K, st)
    y = torch.empty(M, N, device="cuda", dtype=torch.float16)
    rc = sp.sparse_fp8_run(h, x8.data_ptr(), y.data_ptr(), M, 1.0, st); torch.cuda.synchronize()
    ref = torch._scaled_mm(x8, w8.t(), scale_a=one, scale_b=one, out_dtype=torch.float32)
    err = float((y.float() - ref).norm() / ref.norm())
    td = bench(lambda: torch._scaled_mm(x8, w8.t(), scale_a=one, scale_b=one, out_dtype=torch.float16))
    ts = bench(lambda: sp.sparse_fp8_run(h, x8.data_ptr(), y.data_ptr(), M, 1.0, st))
    print(f"M{M} K{K} N{N}: rc {rc} rel err {err:.1e} | dense cuBLAS {td:.3f} ms | 2:4 sparse CUTLASS {ts:.3f} ms | {td/ts:.2f}x", flush=True)
