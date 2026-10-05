"""Sparse FP8 tile sweep (sparse_sweep.cu variants 0..2 = 128x128x128, 128x128x256, 128x256x128) against dense
cuBLAS FP8 at the encoder's GEMM shapes (M = 6400 frames). Run on a quiet GPU."""
import ctypes, os, time
import torch
d = os.path.dirname(os.path.abspath(__file__)); sp = ctypes.CDLL(os.path.join(d, "libsparse_sweep.so"))
sp.sparse_fp8_prepare.restype = ctypes.c_void_p
sp.sparse_fp8_prepare.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
sp.sparse_sweep_run.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
st = torch.cuda.current_stream().cuda_stream; one = torch.ones((), device="cuda")
def bench(f, n=50):
    for _ in range(5): f()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.time() - t) / n * 1e3
def prune(w):
    g = w.abs().view(w.shape[0], -1, 4)
    return w * torch.zeros_like(g, dtype=torch.bool).scatter_(2, g.topk(2, dim=2).indices, True).view(w.shape)
for name, K, N in (("ff1", 1024, 4096), ("ff2", 4096, 1024), ("pw1", 1024, 2048), ("pw2/out", 1024, 1024)):
    M = 6400
    w8 = (prune(torch.randn(N, K, device="cuda")) * 30).to(torch.float8_e4m3fn); x8 = (torch.randn(M, K, device="cuda") * 3).to(torch.float8_e4m3fn)
    h = sp.sparse_fp8_prepare(w8.data_ptr(), N, K, st); y = torch.empty(M, N, device="cuda", dtype=torch.float16)
    td = bench(lambda: torch._scaled_mm(x8, w8.t(), scale_a=one, scale_b=one, out_dtype=torch.float16))
    row = [f"{name}: dense {td:.3f}"]
    for v in range(3):
        rc = sp.sparse_sweep_run(v, h, x8.data_ptr(), y.data_ptr(), M, st)
        row.append(f"v{v} {bench(lambda: sp.sparse_sweep_run(v, h, x8.data_ptr(), y.data_ptr(), M, st)):.3f}" if rc == 0 else f"v{v} rc{rc}")
    print(" | ".join(row), flush=True)
