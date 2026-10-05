"""Sparse FF block (sparse_ffn_run) vs the dense fused FF block (ffn_fp8_run_res) on the same 2:4-pruned weights:
correctness against a torch reference and relative speed (rough if the GPU is shared)."""
import ctypes, os, time
import torch
d = os.path.dirname(os.path.abspath(__file__))
sp = ctypes.CDLL(os.path.join(d, "libffn_fp8.so")); de = ctypes.CDLL(os.path.join(d, "libffn_fp8.so"))
sp.sparse_fp8_prepare.restype = ctypes.c_void_p
sp.sparse_fp8_prepare.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
sp.sparse_ffn_run.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] + [ctypes.c_float] * 4 + [ctypes.c_int, ctypes.c_void_p]
de.ffn_fp8_run_res.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 4 + [ctypes.c_float] * 4 + [ctypes.c_int, ctypes.c_void_p]
st = torch.cuda.current_stream().cuda_stream
M, K1, N1, N2 = 6400, 1024, 4096, 1024
def prune(w):
    g = w.abs().view(w.shape[0], -1, 4)
    return w * torch.zeros_like(g, dtype=torch.bool).scatter_(2, g.topk(2, dim=2).indices, True).view(w.shape)
w1 = prune(torch.randn(N1, K1, device="cuda") * 0.05); w2 = prune(torch.randn(N2, N1, device="cuda") * 0.03)
x = torch.randn(M, K1, device="cuda"); res = torch.randn(M, N2, device="cuda").half()
sx, s1, s2 = float(x.abs().max() / 448), float(w1.abs().max() / 448), float(w2.abs().max() / 448)
x8, w18, w28 = (x / sx).to(torch.float8_e4m3fn), (w1 / s1).to(torch.float8_e4m3fn), (w2 / s2).to(torch.float8_e4m3fn)
h = torch.nn.functional.silu((x8.float() @ w18.float().t()) * sx * s1); sh = float(h.abs().max() / 448)
ref = res.float() + 0.5 * ((h / sh).to(torch.float8_e4m3fn).float() @ w28.float().t()) * sh * s2
h1 = sp.sparse_fp8_prepare(w18.data_ptr(), N1, K1, st); h2 = sp.sparse_fp8_prepare(w28.data_ptr(), N2, N1, st)
h8 = torch.empty(M, N1, device="cuda", dtype=torch.float8_e4m3fn); ys = torch.empty(M, N2, device="cuda", dtype=torch.float16)
yd = torch.empty_like(ys)
run_s = lambda: sp.sparse_ffn_run(h1, h2, x8.data_ptr(), h8.data_ptr(), res.data_ptr(), ys.data_ptr(), M, sx * s1, 1 / sh, sh * s2, 0.5, 0, st)
run_d = lambda: de.ffn_fp8_run_res(x8.data_ptr(), h8.data_ptr(), res.data_ptr(), yd.data_ptr(), w18.data_ptr(), w28.data_ptr(), M, K1, N1, N2, sx * s1, 1 / sh, sh * s2, 0.5, 0, st)
print("rc sparse", run_s(), "dense", run_d()); torch.cuda.synchronize()
rel = lambda a: float((a.float() - ref).norm() / ref.norm())
def bench(f, n=50):
    for _ in range(5): f()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.time() - t) / n * 1e3
print(f"rel err sparse {rel(ys):.1e} dense {rel(yd):.1e} | dense FF block {bench(run_d):.3f} ms, sparse {bench(run_s):.3f} ms", flush=True)
