"""ctypes test of libffn_fp8.so: fused FF block vs torch reference (accuracy) and vs the unfused cuBLAS path
(_scaled_mm, SiLU + quantise, _scaled_mm) for speed, at the encoder's FF shapes (M = 6400 frames, 1024 -> 4096 -> 1024)."""
import ctypes, os, time
import torch

lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libffn_fp8.so"))
lib.ffn_fp8_cutlass_ws.restype = ctypes.c_size_t
lib.ffn_fp8_run.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 4 + [ctypes.c_float] * 3 + \
    [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
M, K1, N1, N2 = 6400, 1024, 4096, 1024
torch.manual_seed(0)
x = torch.randn(M, K1, device="cuda") * 1.5
w1 = torch.randn(N1, K1, device="cuda") * 0.05
w2 = torch.randn(N2, N1, device="cuda") * 0.03
sx, sw1, sw2 = float(x.abs().max() / 448), float(w1.abs().max() / 448), float(w2.abs().max() / 448)
x8 = (x / sx).to(torch.float8_e4m3fn); w18 = (w1 / sw1).to(torch.float8_e4m3fn); w28 = (w2 / sw2).to(torch.float8_e4m3fn)
h_ref = torch.nn.functional.silu((x8.float() @ w18.float().t()) * sx * sw1)
sh = float(h_ref.abs().max() / 448)
h_q = (h_ref / sh).to(torch.float8_e4m3fn)
y_ref = (h_q.float() @ w28.float().t()) * sh * sw2
y_exact = torch.nn.functional.silu(x @ w1.t()) @ w2.t()
h8 = torch.empty(M, N1, device="cuda", dtype=torch.float8_e4m3fn)
y = torch.empty(M, N2, device="cuda", dtype=torch.float16)
wsz = lib.ffn_fp8_cutlass_ws(M, K1, N1, N2)
ws = torch.empty(max(wsz, 1), device="cuda", dtype=torch.uint8)
stream = torch.cuda.current_stream().cuda_stream
run = lambda: lib.ffn_fp8_run(x8.data_ptr(), h8.data_ptr(), y.data_ptr(), w18.data_ptr(), w28.data_ptr(), M, K1, N1, N2,
                              sx * sw1, 1.0 / sh, sh * sw2, ws.data_ptr(), wsz, stream)
rc = run(); torch.cuda.synchronize()
print("rc", rc, "workspace", wsz)
rel = lambda a, b: float((a.float() - b.float()).norm() / b.float().norm())
print(f"H8 vs reference quantised: {rel(h8.float(), h_q.float()):.2e}; Y vs FP8 reference {rel(y, y_ref):.2e}; "
      f"Y vs fp32 exact {rel(y, y_exact):.2e}; FP8 reference vs exact {rel(y_ref, y_exact):.2e}")
one = torch.ones((), device="cuda")
sxa, sw1a, sw2a, sha = [torch.tensor(v, device="cuda") for v in (sx, sw1, sw2, sh)]

def unfused():
    h = torch._scaled_mm(x8, w18.t(), scale_a=sxa, scale_b=sw1a, out_dtype=torch.float16)
    hq = (torch.nn.functional.silu(h) * (1 / sh)).to(torch.float8_e4m3fn)
    return torch._scaled_mm(hq, w28.t(), scale_a=sha, scale_b=sw2a, out_dtype=torch.float16)

def bench(f, n=50):
    for _ in range(5): f()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.time() - t) / n
tf, tu = bench(run), bench(unfused)
g1 = lambda: torch._scaled_mm(x8, w18.t(), scale_a=sxa, scale_b=sw1a, out_dtype=torch.float16)
g2 = lambda: torch._scaled_mm(h8, w28.t(), scale_a=sha, scale_b=sw2a, out_dtype=torch.float16)
fl = 2 * M * K1 * N1 + 2 * M * N1 * N2
print(f"fused CUTLASS {tf*1e3:.3f} ms ({fl/tf/1e12:.0f} TF/s) | unfused cuBLAS {tu*1e3:.3f} ms | cuBLAS GEMM1 "
      f"{bench(g1)*1e3:.3f} GEMM2 {bench(g2)*1e3:.3f} ms")
