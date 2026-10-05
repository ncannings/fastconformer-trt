"""fp8_pw_glu_run vs torch (same e4m3 operands): y = (alpha x8 Wv^T) * sigmoid(alpha x8 Wg^T); speed vs one
[M, 2N] GEMM + torch GLU."""
import ctypes, os, time
import torch
lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libffn_fp8.so"))
lib.fp8_pw_glu_run.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 3 + [ctypes.c_float, ctypes.c_void_p]
M, K, N = 6400, 1024, 1024
x = torch.randn(M, K, device="cuda"); w = torch.randn(2 * N, K, device="cuda") * 0.03
sx, sw = float(x.abs().max() / 448), float(w.abs().max() / 448)
x8, w8 = (x / sx).to(torch.float8_e4m3fn), (w / sw).to(torch.float8_e4m3fn)
h = (x8.float() @ w8.float().t()) * sx * sw
ref = h[:, :N] * torch.sigmoid(h[:, N:].half().float())
gate = torch.empty(M, N, device="cuda", dtype=torch.float16); y = torch.empty(M, N, device="cuda", dtype=torch.float16)
st = torch.cuda.current_stream().cuda_stream
run = lambda: lib.fp8_pw_glu_run(x8.data_ptr(), w8.data_ptr(), gate.data_ptr(), y.data_ptr(), M, N, K, sx * sw, st)
print("rc", run()); torch.cuda.synchronize()
print("rel err", float((y.float() - ref).norm() / ref.norm()))
one = torch.ones((), device="cuda")
def plain():
    o = torch._scaled_mm(x8, w8.t(), scale_a=one * sx, scale_b=one * sw, out_dtype=torch.float16)
    return torch.nn.functional.glu(o, dim=-1)
def bench(f, n=50):
    for _ in range(5): f()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.time() - t) / n * 1e3
print(f"pw+GLU plugin {bench(run):.3f} ms | cuBLAS [M,2N] GEMM + GLU {bench(plain):.3f} ms | GEMM alone "
      f"{bench(lambda: torch._scaled_mm(x8, w8.t(), scale_a=one, scale_b=one, out_dtype=torch.float16)):.3f} ms")
