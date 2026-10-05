"""NVFP4 feed-forward (nvfp4_ffn.cu, CUTLASS >= 4.8): quantise x, GEMM1 with SiLU + NVFP4 output, GEMM2 + residual,
against a float reference and against the FP8 block (ffn_fp8_run_res) on the same random weights; speed of each.
The epilogue's norm_constant convention is checked both ways (output global scale g_h or its inverse)."""
import ctypes, os, time
import torch
d = os.path.dirname(os.path.abspath(__file__))
nv = ctypes.CDLL(os.environ.get("NV4_LIB", os.path.join(d, "libnvfp4.so"))); de = ctypes.CDLL(os.path.join(d, "libffn_fp8.so"))
nv.nvfp4_sf_bytes.restype = ctypes.c_size_t; nv.nvfp4_sf_bytes.argtypes = [ctypes.c_int, ctypes.c_int]
nv.nvfp4_quant.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 2 + [ctypes.c_float, ctypes.c_int, ctypes.c_void_p]
nv.nvfp4_gemm1.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3 + [ctypes.c_float, ctypes.c_void_p, ctypes.c_void_p]
nv.nvfp4_gemm2.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3 + [ctypes.c_float, ctypes.c_float, ctypes.c_void_p]
de.ffn_fp8_run_res.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 4 + [ctypes.c_float] * 4 + [ctypes.c_int, ctypes.c_void_p]
st = torch.cuda.current_stream().cuda_stream
torch.manual_seed(0)
M, K1, N1, N2 = 6400, 1024, 4096, 1024
x = torch.randn(M, K1, device="cuda").half(); res = torch.randn(M, N2, device="cuda").half()
w1 = (torch.randn(N1, K1, device="cuda") * 0.03).half(); w2 = (torch.randn(N2, N1, device="cuda") * 0.02).half()
h_ref = torch.nn.functional.silu(x.float() @ w1.float().t())
y_ref = res.float() + 0.5 * (h_ref @ w2.float().t())
G = lambda t: float(t.abs().max()) / (6 * 448)
def q(t, as_b):
    R, K = t.shape
    qq = torch.empty(R, K // 2, device="cuda", dtype=torch.uint8); sf = torch.empty(nv.nvfp4_sf_bytes(R, K), device="cuda", dtype=torch.uint8)
    g = G(t); assert nv.nvfp4_quant(t.data_ptr(), qq.data_ptr(), sf.data_ptr(), R, K, g, as_b, st) == 0
    return qq, sf, g
xq, sfx, gx = q(x, 0); w1q, sfw1, g1 = q(w1, 1); w2q, sfw2, g2 = q(w2, 1)
gh = float(h_ref.abs().max()) / (6 * 448)
hq = torch.empty(M, N1 // 2, device="cuda", dtype=torch.uint8); sfh = torch.empty(nv.nvfp4_sf_bytes(M, N1), device="cuda", dtype=torch.uint8)
y = torch.empty(M, N2, device="cuda", dtype=torch.float16)
for name, nc, a2 in (("norm = 1/g_h", 1 / gh, gh), ("norm = g_h", gh, 1 / gh)):
    ncd = torch.tensor([nc], device="cuda")
    r1 = nv.nvfp4_gemm1(xq.data_ptr(), sfx.data_ptr(), w1q.data_ptr(), sfw1.data_ptr(), hq.data_ptr(), sfh.data_ptr(), M, N1, K1, gx * g1, ncd.data_ptr(), st)
    r2 = nv.nvfp4_gemm2(hq.data_ptr(), sfh.data_ptr(), w2q.data_ptr(), sfw2.data_ptr(), res.data_ptr(), y.data_ptr(), M, N2, N1, 0.5 * a2 * g2, 1.0, st)
    torch.cuda.synchronize()
    ff = y.float() - res.float(); ffr = y_ref - res.float()
    print(f"{name}: rc {r1} {r2}; FF rel err {float((ff - ffr).norm() / ffr.norm()):.4f}")
# FP8 block on the same weights for comparison
s8 = lambda t: float(t.abs().max()) / 448
sx, s1, s2 = s8(x.float()), s8(w1.float()), s8(w2.float())
x8, w18, w28 = (x.float() / sx).to(torch.float8_e4m3fn), (w1.float() / s1).to(torch.float8_e4m3fn), (w2.float() / s2).to(torch.float8_e4m3fn)
sh = float(h_ref.abs().max()) / 448
h8 = torch.empty(M, N1, device="cuda", dtype=torch.float8_e4m3fn); y8 = torch.empty_like(y)
run8 = lambda: de.ffn_fp8_run_res(x8.data_ptr(), h8.data_ptr(), res.data_ptr(), y8.data_ptr(), w18.data_ptr(), w28.data_ptr(), M, K1, N1, N2, sx * s1, 1 / sh, sh * s2, 0.5, 0, st)
print("fp8 rc", run8()); torch.cuda.synchronize()
print(f"FP8 FF rel err {float(((y8.float() - res.float()) - (y_ref - res.float())).norm() / (y_ref - res.float()).norm()):.4f}")
ncd = torch.tensor([float(os.environ.get('NV4_NORM', 1 / gh))], device="cuda")
def run4():
    nv.nvfp4_quant(x.data_ptr(), xq.data_ptr(), sfx.data_ptr(), M, K1, gx, 0, st)
    nv.nvfp4_gemm1(xq.data_ptr(), sfx.data_ptr(), w1q.data_ptr(), sfw1.data_ptr(), hq.data_ptr(), sfh.data_ptr(), M, N1, K1, gx * g1, ncd.data_ptr(), st)
    nv.nvfp4_gemm2(hq.data_ptr(), sfh.data_ptr(), w2q.data_ptr(), sfw2.data_ptr(), res.data_ptr(), y.data_ptr(), M, N2, N1, 0.5 * gh * g2, 1.0, st)
def bench(f, n=50):
    for _ in range(5): f()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.time() - t) / n * 1e3
print(f"FF block: NVFP4 (incl. x quant) {bench(run4):.3f} ms | FP8 {bench(run8):.3f} ms")
