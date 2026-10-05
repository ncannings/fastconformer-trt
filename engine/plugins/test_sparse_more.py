"""Correctness of sparse_linear_run (plain and residual) and the batched heads-first sparse_qkv_run."""
import ctypes, os
import torch
d = os.path.dirname(os.path.abspath(__file__)); sp = ctypes.CDLL(os.path.join(d, "libffn_fp8.so"))
sp.sparse_fp8_prepare_l.restype = ctypes.c_void_p
sp.sparse_fp8_prepare_l.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
sp.sparse_linear_run.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_void_p]
sp.sparse_qkv_run.argtypes = [ctypes.c_void_p * 3, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p * 3, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
st = torch.cuda.current_stream().cuda_stream
def prune(w):
    g = w.abs().view(w.shape[0], -1, 4)
    return w * torch.zeros_like(g, dtype=torch.bool).scatter_(2, g.topk(2, dim=2).indices, True).view(w.shape)
M, K, N = 3200, 1024, 2048
x8 = (torch.randn(M, K, device="cuda") * 3).to(torch.float8_e4m3fn); w8 = (prune(torch.randn(N, K, device="cuda")) * 30).to(torch.float8_e4m3fn)
res = torch.randn(M, N, device="cuda").half(); h = sp.sparse_fp8_prepare_l(w8.data_ptr(), N, K, 1, st)
ref = x8.float() @ w8.float().t() * 0.01
y = torch.empty(M, N, device="cuda", dtype=torch.float16)
r1 = sp.sparse_linear_run(h, x8.data_ptr(), None, y.data_ptr(), M, 0.01, 1.0, 0, st); torch.cuda.synchronize()
e1 = float((y.float() - ref).norm() / ref.norm())
r2 = sp.sparse_linear_run(h, x8.data_ptr(), res.data_ptr(), y.data_ptr(), M, 0.01, 0.5, 0, st); torch.cuda.synchronize()
ref2 = res.float() + 0.5 * ref; e2 = float((y.float() - ref2).norm() / ref2.norm())
print(f"SPLIN plain rc {r1} rel err {e1:.1e} | residual rc {r2} rel err {e2:.1e}", flush=True)
H, dk = 8, 128
W = [(prune(torch.randn(H * dk, K, device="cuda")) * 30).to(torch.float8_e4m3fn).contiguous() for _ in range(3)]
bias = torch.randn(3, H * dk, device="cuda")
hs = (ctypes.c_void_p * 3)(*[sp.sparse_fp8_prepare_l(w.data_ptr(), dk, K, H, st) for w in W])
outs = [torch.empty(H, M, dk, device="cuda", dtype=torch.float16) for _ in range(3)]
alpha3 = torch.tensor([0.01, 0.02, 0.03])
rq = sp.sparse_qkv_run(hs, x8.data_ptr(), bias.data_ptr(), (ctypes.c_void_p * 3)(*[o.data_ptr() for o in outs]), M, H, dk, alpha3.data_ptr(), st)
torch.cuda.synchronize()
for p in range(3):
    rp = (x8.float() @ W[p].float().t() * float(alpha3[p]) + bias[p]).view(M, H, dk).permute(1, 0, 2)
    print(f"SPQKV part {p} rc {rq} rel err {float((outs[p].float() - rp).norm() / rp.norm()):.1e}", flush=True)
