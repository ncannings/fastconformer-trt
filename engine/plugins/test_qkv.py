"""qkv_heads_run vs torch: heads-first q_u, k, v from one FP8 input, plus speed against the plain GEMM + permute."""
import ctypes, os, time
import torch
lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libffn_fp8.so"))
lib.qkv_heads_run.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 4 + [ctypes.c_void_p, ctypes.c_void_p]
B, T, K, H, dk = 32, 200, 1024, 8, 128
M = B * T
x = torch.randn(M, K, device="cuda"); w = torch.randn(3 * H * dk, K, device="cuda") * 0.03
bias = torch.randn(3 * H * dk, device="cuda") * 0.1
sx, sw = float(x.abs().max() / 448), float(w.abs().max() / 448)
x8, w8 = (x / sx).to(torch.float8_e4m3fn), (w / sw).to(torch.float8_e4m3fn)
ref = (x8.float() @ w8.float().t()) * sx * sw + bias                         # [M, 3*H*dk]
ref = ref.view(M, 3, H, dk).permute(1, 2, 0, 3)                              # [3, H, M, dk]
alpha3 = torch.tensor([sx * sw] * 3, device="cuda").cpu()
outs = [torch.empty(H, M, dk, device="cuda", dtype=torch.float16) for _ in range(3)]
st = torch.cuda.current_stream().cuda_stream
run = lambda: lib.qkv_heads_run(x8.data_ptr(), w8.data_ptr(), bias.data_ptr(), outs[0].data_ptr(), outs[1].data_ptr(),
                                outs[2].data_ptr(), M, K, H, dk, alpha3.data_ptr(), st)
rc = run(); torch.cuda.synchronize(); print("rc", rc)
for i, n in enumerate("qkv"):
    print(n, "rel err", float((outs[i].float() - ref[i]).norm() / ref[i].norm()))
one = torch.ones((), device="cuda")
def plain():
    y = torch._scaled_mm(x8, w8.t(), scale_a=one * sx, scale_b=one * sw, out_dtype=torch.float16, bias=bias.half())
    return y.view(M, 3, H, dk).permute(1, 2, 0, 3).contiguous()
def bench(f, n=50):
    for _ in range(5): f()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.time() - t) / n * 1e3
print(f"heads-first GEMM {bench(run):.3f} ms | cuBLAS GEMM + permute {bench(plain):.3f} ms | cuBLAS GEMM alone "
      f"{bench(lambda: torch._scaled_mm(x8, w8.t(), scale_a=one, scale_b=one, out_dtype=torch.float16)):.3f} ms")
