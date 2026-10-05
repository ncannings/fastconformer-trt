"""Times the production kernels at the encoder's shapes for the variant chosen by the environment (FFN_V1, FFN_VR,
SPL_VR, QKV_V; read once per process, so sweep with one process per setting) and checks them against torch.
Prints one JSON line. usage: FFN_V1=8 FFN_VR=10 python tune_kernels.py [ffn|lin|qkv]"""
import ctypes, json, os, sys, time
import torch
lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libffn_fp8.so"))
st = torch.cuda.current_stream().cuda_stream
M, D, F = 6400, 1024, 4096
torch.manual_seed(0)
def bench(f, n=50):
    for _ in range(5): f()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.time() - t) / n * 1e3
q8 = lambda t, s: (t / s).to(torch.float8_e4m3fn)
what = sys.argv[1] if len(sys.argv) > 1 else "ffn"
out = {"what": what, **{k: os.environ.get(k) for k in ("FFN_V1", "FFN_VR", "SPL_VR", "QKV_V")}}
x = torch.randn(M, D, device="cuda"); res = torch.randn(M, D, device="cuda").half()
sx = float(x.abs().max() / 448); x8 = q8(x, sx)
if what == "ffn":
    lib.ffn_fp8_run_res.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 4 + [ctypes.c_float] * 4 + [ctypes.c_int, ctypes.c_void_p]
    w1 = torch.randn(F, D, device="cuda") * 0.03; w2 = torch.randn(D, F, device="cuda") * 0.02
    s1, s2 = float(w1.abs().max() / 448), float(w2.abs().max() / 448); w18, w28 = q8(w1, s1), q8(w2, s2)
    h = torch.nn.functional.silu((x8.float() @ w18.float().t()) * sx * s1); sh = float(h.abs().max() / 448)
    ref = res.float() + 0.5 * (q8(h, sh).float() @ w28.float().t()) * sh * s2
    h8 = torch.empty(M, F, device="cuda", dtype=torch.float8_e4m3fn); y = torch.empty(M, D, device="cuda", dtype=torch.float16)
    run = lambda: lib.ffn_fp8_run_res(x8.data_ptr(), h8.data_ptr(), res.data_ptr(), y.data_ptr(), w18.data_ptr(), w28.data_ptr(), M, D, F, D, sx * s1, 1 / sh, sh * s2, 0.5, 0, st)
    flops = 2 * 2 * M * D * F
elif what == "lin":
    lib.fp8_linear_res_run.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 3 + [ctypes.c_float] * 2 + [ctypes.c_int, ctypes.c_void_p]
    w = torch.randn(D, D, device="cuda") * 0.03; sw = float(w.abs().max() / 448); w8 = q8(w, sw)
    ref = res.float() + (x8.float() @ w8.float().t()) * sx * sw
    y = torch.empty(M, D, device="cuda", dtype=torch.float16)
    run = lambda: lib.fp8_linear_res_run(x8.data_ptr(), w8.data_ptr(), res.data_ptr(), y.data_ptr(), M, D, D, sx * sw, 1.0, 0, st)
    flops = 2 * M * D * D
else:
    lib.qkv_heads_run.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 4 + [ctypes.c_void_p, ctypes.c_void_p]
    H, dk = 8, 128
    w = torch.randn(3 * D, D, device="cuda") * 0.03; b = torch.randn(3 * D, device="cuda") * 0.1
    sw = float(w.abs().max() / 448); w8 = q8(w, sw)
    full = ((x8.float() @ w8.float().t()) * sx * sw + b).view(M, 3, H, dk).permute(1, 2, 0, 3)
    a3 = torch.tensor([sx * sw] * 3); outs = [torch.empty(H, M, dk, device="cuda", dtype=torch.float16) for _ in range(3)]
    run = lambda: lib.qkv_heads_run(x8.data_ptr(), w8.data_ptr(), b.data_ptr(), outs[0].data_ptr(), outs[1].data_ptr(), outs[2].data_ptr(), M, D, H, dk, a3.data_ptr(), st)
    flops = 2 * M * D * 3 * D
rc = run(); torch.cuda.synchronize(); out["rc"] = rc
if rc == 0:
    y_ = torch.stack(outs) if what == "qkv" else y
    r_ = full if what == "qkv" else ref
    out["rel_err"] = float((y_.float() - r_).norm() / r_.norm())
    ms = bench(run); out["ms"] = round(ms, 4); out["tflops"] = round(flops / ms / 1e9, 1)
print(json.dumps(out), flush=True)
