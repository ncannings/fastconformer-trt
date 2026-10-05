"""Sustained FP8 GEMM load for ~25 s (for clock and throttle observation)."""
import torch, time
a = (torch.randn(8192, 8192, device="cuda") * 0.1).to(torch.float8_e4m3fn); one = torch.ones((), device="cuda")
t = time.time(); n = 0
while time.time() - t < 25:
    for _ in range(20): torch._scaled_mm(a, a.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16)
    torch.cuda.synchronize(); n += 20
print("TF/s", 2 * 8192**3 * n / (time.time() - t) / 1e12)
