"""Compile the fused relative-position attention kernel (relpos_attn.py, production config) for GB10 and save its
cubin plus launch metadata for the RelPosAttn TensorRT plugin. The head dim is baked in (RELPOS_DK, default 128).
usage: [RELPOS_DK=64] relpos_cubin.py OUT_PREFIX"""
import json, os, re, sys
import torch
import relpos_attn as ra

CFG = dict(BM=32, BN=32, num_warps=4, num_stages=2, gather=True)
H, B, T, DK = 8, 3, 63, int(os.environ.get("RELPOS_DK", "128"))         # T, B not multiples of 16 or 1: no integer specialisation baked in
q = torch.randn(H, B, T, DK, device="cuda").half()
p = torch.randn(H, 2 * T - 1, DK, device="cuda").half(); c = torch.randn(H, 2 * T - 1, device="cuda")
L = torch.full((B,), T, device="cuda", dtype=torch.int32)
ra.relpos_attention(q, q, q, p, c, L, CFG["BM"], CFG["BN"], CFG["num_warps"], CFG["num_stages"], CFG["gather"])
k = None
for dev_cache in ra._relpos_attn.device_caches.values():
    for kern in dev_cache[0].values():
        k = kern
ptx = k.asm["ptx"]
entry = re.search(r"\.entry\s+(\w+)\s*\((.*?)\)", ptx, re.S)
params = [ln.strip() for ln in entry.group(2).split(",")]
meta = {"global_scratch": getattr(k.metadata, "global_scratch_size", 0), "name": k.metadata.name, "shared": k.metadata.shared, "num_warps": k.metadata.num_warps,
        "BM": CFG["BM"], "DK": DK, "params": params}
open(sys.argv[1] + ".cubin", "wb").write(k.asm["cubin"])
json.dump(meta, open(sys.argv[1] + ".json", "w"), indent=1)
print(json.dumps(meta, indent=1))
