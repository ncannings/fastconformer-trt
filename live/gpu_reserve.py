# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Nigel Cannings. Part of fastconformer-trt (see LICENSE and NOTICE).
"""Work around GB10 unified memory on a shared box: CUDA's free memory tracks the kernel's MemFree, not MemAvailable,
and with tens of GB of page cache resident a cudaMalloc can fail (we saw "CUDA error: out of memory" moving a 2.6 GB
model to the GPU with MemAvailable at 48 GB). We do not drop caches (system-wide; it would evict other sessions'
memory-mapped models mid-timing). Instead, at start-up, the process touches HOST_GB of ordinary anonymous memory (the
kernel reclaims clean page cache to satisfy it, as for any allocation), frees it, and immediately reserves CUDA_GB in
PyTorch's caching allocator by allocating and freeing one tensor, so later allocations are served from that pool.
No NeMo code is changed. MemAvailable is checked first and the reservation is refused below FLOOR_GB.

usage as a wrapper:  python gpu_reserve.py SCRIPT.py [script args...]     (env RESERVE_HOST_GB, RESERVE_CUDA_GB)
usage in code:       gpu_reserve.reserve(host_gb, cuda_gb)
"""
from __future__ import annotations

import os
import runpy
import sys

FLOOR_GB = 17.0


def mem_available_gb() -> float:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1e6
    raise RuntimeError("MemAvailable not found")


def mem_free_gb() -> float:
    for line in open("/proc/meminfo"):
        if line.startswith("MemFree:"):
            return int(line.split()[1]) / 1e6
    raise RuntimeError("MemFree not found")


def reserve(host_gb: float = 10.0, cuda_gb: float = 8.0) -> None:
    """host_gb is an upper bound: only the shortfall of MemFree below cuda_gb + 1 GB is touched."""
    import numpy as np
    import torch
    avail = mem_available_gb()
    if avail - cuda_gb < FLOOR_GB:
        raise RuntimeError(f"MemAvailable {avail:.1f} GB too low to reserve {cuda_gb} GB (floor {FLOOR_GB} GB)")
    host_gb = min(host_gb, max(0.0, cuda_gb + 1.0 - mem_free_gb()))
    if host_gb > 0:
        a = np.ones(int(host_gb * 1e9), dtype=np.uint8)   # touch every page
        del a
    x = torch.empty(int(cuda_gb * 1e9), dtype=torch.uint8, device="cuda")
    del x                                                  # stays reserved in the caching allocator
    print(f"gpu_reserve: MemAvailable {avail:.1f} GB, touched {host_gb} GB host, reserved "
          f"{torch.cuda.memory_reserved() / 1e9:.1f} GB CUDA", flush=True)


if __name__ == "__main__":
    reserve(float(os.environ.get("RESERVE_HOST_GB", "10")), float(os.environ.get("RESERVE_CUDA_GB", "8")))
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(script, run_name="__main__")
