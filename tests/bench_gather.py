import time

import torch
import torch_npu  # noqa: F401
import flag_gems

x = torch.randn(256, 256, 64, device="npu:0", dtype=torch.float16)
idx = torch.randint(0, 256, (256, 256, 64), device="npu:0", dtype=torch.int64)
flag_gems.enable()

for _ in range(10):
    torch.gather(x, 1, idx)  # warmup
torch.npu.synchronize()

t0 = time.perf_counter()
for _ in range(50):
    torch.gather(x, 1, idx)
torch.npu.synchronize()
print(f"gather contiguous: {(time.perf_counter()-t0)/50*1000:.3f} ms/iter")
