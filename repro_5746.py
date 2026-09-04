import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import torch
import torch_npu  # noqa: F401
import flag_gems

B, S, H = 4, 16, 384
device = "npu:0"
x = torch.randn(B, S, H, device=device, dtype=torch.float16)
idx = torch.full((B, 1, 1), S - 1, device=device, dtype=torch.int64).expand(B, 1, H)

ref = torch.gather(x, 1, idx).squeeze(1).cpu()

flag_gems.enable()
out = torch.gather(x, 1, idx).squeeze(1).cpu()

assert torch.equal(out, ref), f"mismatch:\n{out}\nvs\n{ref}"
print("OK: flag_gems gather matches native on non-contiguous index")
print("flag_gems from:", flag_gems.__file__)
