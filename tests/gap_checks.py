import torch
import torch_npu  # noqa: F401
import flag_gems

device = "npu:0"
results = []

# --- Gap 1: backward (scatter_) with non-contiguous index ---
try:
    x = torch.randn(4, 16, 64, device=device, dtype=torch.float16, requires_grad=True)
    idx = torch.full((4, 1, 1), 15, device=device, dtype=torch.int64).expand(4, 1, 64)
    flag_gems.enable()
    out = torch.gather(x, 1, idx)
    g = torch.randn_like(out)
    (grad,) = torch.autograd.grad(out, x, g)
    torch.npu.synchronize()
    results.append(("backward w/ expanded index", "RAN, no crash"))
except Exception as e:
    results.append(("backward w/ expanded index", f"FAILED: {type(e).__name__}: {str(e)[:120]}"))

# --- Gap 2: rank > 5 non-contiguous index (contiguous() fallback path) ---
try:
    x6 = torch.randn(2, 3, 4, 5, 6, 7, device=device, dtype=torch.float16)
    i6 = torch.full((2, 3, 4, 5, 6, 1), 3, device=device, dtype=torch.int64).expand(
        2, 3, 4, 5, 6, 7
    )
    assert i6.ndim == 6 and not i6.is_contiguous()
    ref6 = torch.gather(x6, 5, i6).cpu()  # native, before enable
    flag_gems.enable()
    res6 = torch.gather(x6, 5, i6).cpu()
    torch.npu.synchronize()
    ok = torch.equal(ref6, res6)
    results.append(("rank-6 expanded index (fallback path)", f"RAN, equal={ok}"))
except Exception as e:
    results.append(("rank-6 expanded index (fallback path)", f"FAILED: {type(e).__name__}: {str(e)[:120]}"))

# --- Gap 3: user-supplied non-contiguous out ---
try:
    flag_gems.enable()
    x = torch.randn(4, 16, 64, device=device, dtype=torch.float16)
    idx = torch.full((4, 1, 1), 15, device=device, dtype=torch.int64).expand(4, 1, 64)
    buf = torch.zeros(4, 64, 1, device=device, dtype=torch.float16).permute(0, 2, 1)  # (4,1,64) non-contig
    torch.gather(x, 1, idx, out=buf)
    torch.npu.synchronize()
    ref = torch.gather(x, 1, idx)
    ok = torch.equal(buf.cpu(), ref.cpu())
    results.append(("explicit non-contiguous out", f"RAN, equal={ok}"))
except Exception as e:
    results.append(("explicit non-contiguous out", f"FAILED: {type(e).__name__}: {str(e)[:120]}"))

for name, r in results:
    print(f"[{name}] {r}")
