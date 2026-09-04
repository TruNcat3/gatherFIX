"""Standalone gather test for issue #5746 — zero dependencies beyond torch/torch_npu/flag_gems.

Run:  ASCEND_LAUNCH_BLOCKING=1 python test_gather_standalone.py
Ref:  native torch.gather (flag_gems disabled), compared bitwise per-case.

Covers:
  - issue #5746 exact repro (expand index)
  - 9-case matrix: expand / transpose / slice  x  dim 0/1/2
  - contiguous index sanity (must still hit the fast path correctly)
  - rank-6 fallback path
  - fp16 / fp32 / bf16 dtypes
"""
import torch
import torch_npu  # noqa: F401
import flag_gems

device = "npu:0"
PASS, FAIL = 0, 0


def check(name, inp, dim, idx, dtype):
    global PASS, FAIL
    ref = torch.gather(inp, dim, idx).cpu()  # native impl (flag_gems off at call site)
    with flag_gems.use_gems():
        res = torch.gather(inp, dim, idx).cpu()
    if torch.equal(ref, res):
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}: max abs diff = {(res.float() - ref.float()).abs().max().item()}")


print("=== 1. issue #5746 exact repro ===")
B, S, H = 4, 16, 384
x = torch.randn(B, S, H, device=device, dtype=torch.float16)
idx = torch.full((B, 1, 1), S - 1, device=device, dtype=torch.int64).expand(B, 1, H)
check("expand (4,1,1)->(4,1,384), dim=1, fp16", x, 1, idx, torch.float16)

print("=== 2. 9-case matrix: non-contiguous index x dim ===")
b, s, h = 4, 16, 64
for dim in (0, 1, 2):
    inp = torch.randn(b, s, h, device=device, dtype=torch.float16)
    size_dim = (b, s, h)[dim]

    # expand: singleton along dim broadcast out
    base_shape = list(inp.shape)
    base_shape[dim] = 1
    idx_e = torch.randint(0, size_dim, tuple(base_shape), device=device, dtype=torch.int64).expand(inp.shape)

    # transpose: permuted view (satisfies idx.shape[i] <= inp.shape[i] for i != dim)
    src = torch.randint(0, min(inp.shape), (h, b, s), device=device, dtype=torch.int64)
    perm = [1, 0, 2] if dim != 1 else [2, 1, 0]
    idx_t = src.permute(*perm)[:b, :s, :h]

    # slice: stride (2h, h, 1)
    idx_s = torch.randint(0, size_dim, (b, 2 * s, h), device=device, dtype=torch.int64)[:, ::2, :]

    for name, idx in (("expand", idx_e), ("transpose", idx_t), ("slice", idx_s)):
        check(f"{name}, dim={dim}", inp, dim, idx, torch.float16)

print("=== 3. contiguous index (fast path regression) ===")
inp = torch.randn(8, 32, 128, device=device, dtype=torch.float32)
idx_c = torch.randint(0, 32, (8, 16, 128), device=device, dtype=torch.int64)
check("contiguous, dim=1, fp32", inp, 1, idx_c, torch.float32)

print("=== 4. rank-6 fallback path ===")
x6 = torch.randn(2, 3, 4, 5, 6, 7, device=device, dtype=torch.float16)
i6 = torch.full((2, 3, 4, 5, 6, 1), 3, device=device, dtype=torch.int64).expand(2, 3, 4, 5, 6, 7)
check("rank-6 expand, dim=5", x6, 5, i6, torch.float16)

print("=== 5. dtype coverage (expand index, dim=1) ===")
for dt in (torch.float16, torch.float32, torch.bfloat16):
    xd = torch.randn(4, 16, 64, device=device, dtype=dt)
    idd = torch.full((4, 1, 1), 15, device=device, dtype=torch.int64).expand(4, 1, 64)
    check(f"expand, dim=1, {dt}", xd, 1, idd, dt)

print(f"\nTOTAL: {PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
