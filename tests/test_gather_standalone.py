"""Standalone unit test for the fixed gather op — imports the op DIRECTLY,
no torch dispatch / flag_gems registration involved.
wt-2026-09-04-fix: direct-call unit test for the stride-aware gather op (issue #5746)

This is the real "standalone" contract: point MOD at the file you want to
test (default: the deployed site-packages copy) and run.

Run:  ASCEND_LAUNCH_BLOCKING=1 python test_gather_standalone.py
"""
import os
import sys
import torch
import torch_npu  # noqa: F401  (must import before flag_gems)

# ---------------------------------------------------------------------------
# What to test: prefer the packaged fix in ../src/gather.py so this test is
# self-contained; fall back to the deployed site-packages copy.
# Override with:  MOD=/path/to/gather.py python test_gather_standalone.py
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src", "gather.py")
MOD = os.environ.get("MOD", SRC if os.path.exists(SRC) else "site-packages")

if MOD != "site-packages":
    import importlib.util
    spec = importlib.util.spec_from_file_location("gather_op_under_test", MOD)
    g = importlib.util.module_from_spec(spec)
    # the op file imports flag_gems.utils... which needs the real package
    spec.loader.exec_module(g)
else:
    import importlib
    g = importlib.import_module("flag_gems.runtime.backend._ascend.ops.gather")

print(f"testing gather op from: {getattr(g, '__file__', MOD)}")
assert hasattr(g, "gather"), "not the gather op file?"
assert hasattr(g, "gather_strided"), "this copy does NOT contain the fix!"

device = "npu:0"
PASS, FAIL = 0, 0


def check(name, inp, dim, idx):
    global PASS, FAIL
    ref = torch.gather(inp, dim, idx).cpu()      # native aclnnGather as oracle
    res = g.gather(inp, dim, idx).cpu()          # DIRECT call, no dispatch
    if torch.equal(ref, res):
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")


print("=== 1. issue #5746 exact repro ===")
B, S, H = 4, 16, 384
x = torch.randn(B, S, H, device=device, dtype=torch.float16)
idx = torch.full((B, 1, 1), S - 1, device=device, dtype=torch.int64).expand(B, 1, H)
check("expand (4,1,1)->(4,1,384), dim=1, fp16", x, 1, idx)

print("=== 2. 9-case matrix: non-contiguous index x dim ===")
b, s, h = 4, 16, 64
for dim in (0, 1, 2):
    inp = torch.randn(b, s, h, device=device, dtype=torch.float16)
    size_dim = (b, s, h)[dim]

    base_shape = list(inp.shape)
    base_shape[dim] = 1
    idx_e = torch.randint(0, size_dim, tuple(base_shape), device=device, dtype=torch.int64).expand(inp.shape)

    src = torch.randint(0, min(inp.shape), (h, b, s), device=device, dtype=torch.int64)
    perm = [1, 0, 2] if dim != 1 else [2, 1, 0]
    idx_t = src.permute(*perm)[:b, :s, :h]

    idx_s = torch.randint(0, size_dim, (b, 2 * s, h), device=device, dtype=torch.int64)[:, ::2, :]

    for name, idx in (("expand", idx_e), ("transpose", idx_t), ("slice", idx_s)):
        check(f"{name}, dim={dim}", inp, dim, idx)

print("=== 3. contiguous index (fast path regression) ===")
inp = torch.randn(8, 32, 128, device=device, dtype=torch.float32)
idx_c = torch.randint(0, 32, (8, 16, 128), device=device, dtype=torch.int64)
check("contiguous, dim=1, fp32", inp, 1, idx_c)

print("=== 4. rank-6 fallback path ===")
x6 = torch.randn(2, 3, 4, 5, 6, 7, device=device, dtype=torch.float16)
i6 = torch.full((2, 3, 4, 5, 6, 1), 3, device=device, dtype=torch.int64).expand(2, 3, 4, 5, 6, 7)
check("rank-6 expand, dim=5", x6, 5, i6)

print("=== 5. dtype coverage (expand index, dim=1) ===")
for dt in (torch.float16, torch.float32, torch.bfloat16):
    xd = torch.randn(4, 16, 64, device=device, dtype=dt)
    idd = torch.full((4, 1, 1), 15, device=device, dtype=torch.int64).expand(4, 1, 64)
    check(f"expand, dim=1, {dt}", xd, 1, idd)

print(f"\nTOTAL: {PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
