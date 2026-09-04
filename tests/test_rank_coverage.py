"""Verify the strided gather kernel across ALL supported ranks (1-5) with
wt-2026-09-04-fix: rank 1-6 stride-pattern coverage test (issue #5746)
mixed non-contiguous stride patterns, not just the 3D issue case."""
import torch
import torch_npu  # noqa: F401
import flag_gems

device = "npu:0"
PASS, FAIL = 0, 0


def check(name, inp, dim, idx):
    global PASS, FAIL
    ref = torch.gather(inp, dim, idx).cpu()
    with flag_gems.use_gems():
        res = torch.gather(inp, dim, idx).cpu()
    if torch.equal(ref, res):
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")


# --- rank 1: stride trick via expand (torch-native legal construction) ---
x = torch.randn(16, device=device, dtype=torch.float16)
i = torch.randint(0, 16, (4,), device=device, dtype=torch.int64)
i_exp = i[:1].expand(8)
check("rank1 expand", x, 0, i_exp)

# --- rank 2: transpose + expand ---
x = torch.randn(8, 32, device=device, dtype=torch.float16)
i = torch.randint(0, 32, (8, 1), device=device, dtype=torch.int64)
check("rank2 expand dim=1", x, 1, i.expand(8, 16))
i2 = torch.randint(0, 8, (32, 8), device=device, dtype=torch.int64).t()  # stride swap
check("rank2 transpose dim=0", x, 0, i2)

# --- rank 4: expand on one axis + slice on another ---
x = torch.randn(2, 8, 4, 16, device=device, dtype=torch.float16)
i = torch.randint(0, 8, (2, 1, 4, 16), device=device, dtype=torch.int64)
check("rank4 expand dim=1", x, 1, i.expand(2, 3, 4, 16))
full = torch.randint(0, 8, (2, 16, 4, 16), device=device, dtype=torch.int64)
check("rank4 slice dim=1", x, 1, full[:, ::2])
i4 = torch.randint(0, 2, (4, 2, 16, 4), device=device, dtype=torch.int64).permute(1, 0, 3, 2)
check("rank4 permute dim=0", x, 0, i4)

# --- rank 5: the kernel's max supported rank, expand middle axis ---
x = torch.randn(2, 3, 8, 4, 6, device=device, dtype=torch.float16)
i = torch.randint(0, 8, (2, 3, 1, 4, 6), device=device, dtype=torch.int64)
check("rank5 expand dim=2", x, 2, i.expand(2, 3, 5, 4, 6))

# --- mixed strides in one tensor: expand + transpose + slice together ---
x = torch.randn(4, 8, 6, device=device, dtype=torch.float16)
src = torch.randint(0, 8, (6, 4, 1), device=device, dtype=torch.int64)
i_mix = src.permute(1, 2, 0).expand(4, 3, 6)  # transpose + expand compound view
check("rank3 compound (permute+expand) dim=1", x, 1, i_mix)

print(f"\nTOTAL: {PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
