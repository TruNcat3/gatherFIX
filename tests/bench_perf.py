"""gatherFIX 性能对照实验 (#5746) — bench_gather.py 的对照版.

修复语义: 非连续 index 分流到 stride-aware kernel; 连续路径零改动。
性能问题域:
  A. 连续 index (快路径): gems vs 原生 —— 修复零改动区, 差距=框架既有
  B. 非连续 index (修复核心): 修复前=崩溃; 修复后 vs "物化 contiguous 兜底"
     (修复前可行的 workaround) —— 修复路径应显著快于 workaround
"""
import statistics
import time

import torch
import torch_npu
import flag_gems

DEV = "npu:0"


def bench(fn, iters=50, warmup=10, rounds=5):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    ts = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.npu.synchronize()
        ts.append((time.perf_counter() - t0) / iters * 1e3)
    return statistics.median(ts)


# 原生基线 (enable 前)
B, S, H = 64, 256, 512
x = torch.randn(B, S, H, device=DEV, dtype=torch.float16)
idx_c = torch.randint(0, S, (B, S, H), device=DEV, dtype=torch.int64)
idx_nc = torch.full((B, 1, 1), S - 1, device=DEV, dtype=torch.int64).expand(B, 1, H)
t_native_c = bench(lambda: torch.gather(x, 1, idx_c))
t_native_nc = bench(lambda: torch.gather(x, 1, idx_nc.contiguous()))  # 原生对非连续也是先物化? 测其行为

flag_gems.enable()

print("=" * 70)
print("gather 性能对照 (fp16, B64xS256xH512)")
print("=" * 70)

# A. 连续 index (修复零改动区)
r1 = torch.gather(x, 1, idx_c)
t_gems_c = bench(lambda: torch.gather(x, 1, idx_c))
print(f"A. 连续 index:   原生 {t_native_c:8.3f} ms | gems {t_gems_c:8.3f} ms  ({t_gems_c/t_native_c:.2f}x)")

# B. 非连续 index: 修复后 strided kernel vs workaround (先 contiguous 再 gather)
r2 = torch.gather(x, 1, idx_nc)  # 修复后: strided 路径
ref = torch.gather(x, 1, idx_nc.contiguous())
torch.npu.synchronize()
ok = torch.equal(r2, ref)
t_gems_nc = bench(lambda: torch.gather(x, 1, idx_nc))
t_workaround = bench(lambda: torch.gather(x, 1, idx_nc.contiguous()))
t_native_nc2 = bench(lambda: torch.gather(x, 1, idx_nc.contiguous()))
print(f"B. 非连续 index (数值一致={ok}):")
print(f"   修复后 strided kernel: {t_gems_nc:8.3f} ms")
print(f"   workaround (先物化):   {t_workaround:8.3f} ms  (修复路径 {'快' if t_gems_nc < t_workaround else '慢'} {abs(t_workaround/t_gems_nc-1)*100:.0f}%)")
print(f"   原生物化对照:          {t_native_nc2:8.3f} ms")
print("\n判读: A 组比值即框架既有差距(与本修复无关); B 组 strided 路径避免物化拷贝,")
print("      应显著快于 workaround —— 这正是修复的性能价值所在。")
