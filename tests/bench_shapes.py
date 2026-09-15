"""小形状对照: 排除大张量下 gems gather 连续路径的特殊慢。"""
import statistics, time
import torch, torch_npu, flag_gems

DEV = "npu:0"
def bench(fn, iters=50, warmup=10, rounds=5):
    for _ in range(warmup): fn()
    torch.npu.synchronize()
    ts = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(iters): fn()
        torch.npu.synchronize()
        ts.append((time.perf_counter()-t0)/iters*1e3)
    return statistics.median(ts)

for B, S, H in [(4, 16, 384), (16, 64, 256), (64, 256, 512)]:
    x = torch.randn(B, S, H, device=DEV, dtype=torch.float16)
    idx = torch.randint(0, S, (B, S, H), device=DEV, dtype=torch.int64)
    t_nat = bench(lambda: torch.gather(x, 1, idx))
    flag_gems.enable()
    t_gems = bench(lambda: torch.gather(x, 1, idx))
    mb = B*S*H*2/1e6
    print(f"[{B}x{S}x{H} {mb:.0f}MB] 原生 {t_nat:7.3f} ms | gems {t_gems:7.3f} ms ({t_gems/t_nat:.1f}x)")
