"""A 组异常定性: 用 NPU event 计设备时间, 消除 host 异步假象。"""
import torch
import torch_npu
import flag_gems

DEV = "npu:0"
B, S, H = 64, 256, 512
x = torch.randn(B, S, H, device=DEV, dtype=torch.float16)
idx_c = torch.randint(0, S, (B, S, H), device=DEV, dtype=torch.int64)

flag_gems.enable()

def event_time(fn, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) / iters  # ms

t_native_c = event_time(lambda: torch.gather(x, 1, idx_c))
t_gems_c = event_time(lambda: torch.gather(x, 1, idx_c))
idx_nc = torch.full((B, 1, 1), S - 1, device=DEV, dtype=torch.int64).expand(B, 1, H)
t_gems_nc = event_time(lambda: torch.gather(x, 1, idx_nc))
t_work = event_time(lambda: torch.gather(x, 1, idx_nc.contiguous()))

print(f"[device-time, NPU events] 连续 index: gems {t_gems_c:.3f} ms")
print(f"[device-time] 非连续 strided: {t_gems_nc:.3f} ms | 物化 workaround: {t_work:.3f} ms")
