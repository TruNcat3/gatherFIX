# 原生 vs flag_gems 全景性能对照（2026-09-15，共享机器，两轮稳定复测）

回答的问题：**不用 flag_gems（纯原生 torch_npu）时性能差多少？什么时候值得开？**

复现脚本：各库 `tests/bench_perf.py` + `native_vs_gems*.py`（本汇总原始数据）。
口径：wall = perf_counter+sync（真实调用成本）；device = NPU events（含 launch
间隙的时间片，非纯 kernel 时长）。

## 主矩阵（4MB 级，两轮中位数）

| 算子 | 原生 wall | gems wall | wall 比值 | device 比值 |
|---|---|---|---|---|
| gelu tanh 4MB | 0.012 ms | 0.113 ms | 9.5x | 10.0x |
| add 同形 4MB | 0.010 ms | 0.125 ms | 12.7x | 11.4x |
| add 广播 | 0.010 ms | 0.150 ms | 15.6x | 14.2x |
| gather 连续 1MB fp16 | 0.122 ms | 5.41 ms | 44.4x | 44.8x |
| one_hot (4k,)×1000 | 0.026 ms | 0.125 ms | 4.8x | 4.7x |
| lift_fresh 非空 4MB | 0.0025 ms | 0.067 ms | 27.0x | 22.6x |
| lift_fresh 空张量 | 2.3 µs | 13.3 µs | 5.8x（修复前=进程崩） | — |

## 关键转折：大形状下比值大幅收敛（128MB）

| 算子 | 原生 device | gems device | device 比值 |
|---|---|---|---|
| add 128MB | 0.338 ms | 0.449 ms | **1.33x** |
| gelu tanh 128MB | 0.212 ms | 0.393 ms | **1.85x** |
| gather 17MB fp16 | ~160 ms | 160.96 ms | **1.0x** |

## 解读（修正了早期单一结论）

1. **小形状（≤4MB）的 10-44x 主要是“每调用固定开销”**，不是 kernel 慢：
   wall ≈ device 比值（几乎重合）说明每次调用存在 ~100-130µs 的 launch
   间隙（Python JIT 链 + NPU enqueue 节奏），kernel 本体只占其中一小段。
   原生路径 C++ 直连 aclnn 没有这个间隙。
2. **kernel 本体质量与原生基本持平**：大形状下（launch 间隙占比变小）
   gather 完全 1.0x、add 1.33x、gelu 1.85x——gelu 的 1.85x 是 erf/tanh
   libdevice 实现差异，属可优化项而非结构性差距。
3. **与 stage-2 分解的一致性**：plan-cache 原型消除的是 host 侧 ~34%，
   但 launch 间隙的另一半在 triton JIT `__call__`（~25µs）与 enqueue 节奏
   （见 pointwiseFIX stage2_locate：裸 triton 33µs vs gems 123µs）——
   完全收敛需要 C++ launch（torch extension / NPU graph capture）。
4. **给使用者的决策表**（当前状态，修复后口径）：
   - 大张量 compute-bound 场景（≥32MB 级）：开 gems 无明显代价
     （1.0-1.9x，gather 甚至持平），获得开源栈可修改性
   - 小张量 launch-bound 场景（attention 内层逐元素等）：gems 的每调用
     ~100µs 固定成本占比高，建议 `use_gems(include=[...])` 白名单或保持原生
   - 修复过的算子（五库覆盖面）在两种场景下 correctness 均有保证

## 与五库修复的关系

五库修复全部零性能回归（各 bench_perf.py），本矩阵补充的是"开 flag_gems
本身"的代价——它是框架现状，不是修复引入的。stage-2（host 减负）与
stage-3（形状感知 fallback）正是针对这个代价的路线。
