# FlagGems #5746：Ascend 后端 gather 非连续 index 越界问题 — 分析与修复报告

- **Issue**: [flagos-ai/FlagGems#5746](https://github.com/flagos-ai/FlagGems/issues/5746)
- **代码基线**: FlagGems v5.3.5（master 分支该文件与 v5.3.5 逐字节一致，已 diff 确认）
- **修复仓库**: `/root/FlagGems`（v5.3.5 + 2 commits）
- **实验环境**: Ascend 910（8 卡）、CANN/npu-smi 25.5.0、torch 2.10.0+cpu、torch_npu 2.10.0、Python 3.11
- **报告日期**: 2026-09-01

---

## 1. 问题

### 1.1 现象

开启 flag_gems 后，`torch.gather` 的 `index` 参数存储非连续时设备侧崩溃：

```python
import torch
import torch_npu
import flag_gems

B, S, H = 4, 16, 384
device = 'npu:0'
x = torch.randn(B, S, H, device=device, dtype=torch.float16)
idx = torch.full((B, 1, 1), S - 1, device=device, dtype=torch.int64).expand(B, 1, H)
out = torch.gather(x, 1, idx).squeeze(1)   # 原生正常
flag_gems.enable()
out = torch.gather(x, 1, idx).squeeze(1)   # 崩溃
```

报错（`ASCEND_LAUNCH_BLOCKING=1`）：

```
RuntimeError: NPU function error: c10_npu::acl::AclrtSynchronizeStreamWithTimeout(stream),
error code is 507035
[Error]: The vector core execution is abnormal.
```

### 1.2 根因（源码级定位）

调用链：`torch.gather` → flag_gems Ascend 后端覆写
`src/flag_gems/runtime/backend/_ascend/ops/gather.py`（经 `ops/__init__.py` L50 注册）→
`gather()` → `gather_flat_fixed()` → `_gather_flat_kernel_fixed`。

问题 kernel 代码：

```python
cur_index = tl.load(index + offset, mask=mask, other=0)   # ← 忽略 index 的 stride
inp_offset = base + cur_index * inp_dim_stride
val = tl.load(inp + inp_offset, mask=mask, other=0)
```

`index + offset` 把 index 当作扁平连续数组线性读取，完全无视其 stride。

以复现用例量化（shape (4,1,384)，stride (1,0,0)，底层 storage 仅 4 个 int64）：

| 量 | 值 |
|---|---|
| index numel（kernel 读取范围）| 1536 |
| 底层 storage 元素数 | 4 |
| 越界读取 | 1532 个 int64 ≈ 12 KB |
| dim_stride（inp.stride(1)）| 384 |

垃圾索引值 × 384 → 对 inp 的 GM 越界访问 → 设备侧 AI/vector core 异常 →
同步时上报 507035。因设备异步执行，Python traceback 落在无关行，容易误判层次。

### 1.3 层次结论

| 层 | 是否有问题 | 依据 |
|---|---|---|
| flag_gems 通用实现（`src/flag_gems/ops/gather.py`）| **无** | codegen kernel 逐维计算 `index_offset = Σ coord_i·stride_i`，正确处理非连续 index |
| flag_gems **Ascend 后端覆写**（`_ascend/ops/gather.py`）| **有（本 bug）** | flat 线性读取，无视 stride |
| `gather_collapsed*.py` | 无 | `_collapsed_3d_views` 已做 `index.contiguous()` |
| `gather_ascend.py`（codegen 路径）| 有同模式 bug | 但 `gather_dispatch.gather_auto` 在 src/ 与 tests/ 无任何调用方，**当前不可达死代码** |
| torch_npu / CANN | 无 | 507035 只是设备侧异常的同步上报，非肇事方 |

---

## 2. 解决方案

单文件修改：`src/flag_gems/runtime/backend/_ascend/ops/gather.py`。
入口 `gather()` 按三路分流：

```
gather(inp, dim, index, out)
├─ index 连续                → gather_flat_fixed（原路径，代码零改动，零性能回归）
├─ index 非连续 且 rank ≤ 5   → gather_strided（新增 _gather_strided_kernel）
└─ index 非连续 且 rank > 5   → index.contiguous() 后走 flat 路径兜底
```

### 2.1 新 kernel 设计要点

- **静态签名，逐维传参**：Triton kernel 签名静态，无法运行时循环 rank。支持 rank ≤ 5，
  缺失维 host 侧补 shape=1 / stride=0（该维坐标恒 0，贡献恒 0）。
- **三组偏移逐维计算**：index/out/inp 各自 `Σ coord_i × stride_i`，非连续张量天然正确。
- **inp 侧沿用 restride_dim 技巧**：`restride_dim(inp, dim, index.shape)` 把 inp 沿 dim 的
  stride 置 0，使所有维统一累加；gather 索引值单独乘**原始** `inp.stride(dim)`（顺序不可换）。
- **out 支持非连续**：out stride 逐维传入，调用方显式传非连续 out 也正确。

### 2.2 改动文件

| 文件 | 改动 |
|---|---|
| `src/flag_gems/runtime/backend/_ascend/ops/gather.py` | +`MAX_RANK` 常量、+`_gather_strided_kernel`、+`gather_strided()`、`gather()` 分流 |
| `tests/test_gather.py` | +`test_gather_non_contiguous_index`（expand/transpose/slice × dim 0/1/2，共 9 用例）|
| `conftest.py`（新增）| 仓库根注入 src/ 到 sys.path，保证测试跑本地源码树而非 site-packages |
| `repro_5746.py`（新增）| issue 复现脚本，改后作为验收 |
| `gap_checks.py` / `bench_gather.py`（新增，未提交）| 补充验证与性能脚本 |

Commits：

```
eec2d2b fix(ascend): stride-aware gather kernel for non-contiguous index (#5746)
b60c3f4 test: add non-contiguous index regression for gather (#5746), currently red
```

---

## 3. 实验结果（全部真实 NPU 执行）

| # | 验证项 | 命令 / 方式 | 结果 |
|---|---|---|---|
| 1 | 基线复现（修复前）| `ASCEND_LAUNCH_BLOCKING=1 python repro_5746.py` | 507035 vector core exception，与 issue 一致 |
| 2 | 红测试（修复前）| `pytest tests/test_gather.py -k non_contiguous -q` | **9/9 失败**（全部 507035）|
| 3 | 绿测试（修复后）| 同上 | **9/9 通过** |
| 4 | issue 复现脚本（修复后）| `repro_5746.py` | `torch.equal` 与原生输出**完全一致** |
| 5 | 全量 gather 套件 | `pytest tests/test_gather.py -q` | **72 passed / 0 failed**（原 63 + 新 9）|
| 6 | backward + expand index | `gap_checks.py` | 正常，无崩溃（scatter_ 路径无此 bug）|
| 7 | rank-6 expand index（fallback 路径）| `gap_checks.py` | 结果与原生一致 |
| 8 | 显式非连续 out | `gap_checks.py` | 结果正确（strided kernel 天然支持）|
| 9 | 连续路径性能护栏 | `bench_gather.py`，(256,256,64) fp16，50 iter × 3 轮 | 修复后 79.7–112.6 ms vs 基线 104.5–109.5 ms，组间差异 < 组内波动（见 §4 近似声明）|

---

## 4. 诚实自查：模糊 / 近似 / 跳过 / 暂缓清单

### ✅ 已闭环（发现 gap 后补实验证实）

1. **backward 路径**：初始未测 → 补测通过，`scatter_` 对 expand index 正常。
2. **rank>5 fallback 正确性**：初始未测 → 补测 rank-6 用例与原生一致。
3. **非连续 out**：初始未测 → 补测正确。

### ⚠️ 已声明的近似（明确标记，非隐藏）

4. **性能护栏数据质量**：单机非独占（机器有外部负载）、单进程对比、组内波动 ±40% 远大于
   组间差异。严格结论是"差异淹没在噪声里"，**不是**精确的"无回归"。逻辑上连续路径代码零改动
   （仅多一次 host 侧 O(1) `is_contiguous()` 判断），无回归机制。若需严格数据需独占机器多轮取中位数。
5. **性能测试形状**：bench 只测了 (256,256,64)（1M 元素）。PR #1290 优化时的大形状
   (4096,4096,1024) 因超出单卡 HBM（fp16 需 32GB + index 32GB = 64GB > 61GB）与
   `randint` 大张量卡死，未完成，标记为**未覆盖**。

### 🚫 有意跳过（范围决策，理由明确）

6. **`gather_ascend.py` codegen 路径的同模式 bug**：grep 证实 `gather_dispatch.gather_auto`
   在 src/ 与 tests/ 无任何调用方，是不可达死代码。决策：不修（修不可达代码无意义），
   非"暂缓"。
7. **上游同步**：commit 就绪但未推送/未提 PR——需要 GitHub 凭据，属外部动作，待指令。

### ⏳ 暂缓（超出本 issue 范围，建议上游跟进）

8. **其他 index 类算子同模式 audit**：`index_fill.py` / `index_add.py` / `index.py` /
   `index_copy_.py` 存在按偏移直读 index 的写法（grep 已列出可疑行号），但其调用点是否
   会收到非连续 index **未逐一验证**。建议上游单独排查。
9. **`gather_flat_fixed` 对非连续 out 的假设**：本次修复范围仅 index 侧（issue 范围）；
   实测 strided 路径天然支持非连续 out（§3 #8），但 flat 路径若被 `gather.out` 变体传入
   非连续 out 仍可能有假设问题——现有测试未覆盖该组合。

### ❌ 过程中修正过的错误（记录以防复现误导）

10. bench 第一版形状 (4096,4096,1024) OOM；第二版 `randint` 大张量卡死被杀。最终缩小形状完成。
11. gap 脚本第一版调用了不存在的 `flag_gems.disable()`，修正为"enable 前算参考值"后重跑。

### 边界说明（设计约束，非偷懒）

- **rank ≤ 5 限制**：Triton 静态签名的必然代价；rank>5 走 `contiguous()` 兜底，正确性不牺牲，
  仅多一次拷贝（罕见场景）。
- **测试跑本地源码树的方式**：通过仓库根 `conftest.py` 注入 `src/` 到 sys.path，
  site-packages 中的 flag_gems 5.3.5 未改动。部署时需 `pip install -e .` 或直接替换文件。

---

## 5. 复现 / 验证命令速查

```bash
cd /root/FlagGems

# issue 复现 + 验收（修复后应打印 OK）
ASCEND_LAUNCH_BLOCKING=1 python repro_5746.py

# 回归测试
ASCEND_LAUNCH_BLOCKING=1 python -m pytest tests/test_gather.py -k non_contiguous -q   # 9 passed
ASCEND_LAUNCH_BLOCKING=1 python -m pytest tests/test_gather.py -q                     # 72 passed

# 边界补充验证
ASCEND_LAUNCH_BLOCKING=1 python gap_checks.py

# 性能护栏
python bench_gather.py
```

## 6. 建议后续

1. 将 2 个 commit 整理为 PR 提交上游 flagos-ai/FlagGems，引用 issue #5746
   （master 与 v5.3.5 该文件一致，可干净 rebase）。
2. 上游排查 §4.8 所列 index 类算子的非连续 index 假设。
3. 考虑在 flag_gems 测试基础设施中加入"非连续 index"的参数化维度
   （本次已在 `test_gather.py` 落地 9 个用例，可推广）。
