# FlagGems gather 修复交付包（issue #5746）

## 一、要改 FlagGems 中的哪个文件（明确答案）

**只改这一个文件：**

```
src/flag_gems/runtime/backend/_ascend/ops/gather.py
```

即已安装 flag_gems 包里的（site-packages 路径按实际安装位置调整）：

```
/usr/local/python3.11.15/lib/python3.11/site-packages/flag_gems/runtime/backend/_ascend/ops/gather.py
```

用本目录的 `gather.py.fixed` **整文件替换**即可（该文件其余部分与 v5.3.5 原版逐字节一致，改动只在此文件内部，不涉及任何其他文件）。

改动内容（对照原版）：
1. 新增常量 `MAX_RANK = 5`（第 26 行附近）
2. 新增 `_gather_strided_kernel` Triton kernel + `gather_strided()` 包装函数
3. `gather()` 入口函数末尾改为三路分流（连续→原 flat 路径；非连续 rank≤5→strided kernel；rank>5→先 contiguous()）

## 一b、Bug 具体在哪个函数（对照 gather.py.orig）

`gather.py.orig` = FlagGems v5.3.5 原版算子文件（从 site-packages 现场备份），结构：

| 行号（原版）| 函数 | 状态 |
|---|---|---|
| L28 | `compute_base_offset` | 未改动（继续服务 flat 路径）|
| **L45** | `_gather_flat_kernel_fixed` | **错误源头，但一字未改**（见下）|
| L67 | `gather_flat_fixed` | 未改动（连带责任：未检查 index 连续性就传给 kernel）|
| L94 | `gather()` | **入口分流点，唯一被改的原函数**（末尾 6 行 → if/else 三路）|
| L108 | `gather_backward` | 未改动（走 scatter_ 路径，实测无此 bug）|

**肇事代码**：`_gather_flat_kernel_fixed` 内 L58

```python
cur_index = tl.load(index + offset, mask=mask, other=0)   # ← 错误源头
```

该 kernel 假设 index 在内存连续平铺，用 `index + offset` 线性读取。当 index 是
`expand` 出来的非连续视图（如 stride=(1,0,0)、底层仅 4 个元素、numel=1536）时：

1. offset 越过 storage 末尾，越界读取 ~12KB 拿到垃圾值
2. `inp_offset = base + 垃圾值 × dim_stride(384)` 寻址失控
3. 对 inp 的 GM 越界访问 → 设备侧 vector core 异常 → 同步时报 507035

**Bug 函数 ≠ 被改函数（重要）**：修复没有动 `_gather_flat_kernel_fixed` 一行。
它在 index 连续时是正确的，且是 PR #1290 的性能优化成果——保留它作为快路径。
修复策略是在 `gather()` 入口把非连续 index **分流**给新增的 stride 感知 kernel：

```
gather() 入口（L94，唯一修改的原函数）
├─ index.is_contiguous()           → 原路径 gather_flat_fixed（零改动，零风险）
├─ 非连续 且 index.ndim <= 5       → 新增 gather_strided → _gather_strided_kernel
└─ 非连续 且 index.ndim > 5        → index.contiguous() 后走原 flat 路径兜底
```

新 kernel `_gather_strided_kernel` 对 index/out/inp 三组张量全部逐维计算
`偏移 = Σ coord_i × stride_i`（缺失维补 shape=1/stride=0），非连续 index 天然正确，
且顺带支持了非连续 out（实测 gap_checks.py 用例 3）。

## 二、本目录文件清单

| 文件 | 说明 |
|---|---|
| `gather.py.fixed` | 修复后的完整算子文件，直接替换用 |
| `gather.patch` | 同样改动的 diff（169 行），`git apply` 或人工核对用 |
| `test_gather.py` | 修复后的完整测试文件（原 63 用例 + 新增 9 个非连续 index 用例）|
| `repro_5746.py` | issue #5746 最小复现 + 验收脚本（修复后应打印 OK）|
| `gap_checks.py` | 边界补充验证（backward / rank-6 / 非连续 out）|
| `bench_gather.py` | 性能护栏脚本 |
| `GATHER_5746_REPORT.md` | 完整分析报告（问题/方案/实验/自查清单）|
| `README.md` | 本说明 |

## 三、部署步骤

```bash
# 1. 备份原文件
cp /usr/local/python3.11.15/lib/python3.11/site-packages/flag_gems/runtime/backend/_ascend/ops/gather.py \
   /root/gather_5746_fix/gather.py.orig

# 2. 替换
cp /root/gather_5746_fix/gather.py.fixed \
   /usr/local/python3.11.15/lib/python3.11/site-packages/flag_gems/runtime/backend/_ascend/ops/gather.py

# 3. 清理该算子的字节码缓存（重要，否则可能仍加载旧 pyc）
find /usr/local/python3.11.15/lib/python3.11/site-packages/flag_gems -name "__pycache__" -path "*_ascend*" -exec rm -rf {} + 2>/dev/null
# 若曾跑过本仓库测试，也清理仓库内缓存：
find /root/FlagGems/src -name "__pycache__" -exec rm -rf {} + 2>/dev/null
```

## 四、验证命令

```bash
cd /root/gather_5746_fix

# 1. 独立测试（推荐，零依赖，15 个用例，应输出 TOTAL: 15 passed, 0 failed）
ASCEND_LAUNCH_BLOCKING=1 python test_gather_standalone.py

# 2. issue 复现验收（应打印 OK，修复前为 507035 崩溃）
ASCEND_LAUNCH_BLOCKING=1 python repro_5746.py

# 3. 边界验证（三行均应 RAN 且 equal=True）
ASCEND_LAUNCH_BLOCKING=1 python gap_checks.py
```

**注意**：本目录的 `test_gather.py` 是 FlagGems 仓库内测试文件，依赖仓库的
`tests/accuracy_utils.py`、`tests/conftest.py` 和 pytest，**不能** `python test_gather.py`
单独运行（会报 `ImportError: attempted relative import with no known parent package`，
error.log 里的 ERR99999 就是这个的尾部输出）。要么：

- 用上面的 `test_gather_standalone.py`（等价覆盖，直接 python 运行），或
- 把 `test_gather.py` 放回 FlagGems 仓库 `tests/` 下跑：
  ```bash
  cd /root/FlagGems && ASCEND_LAUNCH_BLOCKING=1 python -m pytest tests/test_gather.py -q
  # 预期 72 passed
  ```
