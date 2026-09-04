# gatherFIX — FlagGems Ascend gather 算子非连续 index 越界修复

修复 [FlagGems issue #5746](https://github.com/flagos-ai/FlagGems/issues/5746)：在昇腾 NPU 上开启 flag_gems 后，`torch.gather` 传入非连续存储的 `index`（如 `expand` 出来的视图）会导致设备侧越界访问，进程崩溃。

一句话版本：**原 kernel 假设 index 在内存里连续平铺，我们写了一个尊重 stride 的新 kernel，非连续 index 分流过去，连续路径原样保留。**

```
修复前: RuntimeError ... error code is 507035 (vector core exception)
修复后: 输出与原生 torch.gather 逐位一致
```

## 问题是怎么回事

issue 里的最小复现：

```python
x = torch.randn(4, 16, 384, device="npu:0", dtype=torch.float16)
idx = torch.full((4, 1, 1), 15, device="npu:0", dtype=torch.int64).expand(4, 1, 384)
torch.gather(x, 1, idx)   # 崩溃
```

`idx` 是 expand 出来的视图：逻辑上是 (4,1,384) 共 1536 个元素，底层 storage 只有 4 个。
flag_gems 的 Ascend 覆写 kernel（`_gather_flat_kernel_fixed`）里有一行：

```python
cur_index = tl.load(index + offset, mask=mask, other=0)
```

`index + offset` 把 index 当连续数组线性读——于是 kernel 一路读到 storage 之外 12KB，
拿到的内存垃圾被当作 gather 索引，乘上 stride 去寻址输入张量，直接打出 GM 越界。
设备侧异常是异步上报的，所以 Python 端看到的报错（507035）位置和肇事点对不上，
这也是这个 bug 最初不好定位层次的原因（bug 在 flag_gems kernel 层，不在 torch_npu/CANN）。

## 修复思路

肇事函数 `_gather_flat_kernel_fixed` 对连续 index 是**正确的**，而且是社区性能优化
（PR #1290）的成果，所以我们一行没动它。改的是入口 `gather()`，加了一次分流：

```
index.is_contiguous()？ ── 是 ──→ 原路径（零改动，连续场景零风险零回归）
       │
       否，rank ≤ 5 ──────→ 新 kernel：每维传 shape/stride，
       │                    偏移 = Σ coord_i × stride_i，逐维算
       否，rank > 5 ──────→ index.contiguous() 拷贝后走原路径兜底
```

新 kernel 的正确性只依赖"线性坐标逐维分解"这个恒等式，与维度数无关——
rank 1 到 6、expand/transpose/slice/permute 及其复合视图都实测通过
（见 `tests/test_rank_coverage.py`）。rank ≤ 5 是 Triton 静态签名的工程限制，
rank > 5 走拷贝兜底，正确性不受影响。

## 怎么用

前置：昇腾环境（本修复在 Ascend 910 + CANN 8.5 / torch 2.10 / torch_npu 2.10
+ flag_gems 5.3.5 上验证）。

**部署**（唯一要动的文件是 flag_gems 包里的这一个）：

```bash
# 1. 备份并替换：src/gather.py 就是修复后的完整文件，直接替换 flag_gems 包内同名文件
cp /path/to/site-packages/flag_gems/runtime/backend/_ascend/ops/gather.py src/gather.py.orig.bak
cp src/gather.py /path/to/site-packages/flag_gems/runtime/backend/_ascend/ops/gather.py

# 2. 清掉旧字节码缓存，否则可能仍加载旧 .pyc
find /path/to/site-packages/flag_gems/runtime/backend/_ascend -name __pycache__ -exec rm -rf {} +
```

不想手动替换的话，`gather.patch` 是同样的改动，可以在 FlagGems 仓库里 `git apply gather.patch`。

**验证**（都在本目录，零依赖，直接 python 跑）：

```bash
ASCEND_LAUNCH_BLOCKING=1 python tests/repro_5746.py          # issue 原复现，应打印 OK
ASCEND_LAUNCH_BLOCKING=1 python tests/test_gather_standalone.py  # 15 用例，TOTAL: 15 passed
ASCEND_LAUNCH_BLOCKING=1 python tests/test_rank_coverage.py  # rank 1-6 全覆盖，8 passed
ASCEND_LAUNCH_BLOCKING=1 python tests/gap_checks.py          # backward / 非连续 out / rank>5
```

**端到端**（真实 transformers 模型，需先下载 tiny-random-BertModel 到本地缓存）：

```bash
ASCEND_LAUNCH_BLOCKING=1 HF_HUB_OFFLINE=1 python tests/test_model_e2e.py
# 预期: gather op invoked: 1 times / hidden states bitwise equal / grads allclose / PASS
```

这跑的是真实 `BertModel` 前向+反向：不传 `token_type_ids` 时（推理引擎和 torch.export
tracing 的常见输入），`BertEmbeddings` 内部用 `torch.gather(expand(buffer), 1, position_ids)`
重建 segment id——模型内的真实 gather 调用，由修复后的算子接管执行，输出与原生逐位一致。

## 目录结构

```
├── README.md                    # 本文
├── gather.patch                 # 最小 diff（169 行），git apply 用
├── src/
│   ├── gather.py                # 修复后的完整算子文件，直接替换 flag_gems 包内同名文件
│   └── gather.py.orig           # v5.3.5 原版备份（对照/回滚用）
├── tests/                       # 全部零依赖，直接 python 运行
│   ├── repro_5746.py            #   issue 复现 + 修复验收
│   ├── test_gather_standalone.py#   主测试：15 用例（9 非连续矩阵 + 快路径 + dtype）
│   ├── test_rank_coverage.py    #   rank 1-6 全维度验证
│   ├── gap_checks.py            #   backward / 非连续 out / rank>5 边界
│   ├── test_model_e2e.py        #   端到端：真实 BERT 前向+反向（transformers）
│   ├── bench_gather.py          #   连续路径性能护栏
│   └── test_gather.py           #   FlagGems 仓库版 pytest 套件（需放回仓库 tests/ 下跑）
└── docs/
    └── GATHER_5746_REPORT.md    # 完整分析报告：根因、层次定位、实验矩阵、自查清单
```

## 如果你想深究 bug 在哪一行

`src/gather.py.orig`（v5.3.5 原版）的函数地图：

| 原版行号 | 函数 | 与本次修复的关系 |
|---|---|---|
| L58 | `_gather_flat_kernel_fixed` 内 `tl.load(index + offset, ...)` | **肇事行**，但一字未改（它对连续 index 是对的） |
| L67 | `gather_flat_fixed` | 未改（连带责任：没检查 index 连续性就传 kernel） |
| L94 | `gather()` | **唯一被改的原函数**：入口加三路分流 |
| L28 / L108 | `compute_base_offset` / `gather_backward` | 未改（后者走 scatter_ 路径，实测无此 bug） |

完整的失败链条、逐项实验数据和"哪些验证做了/哪些明确没做"的诚实清单，
见 `docs/GATHER_5746_REPORT.md`。

## 已知边界（不装完美）

- 性能护栏只在小形状上测过，且机器有外部负载，结论是"差异淹没在噪声里"而非精确回归数据
- FlagGems 里其他 index 类算子（`index_fill`/`index_add` 等）存在同模式的按偏移直读写法，
  未逐一审计——建议上游单独排查
- 上游 FlagGems master 该文件与 v5.3.5 一致，本修复可直接整理成 PR 提交
- 端到端测试里模型触发的 gather 调用恰好是连续布局（non-contig 0 次）——e2e 证明了
  "修复版算子在真实模型里工作正常"，非连续布局的正确性由单测矩阵（15+8 用例）保证。
  如果你的模型用非连续 index 调 gather（如 MoE top-k 路由），建议加一轮针对性验证

## 环境情报（踩坑记录，与 gather 修复无关但值得知道）

- **flag_gems 5.3.5 全量 `enable()` 在 transformers 5.x 上会崩**：新 Cache 路径触发
  `lift_fresh` 算子的 GIL 崩溃。e2e 测试用 `use_gems(include=["gather"])` 只开单个算子绕开
- **插桩 flag_gems 的正确姿势**：注册表 `FULL_CONFIG_BY_FUNC` 在 import 时捕获函数对象，
  运行时改模块属性对 registrar 不可见，必须直接换表条目；且同一个 gather.py 会因相对导入
  被加载成两份模块（`_ascend.ops.gather` 与 `flag_gems.runtime.backend._ascend.ops.gather`）
- huggingface.co 直连不通时用 `HF_ENDPOINT=https://hf-mirror.com`

## 验证矩阵汇总

| 验证项 | 结果 |
|---|---|
| issue #5746 原复现（修复前） | NPU 507035 崩溃 |
| 非连续 index 回归矩阵（expand/transpose/slice × dim 0/1/2） | 修复前 9/9 失败 → 修复后 9/9 通过 |
| rank 1-6 全维度 + 复合视图 | 8/8 通过 |
| 全量 FlagGems gather 套件（仓库 pytest） | 72 passed / 0 failed |
| 端到端：真实 BERT 前向+反向 | gather 被真实调用，输出逐位一致，梯度误差 1.4e-12 |
| 连续路径性能 | 与修复前差异在机器噪声内 |
