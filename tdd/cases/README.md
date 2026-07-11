# 测试用例规范

本目录存放所有 TDD 测试用例的**规格文件**。每个用例一个 `.md` 文件，是永久资产——一旦创建永不删除，只追加或修改预期结果。

## 命名约定

```
{CATEGORY}_{NNN}.md
```

- **CATEGORY**：算子或分类名称，全大写（如 `GEMM`, `CONV`, `MAXPOOL`, `TOPO`, `NEG`）
- **NNN**：三位递增编号，从 001 开始

示例：`GEMM_001.md`, `CONV_003.md`, `TOPO_001.md`, `NEG_002.md`

## 目录组织

```
cases/
├── core/           ← 原子算子测试（单个算子孤立验证）
│   ├── gemm/
│   ├── conv/
│   ├── maxpool/
│   ├── avgpool/
│   └── softmax/
├── topology/       ← 组合拓扑测试（多算子串接）
├── networks/       ← 完整网络测试（真实或训练得到的典型网络）
├── extension/      ← 驱动扩展测试（预期 blocked，定义未来需求）
└── negative/       ← 负向测试（预期拒绝/失败，验证错误处理）
```

当前历史用例中 `topology/TOPO_003` 已经承担完整 MNIST 网络回归职责。后续新增真实完整网络时，优先放入 `networks/`；历史用例可在合适的迭代中迁移，不强行打断当前回归链路。

## 用例文件格式

每个 `.md` 文件必须包含以下字段：

```markdown
# {ID}: {简短名称}

## 验证目标

一句话描述本用例要验证的核心能力。

## 来源

说明本用例的诞生背景，三选一：
- `内部探索` — 主动设计，扩展能力边界
- `用户反馈` — 用户报告的问题分解而来，附上问题描述或 issue 编号
- `缺陷复现` — 修复过程中构造的防退化用例，说明所防止的缺陷

## ONNX Schema 归属

说明本用例覆盖的 ONNX 官方算子语义。机器可校验的归属写在
`tdd/scripts/cases_registry.py` 的 `schema_refs` 字段中，必须引用
`tdd/scripts/support_matrix.py` 中的 `support_id`。

官方 ONNX 算子全集快照位于 `tdd/onnx_schema/`。新增官方算子 case 时，
必须先确认该 op 存在于当前绑定的 ONNX catalog 中；真实模型中出现但
catalog 没有的 op 只能作为 `extension` 明确登记。

| 字段 | 示例 |
|------|------|
| domain | `ai.onnx` |
| op_type | `Conv` |
| opset_range | `11+` |
| schema_form | `rank=4 NCHW, group=1, QDQ/int8 parameters` |
| lowering | `cmsis_conv2d_s8` |
| backend | `cmsis-nn` |

## 网络结构

​```
Q/DQ Input → [算子序列] → Q/DQ Output
​```

## 输入

- **张量形状**: `[1, ...]`（NCHW 格式）
- **数据类型**: int8

## 算子参数

| 算子 | 参数 |
|------|------|
| ... | ... |

## 量化设计

| 张量 | scale | zero_point | 说明 |
|------|-------|-----------|------|
| input | ... | ... | ... |
| weight | ... | ... | ... |
| output | ... | ... | ... |

## 预期结果

- **codegen status**: `ok` / `blocked` / `unsupported` / `oversize`
- **若 ok**: 编译通过、运行不崩溃
- **关键验证点**: （如：生成代码中应包含 `arm_fully_connected_s8` 调用）

## 数值验收

- **是否需要**: `yes` / `no`
- **数据集**: `tdd/fixtures/datasets/<dataset_id>/dataset.json`
- **参考路径**: 原始 ONNX Runtime
- **通过阈值**: top1 一致率、绝对误差、饱和率等

## 边界/风险

描述本用例设计的边界条件和预期暴露的风险点。
```

## 编写原则

1. **一个用例只验证一个核心点**。不要在一个用例里堆砌多个验证目标。
2. **ONNX schema 是前端事实**。不要用业务名称创造平行前端算子；例如 Conv1d/Conv2d/DepthwiseConv 都应解释为 ONNX `Conv` 的不同 schema 子形态。
3. **官方全集先于支持矩阵**。新增官方算子能力时，先查 `tdd/onnx_schema/` 中的版本绑定 catalog，再新增或细化 support matrix 行。
4. **每个用例必须绑定 support matrix**。新增 case 时必须先在 `support_matrix.py` 中确认或新增 `support_id`，再在 registry 的 `schema_refs` 中引用它。
5. **量化参数必须显式指定**，不留"默认值"歧义。
6. **预期结果必须明确**——`ok`、`blocked`、`unsupported`、`oversize` 四选一，不允许模糊。
7. **边界/风险字段必须填写**——解释为什么选择这组参数，预期能暴露什么问题。
8. **不描述实现细节**——用例只定义"什么是对的"，不描述"怎么做到"。
9. **每个用例就是一项能力声明**——用例通过后，它就成为产品能力集的一部分，永远不允许退化。新增用例时要意识到：你在定义产品应该拥有的能力。
10. **来源字段必须填写**——维护者需要知道"为什么这个用例存在"。来自用户反馈的用例尤其重要：一旦 PASS，它就是防止同一问题再次出现的防退化锁。
11. **完整网络必须有数值验收计划**。真实 MNIST、语音检测、小型图像分类等完整网络，不能只验证生成 C 文件和 API 字符串；必须准备数据集并逐步接入 ONNX-vs-C 对比。
12. **人工数据集必须放在 `tdd/fixtures/`**。`tdd/work/` 是脚本临时目录，`tdd/results/` 是报告目录，都不能作为数据集来源。

## 能力集视角

测试用例集合就是产品的能力集合。规划新用例时，思考方式应该是：

- "我们的产品应该具备什么能力？" → 转化为测试用例
- "这个能力已经具备了吗？" → 查看 `tdd/CAPABILITIES.md`
- "开发这个能力需要做什么？" → 先看 `tdd/reports/onnx_support_map.html` 定位 ONNX 算子地图节点，再执行用例，观察失败报告，按报告修复

规划新算子或新 case 时，推荐先打开 ONNX 支持思维导图：

```bash
python tdd/scripts/generate_support_map.py
```

从图谱中确认目标算子属于：

- **已支持算子的未覆盖子形态**：在现有 support row 下补 case，或细化新的 support row。
- **已登记但未完成**：优先补最小 case 和实现，使其从 planned/blocked 进入 PASS。
- **尚未开始**：先查 ONNX 官方 schema，再新增 planned support row 和最小 case。

**禁止反向操作**：不允许先改代码再补用例来证明它。先有能力定义（用例），后有能力实现（代码修复）。

## 规格校验

新增或修改用例后，必须运行：

```bash
python tdd/scripts/run_tests.py --validate-only
```

校验会检查：

- `cases/`、`cases_registry.py`、`generate_models.py` 中的用例 ID 是否一致。
- 每个规格文件是否包含必需章节。
- 规格中的 `codegen status` 是否与 registry 中的 expected 一致。
- 规格路径是否与 registry 中的分类一致。
- registry 中每个 case 是否绑定 `support_matrix.py` 中存在的 `support_id`。
- `schema_source="official"` 的 support matrix 行是否存在于当前 ONNX catalog。
- `ok` case 是否只引用 `ok` 的 support matrix 行。
- CMSIS-NN `ok` case 是否声明了 support matrix 要求的 CMSIS-NN API。
- support matrix 行是否至少被 case 覆盖，或显式标记为 planned。
