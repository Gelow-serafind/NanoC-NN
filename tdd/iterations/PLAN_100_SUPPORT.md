# PLAN: ONNX 图谱 100% 收敛计划（MCU 相关算子空间）

> 本文件是"从当前状态连续迭代到 ONNX 图谱全收敛"的**唯一权威驱动文档**。
> 任何会话（包括新开会话）读本文件即可精确续跑：范围、成功判据、收获队列、
> 迭代配方、状态指针、护栏、检查点全部在此。
> 静态部分（目标/判据/配方/护栏）不改；动态部分（收获队列状态、指针）每批迭代后更新。

## 1. 目标与成功判据（100% 的定义）

- **范围**：ONNX 官方 schema 中 **MCU 端侧 int8 推理相关**的算子空间（~130 个）。
  字面 221 全集含 ~94 个训练/检测后处理/字符串/序列/树集成等无关算子，**不在**收敛范围。
- **100% 判据**：图谱上**不存在灰色行**——每个相关算子要么有 PASS 用例（`ok`），
  要么有明确的拒绝边界用例（`blocked`/`unsupported` 并记录原因）。
  到那时 `CAPABILITIES.md` 不再出现"已规划但未验证"行。
- 现有基础：**59/130 已确认 `ok`**，71 个相关候选待收敛（截至 2026-08-11）。

## 2. 收获队列（71 个相关候选，按波次）

每批完成后在下方表格标记状态：`ok` / `reject`（并注明原因）/ `pending`。
**指针** = 下一个要做的批次。

### 波次与算子

| 波次 | 主题 | 算子 | 批次 | 状态 |
|------|------|------|------|------|
| W1a | 无属性逐元素激活 | `Erf` `Softplus` `Softsign` `HardSwish` | 批 1 | ✅ done（029） |
| W1b | 属性/二元逐元素激活 | `Elu` `Selu` `HardSigmoid` `ThresholdedRelu` `Celu` `PRelu` | 批 2 | ✅ done（030） |
| W1b* | 需 opset 扩展 | `Mish`（opset 18+，工程当前 cap 17） | 推迟 | ⏸️ deferred |
| W1c | shape/常量折叠 | `Shape` `Size` `Constant` `ConstantOfShape` `Identity` `Cast` `Split` `Expand` `Tile` `Range` | 批 3-4 | pending |
| W1d | 归约/池化扩展 | `GlobalMaxPool` `GlobalLpPool` `LpPool` `LpNormalization` `ReduceLogSum` `ReduceLogSumExp` `ReduceSumSquare` `CumSum` `Mean` `Sum` | 批 5-6 | pending |
| W0 | ABI 基础（index 输出） | `ArgMax` `ArgMin`（需扩展非 int8 外部 ABI + numeric runner） | 批 7 | pending |
| W2 | bool/索引补完 | bool-initializer 逻辑输入、逻辑级联；`NonZero` `Compress` `OneHot` `TopK` `GatherElements` `GatherND` `ScatterND` | 批 8-9 | pending |
| W3 | 归一化 | `BatchNormalization` `InstanceNormalization` `LayerNormalization` `LRN` `GroupNormalization` | 批 10 | pending |
| W4 | 卷积/量化变体 | `ConvTranspose` `ConvInteger` `MatMulInteger` `DynamicQuantizeLinear` `DepthToSpace` `SpaceToDepth` `Resize` `Upsample` | 批 11-12 | pending |
| W5 | 尾项/边缘 | `Mod` `EyeLike` `Trilu` `ReverseSequence` `MaxUnpool` 窗口函数`Hamming/Hann/Blackman` `MelWeightMatrix` `DeformConv` `RoiAlign` | 批 13 | pending |

> 波次顺序说明：先做 W1 纯增量（贴近现有框架、快速积累绿区），再攻 W0 ArgMax ABI
> （结构性改动，且不阻塞其他算子），随后 W2→W5。W5 中检测类算子
> （`DeformConv`/`RoiAlign`）与罕见算子可判 `reject` 记录边界，不必硬做。

### 当前指针

- **当前批次**：W1c（批 3）：`Shape` `Size` `Constant` `ConstantOfShape` `Identity` `Cast` `Split` `Expand` `Tile` `Range`
- **下一批次**：W1d
- **已完成**：批 1（W1a，4 算子，029）、批 2（W1b，6 算子，030）
- **已推迟**：`Mish`（opset 18+，工程 cap 17，纳入后续 opset 扩展迭代）
- **跨路径修复**：round-half-even 量化取整一致性（nearbyintf，030）

## 3. 每批迭代配方（025–028 已验证的 TDD 闭环）

1. **TDD 资产先行**：`cases/core/<op>/<OP>_001.md` 规格（schema 驱动全格式）→
   `cases_registry.py` 登记 expected=ok + NumericCheck → `support_matrix.py` 登记子形态 →
   `generate_models.py` 加构造器并注册 → smoke 数据集。
2. **红阶段**：`run_tests.py --mode target --case ... --generate`，确认预期失败（unsupported）。
3. **converter**：`parser.py` 白名单 + 量化提取。
4. **codegen**：`mapper.py` / `quantization.py` / `generator.py`。
5. **绿阶段**：target 通过（api/compile/run OK）→ numeric 通过（top1/max_abs 达标）。
6. **全量回归**：`run_regression.py --generate`（新功能不得打破旧能力）。
7. **更新图谱与文档**：`generate_support_map.py` → 全量 target（更新 CAPABILITIES.md）→
   写 `iterations/<NNN>_*.md` → 更新 `STATUS.md` → 更新本文件收获队列。
8. **提交**：按范围 3 提交（源码实现 / 用例+迭代记录 / 再生产物），前缀 `新增[ADD]`/`修改[CHG]`。

## 4. 护栏

- **回归是硬门槛**：每批必须全量回归通过；已确认能力退化 = 严重缺陷，立即修复，不得跳过。
- **禁止削弱测试**：不允许为了让测试通过而改预期；结构 PASS 但数值错 = 严重缺陷。
- **数值真值表可判别**：数据集必须非退化（bool 组合含 T 与 F、样本跨越边界），
  避免"恒真/恒假导致逻辑错误被隐藏"。
- **能力只通过测试获得**：CAPABILITIES.md 自动生成，禁手改。
- **提交纪律**：每批 3 提交、信息无 AI 署名；不 push 除非用户明确要求。

## 5. 检查点（仅以下两类需打断用户）

1. **架构决策**：如 ArgMax 的非 int8 外部 ABI / numeric runner 扩展策略、BatchNorm 的
   lowering 方式，若存在多个合理方案且影响后续，停下来与用户确认。
2. **`ok` vs `reject` 判断**：某算子实现代价远超收益（如 DeformConv/RoiAlign 检测类、
   MaxUnpool 反池化），或该形态在真实 MCU 模型中几乎不出现，判 `reject` 并记录边界。

其余全部自主执行，不逐批汇报。

## 6. 运行记录（本计划文件即为状态兜底）

| 批次 | 算子 | target | numeric | 回归 | 提交 | 日期 |
|------|------|--------|---------|------|------|------|
| 批 1 (W1a) | Erf/Softplus/Softsign/HardSwish | 86/86 | 64/64 | PASS | 029 | 2026-08-11 |
| 批 2 (W1b) | Elu/Selu/HardSigmoid/ThresholdedRelu/Celu/PRelu | 92/92 | 70/70 | PASS | 030 | 2026-08-11 |
