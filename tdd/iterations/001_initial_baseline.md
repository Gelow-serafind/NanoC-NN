# 迭代 001: 初始基线 — 核心算子覆盖

## 基本信息
- **日期**: 2026-07-03
- **前置迭代**: 000（历史遗留计划归档）
- **触发来源**: 项目引入 TDD 开发方法论，需要建立首批测试基线

## 目标

建立核心算子的基线测试覆盖，验证代码生成器在"白名单内最基础场景"下的端到端正确性。覆盖 Gemm、Conv、MaxPool、Softmax 四个核心运行期算子，加一个经典 CNN 组合拓扑和一个负向测试。

## 新增测试用例

| ID | 分类 | 名称 | 预期 |
|----|------|------|------|
| GEMM_001 | core/gemm | 最小对称 FC 无 bias | ok |
| GEMM_002 | core/gemm | 非对称输入 zp≠0 含 bias | ok |
| GEMM_003 | core/gemm | 非 4 对齐维度 13→7 | ok |
| GEMM_004 | core/gemm | 中等规模 64→32 | ok |
| CONV_001 | core/conv | 1×1 pointwise 单通道 | ok |
| CONV_002 | core/conv | 3×3 标准 SAME padding | ok |
| CONV_003 | core/conv | stride=2 下采样 | ok |
| CONV_004 | core/conv | 非对称输入 zp≠0 | ok |
| MAXPOOL_001 | core/maxpool | 标准 2×2 stride=2 | ok |
| SOFTMAX_001 | core/softmax | 10 分类标准 | ok |
| TOPO_001 | topology | Conv→Relu→Pool→Flatten→FC | ok |
| NEG_001 | negative | float32 无 Q/DQ 模型拒绝 | unsupported |

## 执行结果

首次执行日期：2026-07-03

| ID | 预期 | 实际 | 结果 | 原因 |
|----|------|------|------|------|
| GEMM_001 | ok | blocked | FAIL | 量化字段未提取：converter 未为 Gemm 节点生成 codegen 所需的 int8 Q/DQ 量化参数 |
| GEMM_002 | ok | blocked | FAIL | 同上 |
| GEMM_003 | ok | blocked | FAIL | 同上 |
| GEMM_004 | ok | blocked | FAIL | 同上 |
| CONV_001 | ok | blocked | FAIL | 同上（Conv 节点同样缺少量化字段） |
| CONV_002 | ok | blocked | FAIL | 同上 |
| CONV_003 | ok | blocked | FAIL | 同上 |
| CONV_004 | ok | blocked | FAIL | 同上 |
| MAXPOOL_001 | ok | ok | PASS | — |
| SOFTMAX_001 | ok | ok | PASS | — |
| TOPO_001 | ok | blocked | FAIL | 包含 Conv+Gemm，同样量化字段缺失 |
| NEG_001 | unsupported | blocked | FAIL | 预期值需调整：float32 模型走到 codegen 后报 blocked 而非 unsupported |

**通过率：2/12 (17%)**

### 关键发现

1. **核心问题**：converter 在解析 Q/DQ 模型时，虽然能识别 QuantizeLinear/DequantizeLinear 节点，但未为运行期算子（Gemm、Conv）生成 codegen 所需的节点级量化字段（`nodes.<name>.multiplier`、`nodes.<name>.shift` 等）。
2. **MaxPool/Softmax 通过**：说明这两个算子有独立的量化参数处理路径，不依赖节点级量化字段提取。
3. **NEG_001 预期值不精确**：float32 模型实际返回 `blocked` 而非 `unsupported`，需要确认 pipeline 的错误分级语义后调整预期。

## 代码修改

> 本轮为建立基线，暂不修改代码。上述发现将作为后续迭代的开发任务。

## 最终结果

基线已建立。测试基础设施（模型生成 → pipeline 执行 → 结果收集 → 报告输出）全链路验证通过。

## 发现与后续

下一轮迭代（002）应聚焦于：
1. **定位 converter 量化字段提取缺口**：为什么 Gemm/Conv 的 Q/DQ 参数未被提取到 `model_graph.json` 的 nodes 字段中？
2. **调整 NEG_001 预期**：确认 `blocked` vs `unsupported` 的语义区分后，修正预期值
3. **确认 model_builder 生成的 Q/DQ 结构**：是否符合 converter 期望的 pattern（可能需要调整模型生成方式）
