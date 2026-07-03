# TDD 状态总览

> 本文件是 agent 进入 `tdd/` 目录后的第一个入口。阅读此文件即可了解当前能力边界和下一步工作方向。

## 核心概念：能力集

测试用例集合 = 产品的能力集。每个 PASS 的用例代表一项已验证能力，每个 FAIL 的用例代表一项目标能力。开发的唯一目标是：**让能力集变大，且永不缩小**。

- 能力集权威声明: `tdd/CAPABILITIES.md`（由脚本自动生成，git 跟踪）
- 能力集大小 = 产品成熟度
- 能力集增长 = 开发进度
- 能力集退化 = 回归缺陷（严禁发生）

## 当前能力矩阵

| ONNX 算子 | Converter 解析 | Codegen 渲染 | CMSIS-NN API | TDD 覆盖 | 能力确认 |
|-----------|:---:|:---:|--------------|:---:|:---:|
| Conv (group=1, dilation=[1,1]) | OK | OK | `arm_convolve_wrapper_s8` | CONV_001~004 | FAIL |
| Gemm/MatMul (transB=1) | OK | OK | `arm_fully_connected_wrapper_s8` | GEMM_001~004 | FAIL |
| MaxPool | OK | OK | `arm_max_pool_s8` | MAXPOOL_001 | PASS |
| AveragePool | OK | OK | `arm_avgpool_s8` | -- | 未覆盖 |
| GlobalAveragePool | OK | OK | `arm_avgpool_s8` | -- | 未覆盖 |
| Softmax | OK | OK | `arm_softmax_s8` | SOFTMAX_001 | PASS |
| Relu/Clip | OK | OK (融合) | activation min/max | TOPO_001 | FAIL |
| Flatten/Reshape | OK | OK (折叠) | -- | TOPO_001 | FAIL |
| QuantizeLinear/DequantizeLinear | OK | OK (折叠) | -- | 全部用例 | 部分 |
| Add | OK | blocked | `arm_elementwise_add_s8` | -- | 未覆盖 |
| Mul | OK | blocked | `arm_elementwise_mul_s8` | -- | 未覆盖 |
| Concat | OK | blocked | `arm_concatenation_s8` | -- | 未覆盖 |
| Transpose (runtime) | OK | blocked | `arm_transpose_s8` | -- | 未覆盖 |
| DepthwiseConv | OK | blocked | -- | -- | 未覆盖 |

## 能力集摘要

| 指标 | 值 |
|------|-----|
| 测试用例总数 | 12 |
| 已确认能力 (PASS) | 2 |
| 目标能力 (FAIL) | 10 |
| 能力集覆盖率 | 17% |
| 最近执行日期 | 2026-07-03 |

## 迭代记录

| 编号 | 日期 | 主题 | 能力集变化 |
|------|------|------|-----------|
| 000 | -- | 历史遗留计划归档 | -- |
| 001 | 2026-07-03 | 初始基线：核心算子覆盖 | 0 -> 2 |

## 下一步方向

基于当前能力集状态（2/12），优先级排列：

1. **[阻塞] 修复 Gemm/Conv 量化字段提取** — 解决后预计 +8 能力（GEMM_001~004, CONV_001~004）
2. **调整 NEG_001 预期** — 确认 `blocked` vs `unsupported` 语义后修正（+1 能力）
3. **TOPO_001 依赖 Gemm/Conv** — 上游修复后预计自动通过（+1 能力）
4. **AveragePool/GlobalAveragePool** — 当前无覆盖，需新增用例
5. **拓展组合拓扑** — 待核心算子通过后再展开

## 约束边界

当前代码生成器仅在以下条件下承诺正确输出：

- 输入模型必须是 ONNX Q/DQ int8 量化模型
- Conv: group=1, dilation=[1,1]
- Gemm: transB=1
- Pool: 输入/输出量化参数必须一致
- 全固定 shape, batch=1
- Opset 11~17

## 如何操作

```bash
# 校验 TDD 用例规格
python tdd/scripts/run_tests.py --validate-only

# 生成并执行稳定回归门禁
python tdd/scripts/run_tests.py --mode baseline --generate

# 生成并执行全部目标用例（更新 CAPABILITIES.md + results/latest.json）
python tdd/scripts/run_tests.py --mode target --generate

# 只执行单个用例
python tdd/scripts/run_tests.py --case GEMM_001 --generate

# 只执行某一类
python tdd/scripts/run_tests.py --category core/gemm --generate
```
