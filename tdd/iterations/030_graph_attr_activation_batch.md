# 030: 官方属性/二元激活批量 TDD 迭代（100% 计划 W1b）

## 基本信息

- **日期**: 2026-08-11
- **前置迭代**: `029_graph_unary_math_batch`
- **触发来源**: `PLAN_100_SUPPORT.md` 图谱 100% 收敛计划 W1b

## 目标

按收获队列执行 W1b：带属性/二元的逐元素激活。本轮选择：

- `Elu`（alpha）
- `Selu`（alpha/gamma）
- `HardSigmoid`（alpha/beta）
- `ThresholdedRelu`（alpha）
- `Celu`（alpha）
- `PRelu`（常量 slope 二元输入）

`Mish` 因需要 opset 18（工程 cap 17）推迟，记为 opset 扩展边界。

## 新增/修改的测试用例

| case | ONNX op | 支持形态 | 数值数据集 |
|------|---------|----------|------------|
| `ELU_001` | `Elu` | 静态 QDQ/int8，alpha=0.5 | `elu_qdq_smoke` |
| `SELU_001` | `Selu` | 静态 QDQ/int8，alpha/gamma 默认 | `selu_qdq_smoke` |
| `HARDSIGMOID_001` | `HardSigmoid` | 静态 QDQ/int8，alpha/beta 默认 | `hardsigmoid_qdq_smoke` |
| `THRESHOLDEDRELU_001` | `ThresholdedRelu` | 静态 QDQ/int8，alpha=1.0 | `thresholdedrelu_qdq_smoke` |
| `CELU_001` | `Celu` | 静态 QDQ/int8，alpha=1.0 | `celu_qdq_smoke` |
| `PRELU_001` | `PRelu` | 同形状 QDQ/int8 + 量化常量 slope | `prelu_qdq_smoke` |

## 执行结果

首次执行 6 个新 case 全部失败（unsupported），符合 TDD 预期。

## 关键修复

1. **属性透传链路**：converter 正确提取了 Elu 的 alpha=0.5，但 `_generated_unary_layer`
   只把 alpha 传给 LeakyRelu，导致生成 C 用默认 alpha=1.0，ELU 数值偏差达 10 个单位。
   修复：`_generated_unary_layer` 与 `_generated_unary_call` 同步支持
   Elu/Selu/HardSigmoid/ThresholdedRelu/Celu 的 alpha/gamma/beta 属性。

2. **量化取整一致性（跨路径修复）**：PRELU 数据恰好命中 -2.5 的量化边界，暴露
   生成 C 的 requant 用 round-half-away（`±0.5` 截断）而 ONNX QuantizeLinear 用
   round-half-even（ties-to-even），差 2 个 int8 单位。
   修复：把 generator.py 全部 4 处 requant 公式改为 `(int32_t)nearbyintf(nanoc_qf)`
   （C 默认 FE_TONEAREST = ties-to-even）。全量回归 70/70 确认无破坏。
   这是影响所有 generated-C 路径的 ONNX 一致性修复，后续算子不再受 .5 边界困扰。

3. **PRelu 二元接入**：`_generated_elementwise_quant_info` 复用为 PRelu 的常量 slope
   量化；补 `_quantized_weight_declarations` 的 constant-input kind 集合与
   tensor run body 的 prelu 分派（初版缺这两处分别导致 `bias_values` 与 `input_size`
   KeyError 落入 FC fallback）。

## 最终结果

执行命令：

```bash
PYTHONPATH=src python tdd/scripts/run_tests.py --validate-only
PYTHONPATH=src python tdd/scripts/run_tests.py --mode target --case <6 个新 case> --generate
PYTHONPATH=src python tdd/scripts/run_numeric_tests.py --case <6 个新 case> --generate
PYTHONPATH=src python tdd/scripts/run_regression.py --generate
PYTHONPATH=src python tdd/scripts/generate_support_map.py
PYTHONPATH=src python tdd/scripts/run_tests.py --mode target --generate
```

结果：

- validate-only: `92` 个用例规格校验通过
- 新增 6 个 case target: 全部 PASS（api/compile/run 全 OK）
- 新增 6 个 case numeric: 全部 PASS，均为 `top1=2/2`、饱和率 `0.00`、最大绝对误差 `0.0`
- regression: PASS（70/70 numeric，含 6 新增 + 64 既有，无退化）
- target: `92/92 PASS`

## 发现与后续

- **opset 边界**：Mish 需 opset 18，工程当前 `SUPPORTED_OPSET_MAX=17`，推迟并记录。
  后续可单独做 opset 18 扩展迭代（同时可覆盖其他新版算子）。
- **量化取整 conformance**：本轮修复了 round-half-even 一致性，是 generated-C 的
  基础正确性改进，已通过全量回归验证。
- 100% 计划下一批（W1c）：shape/常量折叠 `Shape` `Size` `Constant` `ConstantOfShape`
  `Identity` `Cast` `Split` `Expand` `Tile` `Range`。
