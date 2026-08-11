# 029: 官方 unary 数学批量 TDD 迭代（100% 计划 W1a）

## 基本信息

- **日期**: 2026-08-11
- **前置迭代**: `028_graph_bool_logic_compare_ext`
- **触发来源**: `PLAN_100_SUPPORT.md` 图谱 100% 收敛计划 W1a（无属性逐元素激活）

## 目标

按 `PLAN_100_SUPPORT.md` 收获队列执行第一批：从最贴近现有 unary 框架的算子入手，
快速积累绿区。本轮选择：

- `Erf`
- `Softplus`
- `Softsign`
- `HardSwish`

## 新增/修改的测试用例

| case | ONNX op | 支持形态 | 数值数据集 |
|------|---------|----------|------------|
| `ERF_001` | `Erf` | 静态 QDQ/int8 tensor，输入域 `[-1.5, 1.5]` | `erf_qdq_smoke` |
| `SOFTPLUS_001` | `Softplus` | 静态 QDQ/int8 tensor，输入域 `[-5, 5]` | `softplus_qdq_smoke` |
| `SOFTSIGN_001` | `Softsign` | 静态 QDQ/int8 tensor，输入域 `[-5, 5]` | `softsign_qdq_smoke` |
| `HARDSWISH_001` | `HardSwish` | 静态 QDQ/int8 tensor，输入域 `[-6, 6]` | `hardswish_qdq_smoke` |

## 支持形态边界

- 全部为静态 QDQ/int8、同量化输入输出、C99 逐元素 correctness baseline，
  与既有 `Exp/Log/Floor` 等 unary 路径一致。
- 4 个算子无属性，直接复用 `_generated_unary_quant_info` 与 `_generated_unary_layer`。
- 数值验收设 `max_abs=0.05`（允许 1 个 int8 单位的 ULP 取整边界抖动），
  实际执行全部 `max_abs=0.0` 逐字节一致。

## 执行结果

首次执行 4 个新 case 全部失败，符合 TDD 预期：

```text
ERF_001       expected=ok actual=unsupported
SOFTPLUS_001  expected=ok actual=unsupported
SOFTSIGN_001  expected=ok actual=unsupported
HARDSWISH_001 expected=ok actual=unsupported
```

失败根因均为 converter 白名单未包含新算子。

## 代码修改

1. converter：`SUPPORTED_OPS` / `_int8_contract` / unary 量化提取集合增加 4 个算子。
2. codegen：
   - `_generated_unary_call` 增加 4 个数学分支：`erff`、`log1pf(expf)`、
     `x/(1+fabsf(x))`、`x·fminf(fmaxf(x+3,0),6)/6`。
   - `_renderer_is_complete` / `_synthetic_runtime_mappings` / `_runtime_layers` /
     `_layer_output_element_count` / 两个 run body 分派同步注册。
   - mapper / quantization 注册 4 个算子的 `generated_c_*_s8` 与字段校验。

## 最终结果

执行命令：

```bash
PYTHONPATH=src python tdd/scripts/run_tests.py --validate-only
PYTHONPATH=src python tdd/scripts/run_tests.py --mode target --case <4 个新 case> --generate
PYTHONPATH=src python tdd/scripts/run_numeric_tests.py --case <4 个新 case> --generate
PYTHONPATH=src python tdd/scripts/run_regression.py --generate
PYTHONPATH=src python tdd/scripts/generate_support_map.py
PYTHONPATH=src python tdd/scripts/run_tests.py --mode target --generate
```

结果：

- validate-only: `86` 个用例规格校验通过
- 新增 4 个 case target: 全部 PASS（api/compile/run 全 OK）
- 新增 4 个 case numeric: 全部 PASS，均为 `top1=2/2`、饱和率 `0.00`、最大绝对误差 `0.0`
- regression: PASS（64/64 numeric，含 4 新增 + 60 既有，无退化）
- target: `86/86 PASS`

## 设计要点

- 数据集输入全部取 scale 的整数倍（0.05），int8 量化精确，使超越函数结果远离
  量化取整边界，实测 `max_abs=0.0` 逐字节一致。
- 4 个算子落在既有 unary 框架内，代码改动集中在 `_generated_unary_call` 数学分支，
  验证了 100% 计划里"贴近现有框架的算子成本最低"的假设。

## 发现与后续

- 本轮 4 个算子一次性收敛，无架构改动，无卡点。
- 100% 计划下一批（W1b）：带属性/二元逐元素激活 `Elu` `Selu` `HardSigmoid`
  `ThresholdedRelu` `Celu` `Mish` `PRelu`，需要处理属性（alpha/gamma/beta）传入
  `_generated_unary_call` 与 `PRelu` 的 slope 二元输入。
