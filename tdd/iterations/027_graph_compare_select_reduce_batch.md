# 027: 官方 compare/select/reduce 批量 TDD 迭代

## 基本信息

- **日期**: 2026-07-25
- **前置迭代**: `026_graph_math_reduce_batch`
- **触发来源**: 人类需求，继续规划并执行“简单的大规模算子迭代”

## 目标

沿 ONNX 官方图谱继续扩展一批实现风险可控、适合 MCU C99 correctness baseline 的算子。本轮选择：

- `Where`
- `Equal`
- `Greater`
- `Less`
- `ReduceProd`
- `ReduceL1`
- `ReduceL2`

本轮刻意没有纳入 `ArgMax`。`ArgMax` 输出 index，涉及模型最终输出非 int8 的 ABI 和 numeric runner 扩展，适合单独迭代。

## 新增/修改的测试用例

| case | ONNX op | 支持形态 | 数值数据集 |
|------|---------|----------|------------|
| `WHERE_001` | `Where` | 静态 bool mask，同形状 QDQ/int8 then/else | `where_qdq_smoke` |
| `EQUAL_001` | `Equal` + `Where` | 同形状 QDQ/int8 比较，bool 中间张量由 `Where` 消费 | `equal_where_qdq_smoke` |
| `GREATER_001` | `Greater` + `Where` | 同形状 QDQ/int8 比较，bool 中间张量由 `Where` 消费 | `greater_where_qdq_smoke` |
| `LESS_001` | `Less` + `Where` | 同形状 QDQ/int8 比较，bool 中间张量由 `Where` 消费 | `less_where_qdq_smoke` |
| `REDUCEPROD_001` | `ReduceProd` | rank=2，`axes=[1]`，`keepdims=1` | `reduceprod_qdq_smoke` |
| `REDUCEL1_001` | `ReduceL1` | rank=2，`axes=[1]`，`keepdims=1` | `reducel1_qdq_smoke` |
| `REDUCEL2_001` | `ReduceL2` | rank=2，`axes=[1]`，`keepdims=1` | `reducel2_qdq_smoke` |

同时修正 `validate_cases.py` 的 case id 正则，使 `REDUCEL1_001`、`REDUCEL2_001` 这类包含数字的官方算子名可以被规范识别。

## 执行结果

首次执行新增 `WHERE_001` 时失败：

```text
WHERE_001 expected=ok actual=unsupported
```

这符合 TDD 预期：case 先定义产品应具备的能力，源码随后按失败报告补齐。

## 代码修改

1. converter:
   - 增加 `Where`、`Equal`、`Greater`、`Less`、`ReduceProd`、`ReduceL1`、`ReduceL2` 官方算子白名单。
   - 增加 compare bool 中间张量的量化上下文提取。
   - 增加 `Where` 对静态 bool initializer 和 compare 输出 condition 的识别。
   - 将 reduce family 扩展到 `ReduceProd/ReduceL1/ReduceL2`。

2. codegen:
   - 增加 `generated_c_where_s8` 逐元素选择路径。
   - 增加 `generated_c_equal_bool`、`generated_c_greater_bool`、`generated_c_less_bool`，以 `int8_t` buffer 保存 bool condition 的 `0/1`。
   - 增加 `generated_c_reduceprod_s8`、`generated_c_reducel1_s8`、`generated_c_reducel2_s8`。
   - 补齐 tensor-aware run body 对静态 bool mask、compare 输出、Where 三输入的表达式解析。

## 最终结果

执行命令：

```bash
PYTHONPATH=src python tdd/scripts/run_tests.py --validate-only
PYTHONPATH=src python tdd/scripts/run_regression.py --generate
PYTHONPATH=src python tdd/scripts/run_tests.py --mode target --generate
PYTHONPATH=src python tdd/scripts/generate_support_map.py
```

结果：

- validate-only: `76` 个用例规格校验通过
- 新增 7 个 case target: 全部 PASS
- 新增 7 个 case numeric: 全部 PASS，均为 `top1=2/2`、饱和率 `0.00`、最大绝对误差 `0.0`
- baseline structural: `58/58 PASS`
- numeric: `54/54 PASS`
- regression: `PASS`
- target: `76/76 PASS`

## 发现与后续

本轮确认了新范式继续有效：从 ONNX 官方图谱挑选一组简单算子，先建立 support matrix 与 case，再由失败驱动 parser/codegen，实现后进入数值回归与可视化图谱。

比较类算子引入了 bool 中间张量，但本轮没有扩大模型外部 ABI；bool 只在生成 C 内部以 `int8_t` 的 `0/1` buffer 承载，并由 `Where` 消费。这个策略足够支撑阈值选择类图结构，也给后续 `And/Or/Not` 留出了自然扩展点。

后续建议：

- 单独迭代 `ArgMax`，正式处理 index 输出和 numeric runner 的非 int8 输出验收。
- 继续补 `GreaterOrEqual/LessOrEqual`，复用 compare bool 中间张量框架。
- 继续补 `And/Or/Not`，让 bool 子图能力更完整。
- 从 `NET_004` KWS-style 或其他时序网络中继续提炼真实模型驱动 case。
