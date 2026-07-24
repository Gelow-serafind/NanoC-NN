# 026: 官方 math/reduce 批量 TDD 迭代

## 背景

上一轮 `Tanh`、`LeakyRelu`、`Clip`、`Neg`、`Sqrt`、`Reciprocal`、`ReduceMean`、`Unsqueeze` 证明了 schema-driven TDD 可以继续批量推进简单官方算子。本轮继续从 ONNX 支持图谱中选择实现风险可控、常见于模型导出或预后处理边界、适合作为 correctness baseline 的算子：

- `Exp`
- `Log`
- `Floor`
- `Ceil`
- `Round`
- `Sign`
- `Min`
- `Max`
- `Pow`
- `ReduceSum`
- `ReduceMax`
- `ReduceMin`

本轮目标是先让这些官方 schema 子形态进入可生成、可编译、可运行、ONNX-vs-C 数值一致的能力集。当前实现以 C99 逐元素或小型归约 baseline 为主，后续可再替换为 LUT、定点近似或更贴近 CMSIS-DSP/CMSIS-NN 的优化路径。

## 新增用例

| case | ONNX op | 支持形态 | 数值数据集 |
|------|---------|----------|------------|
| `EXP_001` | `Exp` | 静态 QDQ/int8 tensor，受控输入域 | `exp_qdq_smoke` |
| `LOG_001` | `Log` | 静态 QDQ/int8 tensor，正输入域 | `log_qdq_smoke` |
| `FLOOR_001` | `Floor` | 静态 QDQ/int8 tensor | `floor_qdq_smoke` |
| `CEIL_001` | `Ceil` | 静态 QDQ/int8 tensor | `ceil_qdq_smoke` |
| `ROUND_001` | `Round` | 静态 QDQ/int8 tensor | `round_qdq_smoke` |
| `SIGN_001` | `Sign` | 静态 QDQ/int8 tensor | `sign_qdq_smoke` |
| `MIN_001` | `Min` | 同形状 QDQ/int8 双输入，第二输入可为常量 | `min_qdq_smoke` |
| `MAX_001` | `Max` | 同形状 QDQ/int8 双输入，第二输入可为常量 | `max_qdq_smoke` |
| `POW_001` | `Pow` | 同形状 QDQ/int8 双输入，有限输入域 | `pow_qdq_smoke` |
| `REDUCESUM_001` | `ReduceSum` | rank=2，`axes=[1]`，`keepdims=1` | `reducesum_qdq_smoke` |
| `REDUCEMAX_001` | `ReduceMax` | rank=2，`axes=[1]`，`keepdims=1` | `reducemax_qdq_smoke` |
| `REDUCEMIN_001` | `ReduceMin` | rank=2，`axes=[1]`，`keepdims=1` | `reducemin_qdq_smoke` |

## 修复与实现

1. TDD 资产先行：
   - 在 registry 中登记 12 个 expected `ok` 的新 case。
   - 在 support matrix 中登记 12 个官方 schema 子形态。
   - 为每个 case 增加独立规格文档和 smoke dataset。
   - 在 `generate_models.py` 中增加对应 ONNX 构造函数。

2. converter 扩展：
   - 增加 12 个官方算子的 parser 白名单。
   - 增加 `Min/Max/Pow` 的第二输入常量识别。
   - 增加 `ReduceSum/ReduceMax/ReduceMin` 的 axes initializer 辅助输入识别。
   - 将 `ReduceMean` 的量化提取逻辑泛化为 reduce family，共享 rank、axes、keepdims 约束。

3. mapper/codegen 扩展：
   - unary math 生成 `generated_c_exp_s8`、`generated_c_log_s8`、`generated_c_floor_s8`、`generated_c_ceil_s8`、`generated_c_round_s8`、`generated_c_sign_s8`。
   - binary math 生成 `generated_c_min_s8`、`generated_c_max_s8`、`generated_c_pow_s8`。
   - reduce family 生成 `generated_c_reducesum_s8`、`generated_c_reducemax_s8`、`generated_c_reducemin_s8`。

4. 本轮暴露并修复的关键问题：
   - `ReduceSum` 在当前 ONNX opset 下不能使用 legacy `axes` attribute，需要把 axes 作为第二个 input initializer，否则 ONNX checker 会拒绝模型。
   - 初版 reduce runtime 复用了 `ReduceMean` 的 layer kind，导致 `ReduceSum` 也被错误地除以列数，数值回归出现约 `0.45` 的最大误差。本轮将 reduce runtime 按 `ReduceMean/ReduceSum/ReduceMax/ReduceMin` 显式区分，避免同类归约误用。

## 验证结果

执行命令：

```bash
PYTHONPATH=src python tdd/scripts/run_tests.py --validate-only
PYTHONPATH=src python tdd/scripts/run_tests.py --mode target --case EXP_001 --case LOG_001 --case FLOOR_001 --case CEIL_001 --case ROUND_001 --case SIGN_001 --case MIN_001 --case MAX_001 --case POW_001 --case REDUCESUM_001 --case REDUCEMAX_001 --case REDUCEMIN_001 --generate
PYTHONPATH=src python tdd/scripts/run_numeric_tests.py --case EXP_001 --case LOG_001 --case FLOOR_001 --case CEIL_001 --case ROUND_001 --case SIGN_001 --case MIN_001 --case MAX_001 --case POW_001 --case REDUCESUM_001 --case REDUCEMAX_001 --case REDUCEMIN_001 --generate
PYTHONPATH=src python tdd/scripts/run_regression.py --generate
PYTHONPATH=src python tdd/scripts/generate_support_map.py
PYTHONPATH=src python tdd/scripts/run_tests.py --mode target --generate
```

结果：

- validate-only: `69` 个用例规格校验通过
- 新增 12 个 case target: 全部 PASS
- 新增 12 个 case numeric: 全部 PASS，均为 `top1=2/2`、饱和率 `0.00`、最大绝对误差 `0.0`
- baseline structural: `51/51 PASS`
- numeric: `47/47 PASS`
- regression: `PASS`
- target: `69/69 PASS`

## 能力集变化

本轮新增 12 个官方 ONNX schema 子形态进入 `CAPABILITIES.md`：

- `ONNX_EXP_QDQ_INT8`
- `ONNX_LOG_QDQ_INT8`
- `ONNX_FLOOR_QDQ_INT8`
- `ONNX_CEIL_QDQ_INT8`
- `ONNX_ROUND_QDQ_INT8`
- `ONNX_SIGN_QDQ_INT8`
- `ONNX_MIN_QDQ_INT8`
- `ONNX_MAX_QDQ_INT8`
- `ONNX_POW_QDQ_INT8`
- `ONNX_REDUCESUM_QDQ_INT8`
- `ONNX_REDUCEMAX_QDQ_INT8`
- `ONNX_REDUCEMIN_QDQ_INT8`

能力图谱 `tdd/reports/onnx_support_map.html` 已同步刷新。

## 对新范式的评价

这轮验证了当前 schema-driven TDD 路线是可用的：先从 ONNX 图谱选择算子，再沉淀 support matrix、case 文档、模型构造器和数据集，随后让失败报告驱动 converter/codegen 修复。相比早期直接从完整网络里盲目修补，当前流程能把问题拆成可复现、可回归、可可视化追踪的能力节点。

本轮也暴露了一个需要长期警惕的风险：形态相近的算子不能只复用旧 runtime 名称或 layer kind。`ReduceSum` 被误用为 `ReduceMean` 就是典型例子。后续新增同族算子时，应在最小数值用例里覆盖能区分语义的样本，避免结构 PASS 但数值语义错误。

## 后续建议

下一轮继续选择“简单但能显著扩张图谱”的官方算子：

- `Where` 静态 mask
- 比较类布尔输出算子：`Equal`、`Greater`、`Less`
- 分类辅助算子：`ArgMax`
- 归约扩展：`ReduceProd`、`ReduceL1`、`ReduceL2`

同时建议继续从 `NET_004` KWS-style 或其他真实时序网络中反向提炼 case，确保图谱增长与真实嵌入式模型输入空间保持同步。
