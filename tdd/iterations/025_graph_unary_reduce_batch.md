# 025: 官方 unary/reduce/shape 批量 TDD 迭代

## 背景

上一轮 `Sub`、`Div`、`Sigmoid`、`Pad`、`Slice`、`Gather` 证明了 schema-driven TDD 可以成批扩展生成型算子。本轮继续从 ONNX 官方图谱中选择低算力嵌入式模型常见、实现风险可控、适合一次性闭环的算子：

- `Tanh`
- `LeakyRelu`
- `Clip`
- `Neg`
- `Sqrt`
- `Reciprocal`
- `ReduceMean`
- `Unsqueeze`

目标不是追求立即最优 CMSIS-NN kernel，而是先建立“ONNX-vs-C 数值一致”的 correctness baseline，再在后续迭代中替换为 LUT、定点近似或平台优化。

## 新增用例

| case | ONNX op | 支持形态 | 数值数据集 |
|------|---------|----------|------------|
| `TANH_001` | `Tanh` | 静态 QDQ/int8 tensor | `tanh_qdq_smoke` |
| `LEAKYRELU_001` | `LeakyRelu` | 静态 QDQ/int8 tensor，`alpha=0.1` | `leakyrelu_qdq_smoke` |
| `CLIP_001` | `Clip` | 静态 QDQ/int8 tensor，常量 min/max | `clip_qdq_smoke` |
| `NEGOP_001` | `Neg` | 静态 QDQ/int8 tensor | `neg_qdq_smoke` |
| `SQRT_001` | `Sqrt` | 静态 QDQ/int8 tensor，非负输入域 | `sqrt_qdq_smoke` |
| `RECIPROCAL_001` | `Reciprocal` | 静态 QDQ/int8 tensor，非零有限输入域 | `reciprocal_qdq_smoke` |
| `REDUCEMEAN_001` | `ReduceMean` | rank=2，`axes=[1]`，`keepdims=1` | `reducemean_qdq_smoke` |
| `UNSQUEEZE_001` | `Unsqueeze` | 静态 data-path QDQ/int8，常量 axes，元素数量不变 | `unsqueeze_qdq_smoke` |

## 修复与实现

1. converter 增加官方算子白名单、initializer role 和属性归一化：
   - `Clip` 支持 legacy attribute 与 opset 11+ 的 min/max 输入。
   - `LeakyRelu` 归一化 `alpha`。
   - `ReduceMean` 归一化 `axes/keepdims/noop_with_empty_axes`。
   - `Unsqueeze` 的 axes initializer 作为辅助输入。

2. converter 增加 QDQ/int8 量化提取：
   - 通用 unary 提取器覆盖 `Tanh/LeakyRelu/Clip/Neg/Sqrt/Reciprocal`。
   - `ReduceMean` 当前锁定 rank2、`axes=[1]`、`keepdims=1`。
   - `Unsqueeze` 区分 shape-helper 与 data-path，data-path 必须保持同量化和同元素数量。

3. mapper/codegen 增加 runtime lowering：
   - `generated_c_tanh_s8`
   - `generated_c_leakyrelu_s8`
   - `generated_c_clip_s8`
   - `generated_c_neg_s8`
   - `generated_c_sqrt_s8`
   - `generated_c_reciprocal_s8`
   - `generated_c_reducemean_s8`
   - `generated_c_unsqueeze_s8`

4. 修复 `Clip` standalone data-path 假阳性风险：
   - 旧逻辑把 `Clip` 放在 tensor alias 列表中，用于 activation fusion 场景。
   - 当 `Clip` 作为独立输出算子时，alias 会把输出解析回只读 `input`，导致生成 C 试图写入 `const int8_t *input`。
   - 本轮将 standalone `Clip` 从 alias 中移除，并作为独立 runtime op 生成 C。

## 数值规则

`LeakyRelu` 和 `ReduceMean` 当前允许一格输出量化误差：

- `LEAKYRELU_001`: `max_abs_error <= 0.06`
- `REDUCEMEAN_001`: `max_abs_error <= 0.06`

这是 QDQ/int8 correctness baseline 的离散量化误差，不是放弃精度约束。后续若实现整数 rounding 与 ONNX Runtime 完全一致，可再收紧。

## 验证结果

执行命令：

```bash
PYTHONPATH=src python tdd/scripts/run_tests.py --mode target --generate
PYTHONPATH=src python tdd/scripts/run_numeric_tests.py --generate
PYTHONPATH=src python tdd/scripts/run_regression.py --generate
PYTHONPATH=src python tdd/scripts/generate_support_map.py
```

结果：

- target: `57/57 PASS`
- baseline structural: `39/39 PASS`
- numeric: `35/35 PASS`
- regression: `PASS`

## 能力集变化

本轮新增 8 个官方 ONNX schema 子形态进入 `CAPABILITIES.md`：

- `ONNX_TANH_QDQ_INT8`
- `ONNX_LEAKYRELU_QDQ_INT8`
- `ONNX_CLIP_QDQ_INT8`
- `ONNX_NEG_QDQ_INT8`
- `ONNX_SQRT_QDQ_INT8`
- `ONNX_RECIPROCAL_QDQ_INT8`
- `ONNX_REDUCEMEAN_QDQ_INT8`
- `ONNX_UNSQUEEZE_QDQ_INT8`

能力图谱 `tdd/reports/onnx_support_map.html` 已同步刷新。

## 后续建议

下一轮继续从官方图谱选择可批量闭环的简单算子：

- `Where` 静态 mask
- `Min/Max` 逐元素
- `ArgMax` 分类输出
- `ReduceMax/ReduceSum` 小型归约

同时建议将 `NET_004` KWS-style 升级为正式 ONNX-vs-C 数值验收候选，用真实音频/特征数据继续反向提炼缺口。
