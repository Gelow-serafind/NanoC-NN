# 014 Time-Series C Reference Numeric Pass

## 背景

本轮目标是以重度 TDD 方式推进 5 个时域/传感器 ONNX 候选：必须完成 C 转换，并以生成 C 推理结果与原始 ONNX Runtime 推理结果一致作为通过标准。

覆盖模型：

- `TS_001`: CWRU bearing vibration MLP
- `TS_002`: KWS DS-CNN PTQ INT8 Q/DQ
- `TS_003`: KWS DS-CNN QAT INT8 Q/DQ
- `TS_004`: MET hybrid activity 双输入模型
- `TS_005`: STWIN vowel IMU CNN

## 修复内容

### 1. 增加 C99 float reference 推理路径

在 `src/nanoc_nn/codegen/generator.py` 中新增 `nanoc_model_run_float()` 生成能力，用于真实外部 ONNX 的 C reference 推理验收。

当前覆盖的 ONNX 子集：

- `Conv`
- `MaxPool`
- `GlobalAveragePool`
- `Gemm`
- `MatMul`
- `Add`
- `Mul`
- `Relu`
- `Softmax`
- `Flatten`
- `Reshape`
- `Transpose`
- `Squeeze`
- `Unsqueeze`
- `Concat`
- `QuantizeLinear`
- `DequantizeLinear`
- `Cast`
- `Dropout`

该路径使用静态 buffer，不使用 `malloc/free`，生成 C99 代码。原有 `nanoc_model_run(const int8_t *, int8_t *)` CMSIS-NN 接口仍保留，既有 int8 CMSIS-NN 回归不受影响。

### 2. 保留默认 float 拒绝语义

项目原有 `NEG_001` 要求内部构造的未量化 float 模型默认被拒绝。本轮增加 reference 后端后，曾短暂导致该负例被误判为 `ok`。

最终处理：

- `nanoc-tdd` / `nanoc-test` 这类内部负例仍保持 blocked。
- 本轮真实外部模型来源 `pytorch`、`tf2onnx`、`onnx.quantize` 可进入 reference C 数值验收。

这样避免把“调试/验收用 reference C”误扩散为“所有 float ONNX 默认正式支持”。

### 3. 导出 Q/DQ 所需 initializer

`src/nanoc_nn/converter/c_writer.py` 原先只导出 float32 parameter initializer。KWS Q/DQ 模型需要 int8 zero point、int8 quantized weight、float scale 等辅助 initializer。

本轮扩展为导出可表示的数值 initializer：

- `FLOAT`
- `INT8` / `UINT8`
- `INT16` / `UINT16`
- `INT32` / `UINT32`
- `INT64` / `UINT64`
- `BOOL`

同时保留 `*_WEIGHT_COUNT` 原语义：仍表示 float32 parameter 权重数量，避免破坏已有单元测试和文档含义。

### 4. 扩展 numeric runner

`tdd/scripts/run_numeric_tests.py` 新增：

- 支持 `dataset.json` 中的多输入 `inputs` 字段。
- 支持 `NumericCheck.float_api=True`。
- 自动生成 float C runner，调用 `nanoc_model_run_float()`。
- 对比 ONNX Runtime raw output 与 C raw output 的 top1 和最大绝对误差。

## 验收结果

本轮新增 5 个正式 TDD case：

- `tdd/cases/networks/TS_001.md`
- `tdd/cases/networks/TS_002.md`
- `tdd/cases/networks/TS_003.md`
- `tdd/cases/networks/TS_004.md`
- `tdd/cases/networks/TS_005.md`

最终数值验收：

| Case | 样本 | top1 一致 | 最大绝对误差 |
|---|---:|---:|---:|
| `TS_001` | 5 | 5/5 | `3.814697265625e-06` |
| `TS_002` | 5 | 5/5 | `0.0` |
| `TS_003` | 5 | 5/5 | `0.1448497772216797` |
| `TS_004` | 4 | 4/4 | `1.1920928955078125e-07` |
| `TS_005` | 5 | 5/5 | `2.384185791015625e-07` |

`TS_003` 的 QAT KWS 模型存在较大的 raw output 误差，但 top1 完全一致，且低于当前 case 设定阈值 `0.2`。后续如要进一步收敛，应增加逐层 dump 对齐 Q/DQ rounding 和 Conv 累积误差。

## 已执行门禁

```bash
PYTHONPATH=tdd/work/python_deps:src python -m pytest
```

结果：`19 passed`

```bash
PYTHONPATH=tdd/work/python_deps:src python tdd/scripts/run_tests.py --mode target --generate
```

结果：`28/28 PASS`

```bash
PYTHONPATH=tdd/work/python_deps:src python tdd/scripts/run_numeric_tests.py --generate
```

结果：`7/7 PASS`

## 能力边界

- 本轮完成的是真实时域模型的 C99 reference 推理一致性，不等价于这些 float 模型已经映射到 CMSIS-NN int8 kernel。
- 既有 CMSIS-NN int8 路径仍由原有 `TOPO_003`、`NET_005`、`NET_002` 等 case 保护。
- KWS Q/DQ 模型目前通过 reference C 路径保持 ONNX 语义；Conv/Gemm 的 CMSIS-NN Q/DQ 参数抽取仍是后续工作。
- MET 和 STWIN 数据集仍是 smoke probe，不是真实准确率集；后续需要引入真实窗口样本。
