# TOPO_003: 真实 MNIST QLinear int8 分类链路

## 验证目标

固化真实 MNIST int8 ONNX 的 QLinear 量化链路，验证 converter 能解析直接 QLinear 形式，codegen 能生成包含 Conv、MaxPool、FC 和 Add 的 CMSIS-NN C 推理路径。

## 来源

缺陷复现 / 真实模型回归：`mnist-12-int8.onnx`，来自 ONNX Model Zoo MNIST int8 模型镜像。本用例用于固化此前 MNIST 暴露的问题，包括 QLinear 算子支持、uint8 量化域转换、QLinearMatMul 权重方向和 QLinearAdd 参数生成。

## 网络结构

```
Float Input(1,1,28,28)
  -> QuantizeLinear
  -> QLinearConv
  -> MaxPool
  -> QLinearConv
  -> DequantizeLinear
  -> MaxPool
  -> Reshape
  -> QuantizeLinear
  -> QLinearMatMul
  -> QLinearAdd
  -> DequantizeLinear
  -> Float Output(1,10)
```

## 输入

- **张量形状**: `[1, 1, 28, 28]`（NCHW: batch=1, channels=1, H=28, W=28）
- **ONNX 边界数据类型**: float32
- **运行期核心数据类型**: int8

## 算子参数

| 算子 | 参数 |
|------|------|
| QLinearConv x2 | 标准 MNIST 卷积层，权重量化参数来自 initializer |
| MaxPool x2 | 池化参数来自 ONNX node attributes |
| Reshape | 将卷积特征展平到 FC 输入 |
| QLinearMatMul | FC 主计算，要求权重方向转换正确 |
| QLinearAdd | FC bias/残差式加法，要求生成 elementwise add 量化参数 |

## 量化设计

| 张量 | scale | zero_point | 说明 |
|------|-------|-----------|------|
| Input3 | initializer | initializer | ONNX float 输入经 QuantizeLinear 进入量化域 |
| QLinearConv weights | initializer | initializer | 真实模型直接提供量化权重 |
| Pool/MatMul/Add 中间张量 | initializer | initializer | 由真实 ONNX 量化参数驱动 |
| Plus214_Output_0 | initializer | initializer | DequantizeLinear 输出到 float 边界 |

## 预期结果

- **codegen status**: `ok`
- **若 ok**: `model.c` 必须包含真实 CMSIS-NN 调用，C99 smoke compile 通过，运行 smoke binary 不崩溃
- **关键验证点**:
  - 解析并生成 `QLinearConv` -> `arm_convolve_wrapper_s8`
  - 解析并生成 `MaxPool` -> `arm_max_pool_s8`
  - 解析并生成 `QLinearMatMul` -> `arm_fully_connected_s8`
  - 解析并生成 `QLinearAdd` -> `arm_elementwise_add_s8`
  - uint8 ONNX 量化域转换到 int8 runtime 时 zero point 不丢失
  - QLinearMatMul 权重方向与 CMSIS-NN FC 输入约定一致

## 边界/风险

- 这是直接 QLinear 模型，不是合成 Q/DQ float wrapper 模型。
- ONNX 输入/输出是 float32，但核心推理链路必须落到 int8 CMSIS-NN。
- `status: ok` 不能单独代表成功，必须以生成物和 C99 smoke compile 为准。
- fixture 是二进制 ONNX 文件，更新时需要同步确认来源和哈希。
