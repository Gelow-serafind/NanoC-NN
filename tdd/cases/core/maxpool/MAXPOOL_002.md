# MAXPOOL_002: SqueezeNet 风格 DQ/MaxPool/Q int8 边界

## 验证目标

验证真实 SqueezeNet 中出现的 `DequantizeLinear -> MaxPool -> QuantizeLinear` 浮点域池化边界，可以被 converter 反向识别为 int8 MaxPool contract，并生成真实 CMSIS-NN `arm_max_pool_s8` 调用。

## 来源

内部探索 / `NET_001` SqueezeNet 反向提炼。

SqueezeNet 的 `pool3_1` 和 `pool5_1` 节点不是直接以量化 tensor 作为 ONNX 输入输出，而是处在 DQ/Q 包围的 float-domain 边界中。本用例用于防止该类边界被误判为缺少 int8 量化字段。

## ONNX Schema 归属

| 字段 | 值 |
|------|------|
| domain | `ai.onnx` |
| op_type | `MaxPool` |
| opset_range | `11+` |
| schema_form | rank=4 static NCHW, DQ/MaxPool/Q int8 boundary, same quantization |
| lowering | `cmsis_maxpool_s8` |
| backend | `cmsis-nn` |

## 网络结构

```text
Input(1,4,4,4)
  -> QuantizeLinear
  -> DequantizeLinear
  -> MaxPool(kernel=2x2, stride=2)
  -> QuantizeLinear
  -> DequantizeLinear
  -> Output(1,4,2,2)
```

## 输入

- **张量形状**: `[1, 4, 4, 4]`（NCHW）
- **数据类型**: int8 运行期，float32 ONNX 边界

## 算子参数

| 算子 | 参数 |
|------|------|
| QuantizeLinear | scale=0.1, zp=0, int8 |
| MaxPool | kernel_shape=[2,2], strides=[2,2], pads=[0,0,0,0] |
| DequantizeLinear | scale=0.1, zp=0, int8 |

## 量化设计

| 张量 | scale | zero_point | 说明 |
|------|-------|------------|------|
| input/input_q/input_dq | 0.1 | 0 | 与输出保持一致 |
| pool_out/pool_q/output | 0.1 | 0 | MaxPool 不做 requantize |

## 预期结果

- **codegen status**: `ok`
- **若 ok**: 编译通过、运行不崩溃
- **关键验证点**: converter 为 `MaxPool` 写入 node quantization，生成代码包含 `arm_max_pool_s8`

## 数值验收

- **是否需要**: `no`
- **数据集**: 暂无
- **参考路径**: 后续可补 ONNX Runtime vs C 最小数值 smoke
- **通过阈值**: 暂不设置

## 边界/风险

- 当前只确认输入输出量化参数一致的 MaxPool。
- 不同 scale/zero_point 的 Pool 仍应保持 blocked，直到 renderer 支持显式 requantize。
- 本用例是推进 `NET_001` 的中间能力，不代表 SqueezeNet 完整网络已可交付。
