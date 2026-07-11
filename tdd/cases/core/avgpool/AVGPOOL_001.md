# AVGPOOL_001: QLinearGlobalAveragePool int8 全局池化

## 验证目标

验证 SqueezeNet 分类头中出现的 `QLinearGlobalAveragePool` 最小形态可以生成真实 CMSIS-NN `arm_avgpool_s8` 调用，并且进入独立算子能力集保护，而不是只被完整网络间接覆盖。

## 来源

内部探索 / `NET_001` SqueezeNet 反向提炼。

本用例来自 SqueezeNet 尾部：

```text
QLinearConv -> QLinearGlobalAveragePool -> DequantizeLinear -> Softmax
```

当前先提炼其中的量化全局平均池化，作为 ONNX 导图里 `QLinearGlobalAveragePool` 节点的独立 TDD 分支。

## 网络结构

```text
Input(1,4,3,3)
  -> QuantizeLinear
  -> QLinearGlobalAveragePool
  -> DequantizeLinear
  -> Output(1,4,1,1)
```

## 输入

- **张量形状**: `[1, 4, 3, 3]`（NCHW）
- **ONNX 边界数据类型**: float32
- **运行期核心数据类型**: uint8/int8 量化张量

## 算子参数

| 算子 | 参数 |
|------|------|
| QuantizeLinear | scale=0.05, zp=128, uint8 |
| QLinearGlobalAveragePool | kernel 覆盖完整 H/W，即 3x3 |
| DequantizeLinear | scale=0.05, zp=128, uint8 |

## 量化设计

| 张量 | scale | zero_point | 说明 |
|------|-------|------------|------|
| input/input_q | 0.05 | 128 | float 输入进入 uint8 量化域 |
| pool_out/output | 0.05 | 128 | 与输入保持一致，用于先保护 CMSIS-NN avgpool 基本路径 |

## 预期结果

- **codegen status**: `ok`
- 生成代码包含 `arm_avgpool_s8`
- 生成物通过 C99 smoke compile
- 输出 shape 为 `[1,4,1,1]`

## 边界/风险

- 当前只确认静态 rank=4 NCHW、全局空间池化、输入输出量化参数可由 QLinear 节点直接解析的形态。
- 普通 `AveragePool`、不同输入输出量化参数、NHWC 原生图、float-domain `GlobalAveragePool` 仍需后续独立 case。
- 该用例只解决 SqueezeNet 的全局池化分支，不代表 SqueezeNet 末端 float `Softmax` 已经完成。
