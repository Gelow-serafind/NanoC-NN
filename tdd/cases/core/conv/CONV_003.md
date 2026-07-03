# CONV_003: stride=2 下采样

## 验证目标

验证 stride>1 时的输出 shape 推导正确性和 CMSIS-NN stride 参数映射。

## 来源

内部探索：覆盖 Conv2D 下采样路径和输出 shape 边界。

## 网络结构

```
Q/DQ Input → Conv(kernel=3×3, stride=2, padding=SAME) → Q/DQ Output
```

## 输入

- **张量形状**: `[1, 4, 8, 8]`（NCHW: batch=1, channels=4, H=8, W=8）
- **数据类型**: int8

## 算子参数

| 算子 | 参数 |
|------|------|
| Conv | kernel=[3,3], stride=[2,2], pads=[1,1,1,1], group=1, with_bias=True, weight shape=[8,4,3,3] |

## 量化设计

| 张量 | scale | zero_point | 说明 |
|------|-------|-----------|------|
| input | 0.01 | 0 | 对称 |
| weight | 0.02 | 0 | 对称 |
| output | 0.03 | 0 | 对称 |

## 预期结果

- **codegen status**: `ok`
- **若 ok**: 编译通过、运行不崩溃
- **关键验证点**: 输出 shape 为 [1,8,4,4]（H/W 减半）；stride 参数正确传入 CMSIS-NN；输出 buffer 大小按 8*4*4=128 分配

## 边界/风险

- stride=2 使输出空间维度减半，如果 shape 推导错误会导致 buffer 越界
- stride 参数映射到 CMSIS-NN 的 stride_height/stride_width 字段
- padding 在 stride>1 时的实际填充量计算更复杂
