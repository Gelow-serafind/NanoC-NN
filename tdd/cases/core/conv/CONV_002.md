# CONV_002: 3×3 标准 SAME padding

## 验证目标

验证标准 3×3 卷积的完整参数映射，包括 SAME padding 计算、多通道权重布局转换、multiplier/shift 数组生成。

## 来源

内部探索：覆盖 CNN 中最高频的 3x3 SAME padding Conv2D 形态。

## 网络结构

```
Q/DQ Input → Conv(kernel=3×3, padding=SAME_UPPER) → Q/DQ Output
```

## 输入

- **张量形状**: `[1, 3, 8, 8]`（NCHW: batch=1, channels=3, H=8, W=8）
- **数据类型**: int8

## 算子参数

| 算子 | 参数 |
|------|------|
| Conv | kernel=[3,3], stride=[1,1], pads=[1,1,1,1], group=1, with_bias=True, weight shape=[8,3,3,3] |

## 量化设计

| 张量 | scale | zero_point | 说明 |
|------|-------|-----------|------|
| input | 0.01 | 0 | 对称 |
| weight | 0.02 | 0 | 对称 |
| output | 0.05 | 0 | 对称 |

## 预期结果

- **codegen status**: `ok`
- **若 ok**: 编译通过、运行不崩溃
- **关键验证点**: padding 参数正确映射到 CMSIS-NN 格式；权重 8*3*3*3=216 字节完整生成；输出 shape 为 [1,8,8,8]

## 边界/风险

- SAME padding 需要 codegen 正确计算 pad_top/bottom/left/right
- 多通道输入(3)到多通道输出(8)的权重重排 OIHW→OHWI 涉及 4 维转置
- 权重 216 个 int8 元素，验证数组不截断
