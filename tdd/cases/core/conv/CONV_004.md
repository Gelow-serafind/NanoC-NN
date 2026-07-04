# CONV_004: 非对称输入 zp≠0

## 验证目标

验证非对称输入量化时 Conv 的 input_offset 是否正确设置为 -zero_point。

## 来源

内部探索：覆盖 Conv2D int8 非对称输入量化路径。

## 网络结构

```
Q/DQ Input(zp=12) → Conv(kernel=3×3, padding=SAME) → Q/DQ Output
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
| input | 0.01 | 12 | 非对称 |
| weight | 0.02 | 0 | 对称 |
| output | 0.05 | 0 | 对称 |

## 预期结果

- **codegen status**: `ok`
- **若 ok**: 编译通过、运行不崩溃
- **关键验证点**: 生成代码中 input_offset 值为 -12；权重 offset 为 0

## 边界/风险

- 输入 zp=12：非零 zero_point 在 Conv 中同样需要 input_offset=-zp
- 如果 codegen 对 Conv 和 Gemm 使用不同的 offset 计算路径，可能在其中一个遗漏符号取反
- Conv 的 input_offset 影响所有 im2col + GEMM 计算，错误会导致输出完全偏移
