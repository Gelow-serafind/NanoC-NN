# CONV_001: 1×1 pointwise 单通道

## 验证目标

验证退化为逐通道缩放的 pointwise 卷积的权重布局转换（OIHW→OHWI）和最简 Conv 代码生成。

## 来源

内部探索：建立 Conv2D int8 主链路的最小基线。

## 网络结构

```
Q/DQ Input → Conv(kernel=1×1, group=1) → Q/DQ Output
```

## 输入

- **张量形状**: `[1, 1, 4, 4]`（NCHW: batch=1, channels=1, H=4, W=4）
- **数据类型**: int8

## 算子参数

| 算子 | 参数 |
|------|------|
| Conv | kernel=[1,1], stride=[1,1], pads=[0,0,0,0], group=1, with_bias=True, weight shape=[1,1,1,1] |

## 量化设计

| 张量 | scale | zero_point | 说明 |
|------|-------|-----------|------|
| input | 0.01 | 0 | 对称 |
| weight | 0.02 | 0 | 对称 |
| output | 0.05 | 0 | 对称 |

## 预期结果

- **codegen status**: `ok`
- **若 ok**: 编译通过、运行不崩溃
- **关键验证点**: 生成代码包含 `arm_convolve_wrapper_s8` 调用；权重仅 1 字节；输出 shape 保持 [1,1,4,4]

## 边界/风险

- kernel=1×1, in_ch=1, out_ch=1：最简 Conv，权重只有 1 个元素
- OIHW→OHWI 权重重排在单元素时退化为恒等变换，不容易暴露重排 bug
- 但作为基线确保最简路径不崩溃
