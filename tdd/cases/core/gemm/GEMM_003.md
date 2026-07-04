# GEMM_003: 非 4 对齐维度 13→7

## 验证目标

验证 CMSIS-NN `arm_fully_connected_s8` 在输入/输出维度均不被 4 整除时的正确处理。

## 来源

内部探索：覆盖 MCU int8 FC 常见的非对齐维度边界。

## 网络结构

```
Q/DQ Input → Gemm(transB=1, with_bias) → Q/DQ Output
```

## 输入

- **张量形状**: `[1, 13]`
- **数据类型**: int8

## 算子参数

| 算子 | 参数 |
|------|------|
| Gemm | transB=1, with_bias=True, weight shape=[7, 13] |

## 量化设计

| 张量 | scale | zero_point | 说明 |
|------|-------|-----------|------|
| input | 0.03 | 0 | 对称 |
| weight | 0.02 | 0 | 对称 |
| output | 0.04 | 0 | 对称 |

## 预期结果

- **codegen status**: `ok`
- **若 ok**: 编译通过、运行不崩溃
- **关键验证点**: 权重数组大小正确为 7*13=91 字节；buffer 分配不因对齐假设而越界

## 边界/风险

- in_features=13, out_features=7：均不被 4 整除
- CMSIS-NN 内部对 SIMD 有 4 字节对齐优化，非对齐维度可能触发内部 padding 逻辑
- 权重数组 layout 在非对齐时如果按 4 对齐分配，可能读到垃圾数据
