# GEMM_002: 非对称输入 zp≠0 含 bias

## 验证目标

验证非对称输入量化时 `input_offset = -zero_point` 是否正确传入 CMSIS-NN，以及 bias int32 量化系数是否正确计算。

## 网络结构

```
Q/DQ Input(zp=10) → Gemm(transB=1, with_bias) → Q/DQ Output
```

## 输入

- **张量形状**: `[1, 4]`
- **数据类型**: int8

## 算子参数

| 算子 | 参数 |
|------|------|
| Gemm | transB=1, with_bias=True, weight shape=[8, 4] |

## 量化设计

| 张量 | scale | zero_point | 说明 |
|------|-------|-----------|------|
| input | 0.01 | 10 | 非对称 |
| weight | 0.02 | 0 | 对称 |
| bias | input_scale * weight_scale | — | int32 量化 |
| output | 0.05 | 0 | 对称 |

## 预期结果

- **codegen status**: `ok`
- **若 ok**: 编译通过、运行不崩溃
- **关键验证点**: 生成代码中 input_offset 值为 -10（而非 +10）；bias 量化使用 input_scale * weight_scale

## 边界/风险

- 输入 zp=10：非零 zero_point 是嵌入式量化中的常见场景
- input_offset 符号错误是高频 bug：应为 `-zp` 而非 `+zp`
- bias 量化系数计算依赖 input_scale 和 weight_scale 的乘积
