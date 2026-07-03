# GEMM_004: 中等规模 64→32

## 验证目标

验证中等规模 FC 层的权重数组正确性和 buffer 预算合理性。

## 网络结构

```
Q/DQ Input → Gemm(transB=1, with_bias) → Q/DQ Output
```

## 输入

- **张量形状**: `[1, 64]`
- **数据类型**: int8

## 算子参数

| 算子 | 参数 |
|------|------|
| Gemm | transB=1, with_bias=True, weight shape=[32, 64] |

## 量化设计

| 张量 | scale | zero_point | 说明 |
|------|-------|-----------|------|
| input | 0.01 | 0 | 对称 |
| weight | 0.005 | 0 | 对称 |
| output | 0.02 | 0 | 对称 |

## 预期结果

- **codegen status**: `ok`
- **若 ok**: 编译通过、运行不崩溃
- **关键验证点**: 权重数组 2048 字节完整生成；multiplier/shift 计算不溢出；memory report 中 Flash 估算包含权重大小

## 边界/风险

- 权重 64*32=2048 个 int8 元素：验证大数组的正确导出
- multiplier 计算涉及 input_scale * weight_scale / output_scale，数值范围合理性
- buffer 预算报告应正确反映输入 64 字节 + 输出 32 字节 + 权重 2048 字节
