# GEMM_001: 最小对称 FC 无 bias

## 验证目标

验证最小 FC 层在全对称量化、无 bias 条件下的 transB=1 权重映射与代码生成正确性。

## 网络结构

```
Q/DQ Input → Gemm(transB=1, no bias) → Q/DQ Output
```

## 输入

- **张量形状**: `[1, 2]`
- **数据类型**: int8

## 算子参数

| 算子 | 参数 |
|------|------|
| Gemm | transB=1, with_bias=False, weight shape=[3, 2] |

## 量化设计

| 张量 | scale | zero_point | 说明 |
|------|-------|-----------|------|
| input | 0.01 | 0 | 对称 |
| weight | 0.02 | 0 | 对称 |
| output | 0.05 | 0 | 对称 |

## 预期结果

- **codegen status**: `ok`
- **若 ok**: 编译通过、运行不崩溃
- **关键验证点**: 生成代码包含 `arm_fully_connected_s8` 调用；无 bias 时不引用 bias 数组

## 边界/风险

- in_features=2, out_features=3：极小尺寸，验证循环边界不出错
- 无 bias：验证 codegen 在 bias 缺失时的处理路径（传 NULL 或全零数组）
- 全对称量化：最简路径，排除 offset 计算干扰
