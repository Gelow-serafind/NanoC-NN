# SOFTMAX_001: 10 分类标准

## 验证目标

验证 Softmax 的 multiplier/shift/diff_min 参数计算和 CMSIS-NN 调用生成。

## 网络结构

```
Q/DQ Input → Softmax → Q/DQ Output
```

## 输入

- **张量形状**: `[1, 10]`
- **数据类型**: int8

## 算子参数

| 算子 | 参数 |
|------|------|
| Softmax | axis=1（默认最后一维） |

## 量化设计

| 张量 | scale | zero_point | 说明 |
|------|-------|-----------|------|
| input | 0.05 | 0 | 对称 |
| output | 1/256 (≈0.00390625) | -128 | Softmax 标准输出量化 |

## 预期结果

- **codegen status**: `ok`
- **若 ok**: 编译通过、运行不崩溃
- **关键验证点**: 生成代码包含 `arm_softmax_s8` 调用；multiplier/shift/diff_min 参数不为 0 且不溢出

## 边界/风险

- Softmax 的量化参数计算涉及 log2 和定点数转换，容易出现精度问题
- num_rows=1, row_size=10：标准 10 分类场景
- diff_min 参数如果计算错误会导致 Softmax 输出全为同一值
