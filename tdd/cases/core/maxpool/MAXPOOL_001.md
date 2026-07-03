# MAXPOOL_001: 标准 2×2 stride=2

## 验证目标

验证 MaxPool 的基本参数映射和量化一致性约束满足时的正确代码生成。

## 来源

内部探索：建立 pooling int8 路径的最小稳定能力。

## 网络结构

```
Q/DQ Input → MaxPool(kernel=2×2, stride=2) → Q/DQ Output
```

## 输入

- **张量形状**: `[1, 4, 8, 8]`（NCHW: batch=1, channels=4, H=8, W=8）
- **数据类型**: int8

## 算子参数

| 算子 | 参数 |
|------|------|
| MaxPool | kernel_shape=[2,2], strides=[2,2], pads=[0,0,0,0] |

## 量化设计

| 张量 | scale | zero_point | 说明 |
|------|-------|-----------|------|
| input | 0.05 | 0 | 对称 |
| output | 0.05 | 0 | 对称，与 input 一致（满足 Pool 约束） |

## 预期结果

- **codegen status**: `ok`
- **若 ok**: 编译通过、运行不崩溃
- **关键验证点**: 生成代码包含 `arm_max_pool_s8` 调用；输出 shape 为 [1,4,4,4]；输入/输出量化参数一致

## 边界/风险

- Pool 要求输入/输出量化参数严格一致，本用例满足该约束
- 输出 H/W 减半：8/2=4
- 作为 Pool 基线用例确保正常路径不崩溃
