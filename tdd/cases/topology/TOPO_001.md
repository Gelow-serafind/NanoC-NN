# TOPO_001: Conv→Relu→Pool→Flatten→FC 经典 CNN 分类头

## 验证目标

验证经典 CNN 分类链路的全链路数据流正确性，包括算子间 buffer 切换、NCHW→NHWC 布局处理、Flatten 折叠后 FC 输入维度推导。

## 网络结构

```
Q/DQ Input(1,3,8,8) → Conv(3→8,k3,SAME) → Relu → MaxPool(k2,s2) → Flatten → FC(128→10) → Q/DQ Output
```

## 输入

- **张量形状**: `[1, 3, 8, 8]`（NCHW: batch=1, channels=3, H=8, W=8）
- **数据类型**: int8

## 算子参数

| 算子 | 参数 |
|------|------|
| Conv | kernel=[3,3], stride=[1,1], pads=[1,1,1,1], group=1, with_bias=True, weight shape=[8,3,3,3] |
| Relu | 融合到 Conv 的 activation_min=0, activation_max=127 |
| MaxPool | kernel_shape=[2,2], strides=[2,2], pads=[0,0,0,0] |
| Flatten | axis=1 |
| FC (Gemm) | transB=1, with_bias=True, weight shape=[10, 128] |

## 量化设计

| 张量 | scale | zero_point | 说明 |
|------|-------|-----------|------|
| input | 0.01 | 0 | 对称 |
| conv_weight | 0.02 | 0 | 对称 |
| conv_output | 0.03 | 0 | 对称（同时是 Pool 输入） |
| pool_output | 0.03 | 0 | 与 pool 输入一致（满足约束） |
| fc_weight | 0.01 | 0 | 对称 |
| fc_output | 0.05 | 0 | 对称 |

## 预期结果

- **codegen status**: `ok`
- **若 ok**: 编译通过、运行不崩溃
- **关键验证点**:
  - Conv 输出 shape: [1,8,8,8] → Pool 后: [1,8,4,4] → Flatten 后: [1,128] → FC 输出: [1,10]
  - Relu 融合到 Conv（activation_min=0）
  - Pool 输入/输出量化参数一致
  - FC 输入维度 = 8*4*4 = 128

## 边界/风险

- 多算子串接时 buffer ping-pong 切换是否正确
- Flatten 在 NHWC 布局下的展开顺序是否匹配 FC 权重的期望
- Conv 输出 → Pool 输入的中间量化参数必须一致
- 这是代码生成器"核心价值路径"的最小完整验证
