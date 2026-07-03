# TOPO_002: Conv1d→Relu→Conv1d→Relu→Flatten→FC 时序分类链路

## 验证目标

验证 ONNX rank-3 Conv1d 网络可以被解析为 int8 Q/DQ 合同，并由 codegen 映射到 CMSIS-NN `arm_convolve_wrapper_s8` 的高度为 1 的 2D 卷积调用。

## 来源

用户反馈：参考 `NanoC-NN-ONNX-Examples/example-2-detect-signal-jump` 的简单完整时序分类网络，扩展当前 TDD 网络覆盖面。

## 网络结构

```
Q/DQ Input(1,1,10) → Conv1d(1→8,k2) → Relu → Conv1d(8→8,k2) → Relu → Flatten → FC(64→3) → Q/DQ Output
```

## 输入

- **张量形状**: `[1, 1, 10]`（NCL: batch=1, channels=1, length=10）
- **数据类型**: int8

## 算子参数

| 算子 | 参数 |
|------|------|
| Conv1d #1 | kernel=[2], stride=[1], pads=[0,0], group=1, with_bias=True, weight shape=[8,1,2] |
| Relu #1 | 融合到 Conv1d #1 的 activation_min=0, activation_max=127 |
| Conv1d #2 | kernel=[2], stride=[1], pads=[0,0], group=1, with_bias=True, weight shape=[8,8,2] |
| Relu #2 | 融合到 Conv1d #2 的 activation_min=0, activation_max=127 |
| Flatten | axis=1 |
| FC (Gemm) | transB=1, with_bias=True, weight shape=[3,64] |

## 量化设计

| 张量 | scale | zero_point | 说明 |
|------|-------|-----------|------|
| input | 0.02 | 0 | 对称 |
| conv_weight | 0.02 | 0 | 对称 |
| conv_output | 0.03 | 0 | 对称 |
| fc_weight | 0.02 | 0 | 对称 |
| fc_output | 0.05 | 0 | 对称 |

## 预期结果

- **codegen status**: `ok`
- **若 ok**: 编译通过、运行不崩溃
- **关键验证点**:
  - Conv1d 输入 `[1,1,10]` 映射为 CMSIS-NN input dims `{1,1,10,1}`
  - Conv1d 权重 `[O,I,K]` 映射为高度 1 的 OHWI 排布
  - 两层 Conv1d 后长度从 10 变为 8，Flatten 后 FC 输入维度为 64
  - 生成代码包含 `arm_convolve_wrapper_s8` 和 `arm_fully_connected_s8`

## 边界/风险

- 该用例只覆盖 group=1、dilation=1、NCL 输入布局的 Conv1d 正向路径。
- Conv1d 通过 CMSIS-NN 2D wrapper 表达，后续需要继续补充 padding、stride>1、不同通道数和真实板端数值对齐测试。
