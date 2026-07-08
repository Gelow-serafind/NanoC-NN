# NanoC-NN 能力集

> 本文件由 `python tdd/scripts/run_tests.py --mode target` 自动生成，禁止手动编辑。
> 最后更新: 2026-07-06T02:36:19.854203
> Git commit: `34e7352`

**能力集大小: 23 / 23 (100%)**

## 已验证能力 (PASS)

以下用例通过测试，代表代码生成器已确认支持的能力：

| 用例 ID | 分类 | 能力描述 | 验证日期 |
|---------|------|---------|---------|
| CONV_001 | core/conv | 1x1 pointwise 单通道 | 2026-07-06 |
| CONV_002 | core/conv | 3x3 标准 SAME padding | 2026-07-06 |
| CONV_003 | core/conv | stride=2 下采样 | 2026-07-06 |
| CONV_004 | core/conv | 非对称输入 zp!=0 | 2026-07-06 |
| GEMM_001 | core/gemm | 最小对称 FC 无 bias | 2026-07-06 |
| GEMM_002 | core/gemm | 非对称输入 zp!=0 含 bias | 2026-07-06 |
| GEMM_003 | core/gemm | 非 4 对齐维度 13->7 | 2026-07-06 |
| GEMM_004 | core/gemm | 中等规模 64->32 | 2026-07-06 |
| MAXPOOL_001 | core/maxpool | 标准 2x2 stride=2 MaxPool int8 代码生成 | 2026-07-06 |
| QLINEAR_NUM_001 | core/qlinear | QLinearConv uint8 输入数值精度 (CMSIS-NN vs ONNX) | 2026-07-06 |
| QLINEAR_NUM_002 | core/qlinear | 多通道 QLinearConv + bias + per-channel scale | 2026-07-06 |
| QLINEAR_NUM_003 | core/qlinear | 高通道 QLinearConv 5×5 per-channel (仿 MNIST Conv1) | 2026-07-06 |
| SOFTMAX_001 | core/softmax | 10 分类 Softmax int8 代码生成 | 2026-07-06 |
| NEG_001 | negative | float32 无 Q/DQ 模型正确拒绝 | 2026-07-06 |
| NET_001 | networks | 真实 SqueezeNet 1.0 int8 图像分类网络导入评估 | 2026-07-06 |
| NET_002 | networks | 真实 MobileNetV2 int8/QLinear 图像分类网络导入评估 | 2026-07-06 |
| NET_003 | networks | 真实 SSD-MobileNet int8 目标检测网络导入评估 | 2026-07-06 |
| NET_004 | networks | KWS DS-CNN-style int8 网络导入评估 | 2026-07-06 |
| NET_005 | networks | Tiny signal jump int8 时序分类网络导入评估 | 2026-07-06 |
| NET_006 | networks | EfficientNet-Lite4 int8 小型图像分类网络导入评估 | 2026-07-06 |
| TOPO_001 | topology | Conv->Relu->Pool->Flatten->FC 组合链路 | 2026-07-06 |
| TOPO_002 | topology | Conv1d->Relu->Conv1d->Relu->Flatten->FC 时序分类链路 | 2026-07-06 |
| TOPO_003 | topology | 真实 MNIST QLinear int8 分类链路 | 2026-07-06 |

## 未通过用例 (FAIL)

以下用例代表目标能力但尚未实现：

*所有用例均已通过。*
