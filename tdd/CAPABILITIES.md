# NanoC-NN 能力集

> 本文件由 `python tdd/scripts/run_tests.py --mode target` 自动生成，禁止手动编辑。
> 最后更新: 2026-07-11T04:01:39.444761
> Git commit: `537eb10`

**能力集大小: 28 / 28 (100%)**

## 已验证能力 (PASS)

以下能力按 `ONNX op -> schema 子形态 -> backend -> case` 展示。能力必须同时存在 support matrix 行和 PASS 用例，才可视为已确认。

| schema source | ONNX op | schema 子形态 | backend | lowering | 用例 ID | 分类 | 能力描述 | 验证日期 |
|---------------|---------|--------------|---------|----------|---------|------|---------|---------|
| official | Conv | rank=4 NCHW, group=1, static shape, QDQ/int8 parameters | cmsis-nn | cmsis_conv2d_s8 | CONV_001 | core/conv | 1x1 pointwise 单通道 | 2026-07-11 |
| official | Conv | rank=4 NCHW, group=1, static shape, QDQ/int8 parameters | cmsis-nn | cmsis_conv2d_s8 | CONV_002 | core/conv | 3x3 标准 SAME padding | 2026-07-11 |
| official | Conv | rank=4 NCHW, group=1, static shape, QDQ/int8 parameters | cmsis-nn | cmsis_conv2d_s8 | CONV_003 | core/conv | stride=2 下采样 | 2026-07-11 |
| official | Conv | rank=4 NCHW, group=1, static shape, QDQ/int8 parameters | cmsis-nn | cmsis_conv2d_s8 | CONV_004 | core/conv | 非对称输入 zp!=0 | 2026-07-11 |
| official | Gemm | static 2-D fully connected form, optional bias, QDQ/int8 parameters | cmsis-nn | cmsis_fully_connected_s8 | GEMM_001 | core/gemm | 最小对称 FC 无 bias | 2026-07-11 |
| official | Gemm | static 2-D fully connected form, optional bias, QDQ/int8 parameters | cmsis-nn | cmsis_fully_connected_s8 | GEMM_002 | core/gemm | 非对称输入 zp!=0 含 bias | 2026-07-11 |
| official | Gemm | static 2-D fully connected form, optional bias, QDQ/int8 parameters | cmsis-nn | cmsis_fully_connected_s8 | GEMM_003 | core/gemm | 非 4 对齐维度 13->7 | 2026-07-11 |
| official | Gemm | static 2-D fully connected form, optional bias, QDQ/int8 parameters | cmsis-nn | cmsis_fully_connected_s8 | GEMM_004 | core/gemm | 中等规模 64->32 | 2026-07-11 |
| official | MaxPool | rank=4 static NCHW, QDQ/int8 tensor, CMSIS-compatible pool window | cmsis-nn | cmsis_maxpool_s8 | MAXPOOL_001 | core/maxpool | 标准 2x2 stride=2 MaxPool int8 代码生成 | 2026-07-11 |
| official | QLinearConv | uint8 activation, static rank=4, CMSIS-compatible convolution | cmsis-nn | cmsis_qlinearconv_s8 | QLINEAR_NUM_001 | core/qlinear | QLinearConv uint8 输入数值精度 (CMSIS-NN vs ONNX) | 2026-07-11 |
| official | QLinearConv | static rank=4 convolution with per-channel weight scale | cmsis-nn | cmsis_qlinearconv_per_channel_s8 | QLINEAR_NUM_002 | core/qlinear | 多通道 QLinearConv + bias + per-channel scale | 2026-07-11 |
| official | QLinearConv | static rank=4 convolution with per-channel weight scale | cmsis-nn | cmsis_qlinearconv_per_channel_s8 | QLINEAR_NUM_003 | core/qlinear | 高通道 QLinearConv 5×5 per-channel (仿 MNIST Conv1) | 2026-07-11 |
| official | Softmax | static classification vector, int8 output path | cmsis-nn | cmsis_softmax_s8 | SOFTMAX_001 | core/softmax | 10 分类 Softmax int8 代码生成 | 2026-07-11 |
| mixed | multiple | internal float32 graph without QDQ quantization section | reject | reject_without_quantization | NEG_001 | negative | float32 无 Q/DQ 模型正确拒绝 | 2026-07-11 |
| mixed | multiple | SqueezeNet int8 with QDQ/Concat/GlobalAveragePool/Softmax boundary gaps | reject | blocked_import_boundary | NET_001 | networks | 真实 SqueezeNet 1.0 int8 图像分类网络导入评估 | 2026-07-11 |
| official | Conv | rank=4 NCHW, group=1, static shape, QDQ/int8 parameters | cmsis-nn | cmsis_conv2d_s8 | NET_002 | networks | 真实 MobileNetV2 int8/QLinear 图像分类网络导入评估 | 2026-07-11 |
| official | Conv | rank=4, group=input_channels, depthwise multiplier, QDQ/int8 parameters | cmsis-nn | cmsis_depthwise_conv2d_s8 | NET_002 | networks | 真实 MobileNetV2 int8/QLinear 图像分类网络导入评估 | 2026-07-11 |
| extension | QLinearAdd | two static int8 tensors or activation plus bias-like tensor | cmsis-nn | cmsis_elementwise_add_s8 | NET_002 | networks | 真实 MobileNetV2 int8/QLinear 图像分类网络导入评估 | 2026-07-11 |
| extension | QLinearGlobalAveragePool | static rank=4 quantized global average pool | cmsis-nn | cmsis_avgpool_s8 | NET_002 | networks | 真实 MobileNetV2 int8/QLinear 图像分类网络导入评估 | 2026-07-11 |
| official | QLinearMatMul | static fully connected compatible quantized matmul | cmsis-nn | cmsis_fully_connected_per_channel_s8 | NET_002 | networks | 真实 MobileNetV2 int8/QLinear 图像分类网络导入评估 | 2026-07-11 |
| mixed | multiple | SSD-MobileNet detection graph with multi-output heads/dynamic or postprocess ops | reject | unsupported_detection_boundary | NET_003 | networks | 真实 SSD-MobileNet int8 目标检测网络导入评估 | 2026-07-11 |
| official | Conv | rank=4 NCHW, group=1, static shape, QDQ/int8 parameters | cmsis-nn | cmsis_conv2d_s8 | NET_004 | networks | KWS DS-CNN-style int8 网络导入评估 | 2026-07-11 |
| official | Conv | rank=4, group=input_channels, depthwise multiplier, QDQ/int8 parameters | cmsis-nn | cmsis_depthwise_conv2d_s8 | NET_004 | networks | KWS DS-CNN-style int8 网络导入评估 | 2026-07-11 |
| official | MaxPool | rank=4 static NCHW, QDQ/int8 tensor, CMSIS-compatible pool window | cmsis-nn | cmsis_maxpool_s8 | NET_004 | networks | KWS DS-CNN-style int8 网络导入评估 | 2026-07-11 |
| official | Gemm | static 2-D fully connected form, optional bias, QDQ/int8 parameters | cmsis-nn | cmsis_fully_connected_s8 | NET_004 | networks | KWS DS-CNN-style int8 网络导入评估 | 2026-07-11 |
| official | Conv | rank=3 NCW, group=1, static shape, QDQ/int8 parameters | cmsis-nn | cmsis_conv1d_as_2d_s8 | NET_005 | networks | Tiny signal jump int8 时序分类网络导入评估 | 2026-07-11 |
| official | Gemm | static 2-D fully connected form, optional bias, QDQ/int8 parameters | cmsis-nn | cmsis_fully_connected_s8 | NET_005 | networks | Tiny signal jump int8 时序分类网络导入评估 | 2026-07-11 |
| mixed | multiple | EfficientNet-Lite quantized graph forms outside current layout/op whitelist | reject | unsupported_import_boundary | NET_006 | networks | EfficientNet-Lite4 int8 小型图像分类网络导入评估 | 2026-07-11 |
| mixed | multiple | static-shape float or QDQ graph executed by generated C reference backend | c-reference | c99_reference_runtime | TS_001 | networks | CWRU bearing vibration MLP float C reference 数值回归 | 2026-07-11 |
| mixed | multiple | static-shape float or QDQ graph executed by generated C reference backend | c-reference | c99_reference_runtime | TS_002 | networks | KWS DS-CNN PTQ INT8 Q/DQ float C reference 数值回归 | 2026-07-11 |
| mixed | multiple | static-shape float or QDQ graph executed by generated C reference backend | c-reference | c99_reference_runtime | TS_003 | networks | KWS DS-CNN QAT INT8 Q/DQ float C reference 数值回归 | 2026-07-11 |
| mixed | multiple | static-shape float or QDQ graph executed by generated C reference backend | c-reference | c99_reference_runtime | TS_004 | networks | MET hybrid 双输入 float C reference 数值回归 | 2026-07-11 |
| mixed | multiple | static-shape float or QDQ graph executed by generated C reference backend | c-reference | c99_reference_runtime | TS_005 | networks | STWIN vowel IMU CNN float C reference 数值回归 | 2026-07-11 |
| official | Conv | rank=4 NCHW, group=1, static shape, QDQ/int8 parameters | cmsis-nn | cmsis_conv2d_s8 | TOPO_001 | topology | Conv->Relu->Pool->Flatten->FC 组合链路 | 2026-07-11 |
| official | MaxPool | rank=4 static NCHW, QDQ/int8 tensor, CMSIS-compatible pool window | cmsis-nn | cmsis_maxpool_s8 | TOPO_001 | topology | Conv->Relu->Pool->Flatten->FC 组合链路 | 2026-07-11 |
| official | Gemm | static 2-D fully connected form, optional bias, QDQ/int8 parameters | cmsis-nn | cmsis_fully_connected_s8 | TOPO_001 | topology | Conv->Relu->Pool->Flatten->FC 组合链路 | 2026-07-11 |
| official | Conv | rank=3 NCW, group=1, static shape, QDQ/int8 parameters | cmsis-nn | cmsis_conv1d_as_2d_s8 | TOPO_002 | topology | Conv1d->Relu->Conv1d->Relu->Flatten->FC 时序分类链路 | 2026-07-11 |
| official | Gemm | static 2-D fully connected form, optional bias, QDQ/int8 parameters | cmsis-nn | cmsis_fully_connected_s8 | TOPO_002 | topology | Conv1d->Relu->Conv1d->Relu->Flatten->FC 时序分类链路 | 2026-07-11 |
| official | QLinearConv | static rank=4 convolution with per-channel weight scale | cmsis-nn | cmsis_qlinearconv_per_channel_s8 | TOPO_003 | topology | 真实 MNIST QLinear int8 分类链路 | 2026-07-11 |
| official | MaxPool | rank=4 static NCHW, QDQ/int8 tensor, CMSIS-compatible pool window | cmsis-nn | cmsis_maxpool_s8 | TOPO_003 | topology | 真实 MNIST QLinear int8 分类链路 | 2026-07-11 |
| extension | QLinearAdd | two static int8 tensors or activation plus bias-like tensor | cmsis-nn | cmsis_elementwise_add_s8 | TOPO_003 | topology | 真实 MNIST QLinear int8 分类链路 | 2026-07-11 |
| official | QLinearMatMul | static fully connected compatible quantized matmul | cmsis-nn | cmsis_fully_connected_per_channel_s8 | TOPO_003 | topology | 真实 MNIST QLinear int8 分类链路 | 2026-07-11 |

## 未通过用例 (FAIL)

以下用例代表目标能力或拒绝边界，但本轮执行未达到 registry 预期：

*所有用例均已通过。*

## 已规划但未验证

以下 support matrix 行尚未绑定 PASS 用例，不能作为当前能力声明：

| support ID | schema source | ONNX op | schema 子形态 | 目标 lowering | 当前状态 |
|------------|---------------|---------|--------------|---------------|----------|
| ONNX_MATMUL_QDQ_INT8 | official | MatMul | static fully connected compatible form, QDQ/int8 parameters | planned_cmsis_fully_connected_s8 | blocked |
| ONNX_CONCAT_QDQ_INT8 | official | Concat | static same-quantization int8 tensors, explicit axis | planned_cmsis_concatenation_s8 | blocked |
