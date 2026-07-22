# NanoC-NN 能力集

> 本文件由 `python tdd/scripts/run_tests.py --mode target` 自动生成，禁止手动编辑。
> 最后更新: 2026-07-22T08:19:32.895115
> Git commit: `09f84c4`

**能力集大小: 49 / 49 (100%)**

## 已验证能力 (PASS)

以下能力按 `ONNX op -> schema 子形态 -> backend -> case` 展示。能力必须同时存在 support matrix 行和 PASS 用例，才可视为已确认。

| schema source | ONNX op | schema 子形态 | backend | lowering | 用例 ID | 分类 | 能力描述 | 验证日期 |
|---------------|---------|--------------|---------|----------|---------|------|---------|---------|
| official | Abs | static QDQ/int8 tensor, same input/output quantization | cmsis-nn | generated_c_abs_s8 | ABS_001 | core/abs | 官方 Abs QDQ/int8 同量化数值精度 | 2026-07-22 |
| official | Add | same-shape QDQ/int8 tensors, second input may be constant | cmsis-nn | cmsis_elementwise_add_s8 | ADD_001 | core/add | 官方 Add QDQ/int8 同形状常量分支数值精度 | 2026-07-22 |
| extension | QLinearGlobalAveragePool | static rank=4 quantized global average pool | cmsis-nn | cmsis_avgpool_s8 | AVGPOOL_001 | core/avgpool | QLinearGlobalAveragePool int8 全局池化代码生成 | 2026-07-22 |
| official | GlobalAveragePool | rank=4 static NCHW, QDQ/int8 tensor, full spatial H/W average | cmsis-nn | cmsis_global_avgpool_s8 | AVGPOOL_002 | core/avgpool | 官方 GlobalAveragePool QDQ/int8 全局池化数值精度 | 2026-07-22 |
| official | AveragePool | rank=4 static NCHW, QDQ/int8 tensor, CMSIS-compatible average pool window | cmsis-nn | cmsis_avgpool_s8 | AVGPOOL_003 | core/avgpool | 官方 AveragePool QDQ/int8 2x2 stride=2 数值精度 | 2026-07-22 |
| official | Concat | rank=4 NCHW, axis=1 channel concat, same-quantization QDQ/int8 tensors | cmsis-nn | cmsis_concatenation_s8_z | CONCAT_001 | core/concat | rank=4 channel 维 int8 Concat 数值精度 | 2026-07-22 |
| official | Concat | rank=4 NCHW, axis=1 channel concat, same-quantization QDQ/int8 tensors | cmsis-nn | cmsis_concatenation_s8_z | CONCAT_002 | core/concat | SqueezeNet 风格不同量化分支 Concat 数值精度 | 2026-07-22 |
| official | Conv | rank=4 NCHW, group=1, static shape, QDQ/int8 parameters | cmsis-nn | cmsis_conv2d_s8 | CONV_001 | core/conv | 1x1 pointwise 单通道 | 2026-07-22 |
| official | Conv | rank=4 NCHW, group=1, static shape, QDQ/int8 parameters | cmsis-nn | cmsis_conv2d_s8 | CONV_002 | core/conv | 3x3 标准 SAME padding | 2026-07-22 |
| official | Conv | rank=4 NCHW, group=1, static shape, QDQ/int8 parameters | cmsis-nn | cmsis_conv2d_s8 | CONV_003 | core/conv | stride=2 下采样 | 2026-07-22 |
| official | Conv | rank=4 NCHW, group=1, static shape, QDQ/int8 parameters | cmsis-nn | cmsis_conv2d_s8 | CONV_004 | core/conv | 非对称输入 zp!=0 | 2026-07-22 |
| official | Div | same-shape QDQ/int8 tensors, non-zero second input may be constant | cmsis-nn | generated_c_div_s8 | DIV_001 | core/div | 官方 Div QDQ/int8 同形状非零常量分支数值精度 | 2026-07-22 |
| official | Flatten | static QDQ/int8 tensor, axis=1, preserves ONNX row-major flatten order | cmsis-nn | generation_time_shape_alias_or_layout_copy | FLATTEN_001 | core/flatten | 官方 Flatten QDQ/int8 axis=1 数值顺序 | 2026-07-22 |
| official | Gather | static QDQ/int8 tensor, constant indices, one axis | cmsis-nn | generated_c_gather_s8 | GATHER_001 | core/gather | 官方 Gather QDQ/int8 静态 indices 数值顺序 | 2026-07-22 |
| official | Gemm | static 2-D fully connected form, optional bias, QDQ/int8 parameters | cmsis-nn | cmsis_fully_connected_s8 | GEMM_001 | core/gemm | 最小对称 FC 无 bias | 2026-07-22 |
| official | Gemm | static 2-D fully connected form, optional bias, QDQ/int8 parameters | cmsis-nn | cmsis_fully_connected_s8 | GEMM_002 | core/gemm | 非对称输入 zp!=0 含 bias | 2026-07-22 |
| official | Gemm | static 2-D fully connected form, optional bias, QDQ/int8 parameters | cmsis-nn | cmsis_fully_connected_s8 | GEMM_003 | core/gemm | 非 4 对齐维度 13->7 | 2026-07-22 |
| official | Gemm | static 2-D fully connected form, optional bias, QDQ/int8 parameters | cmsis-nn | cmsis_fully_connected_s8 | GEMM_004 | core/gemm | 中等规模 64->32 | 2026-07-22 |
| official | MatMul | static fully connected compatible form, QDQ/int8 parameters | cmsis-nn | cmsis_fully_connected_s8 | MATMUL_001 | core/matmul | 官方 MatMul QDQ/int8 FC-compatible 形态 | 2026-07-22 |
| official | MaxPool | rank=4 static NCHW, QDQ/int8 tensor, CMSIS-compatible pool window | cmsis-nn | cmsis_maxpool_s8 | MAXPOOL_001 | core/maxpool | 标准 2x2 stride=2 MaxPool int8 代码生成 | 2026-07-22 |
| official | MaxPool | rank=4 static NCHW, QDQ/int8 tensor, CMSIS-compatible pool window | cmsis-nn | cmsis_maxpool_s8 | MAXPOOL_002 | core/maxpool | SqueezeNet 风格 DQ/MaxPool/Q int8 边界代码生成 | 2026-07-22 |
| official | Mul | same-shape QDQ/int8 tensors, second input may be constant | cmsis-nn | cmsis_elementwise_mul_s8 | MUL_001 | core/mul | 官方 Mul QDQ/int8 同形状常量分支数值精度 | 2026-07-22 |
| official | Pad | static rank=4 QDQ/int8 tensor, constant mode, constant pads initializer | cmsis-nn | generated_c_pad_s8 | PAD_001 | core/pad | 官方 Pad QDQ/int8 rank4 constant mode 数值顺序 | 2026-07-22 |
| official | QLinearConv | uint8 activation, static rank=4, CMSIS-compatible convolution | cmsis-nn | cmsis_qlinearconv_s8 | QLINEAR_NUM_001 | core/qlinear | QLinearConv uint8 输入数值精度 (CMSIS-NN vs ONNX) | 2026-07-22 |
| official | QLinearConv | static rank=4 convolution with per-channel weight scale | cmsis-nn | cmsis_qlinearconv_per_channel_s8 | QLINEAR_NUM_002 | core/qlinear | 多通道 QLinearConv + bias + per-channel scale | 2026-07-22 |
| official | QLinearConv | static rank=4 convolution with per-channel weight scale | cmsis-nn | cmsis_qlinearconv_per_channel_s8 | QLINEAR_NUM_003 | core/qlinear | 高通道 QLinearConv 5×5 per-channel (仿 MNIST Conv1) | 2026-07-22 |
| official | Reshape | static QDQ/int8 tensor, constant shape, same element count | cmsis-nn | generation_time_shape_alias_or_layout_copy | RESHAPE_001 | core/reshape | 官方 Reshape QDQ/int8 静态形状数值顺序 | 2026-07-22 |
| official | Sigmoid | static QDQ/int8 tensor | cmsis-nn | generated_c_sigmoid_s8 | SIGMOID_001 | core/sigmoid | 官方 Sigmoid QDQ/int8 静态张量数值精度 | 2026-07-22 |
| official | Slice | static QDQ/int8 tensor, constant starts/ends/axes/steps | cmsis-nn | generated_c_slice_s8 | SLICE_001 | core/slice | 官方 Slice QDQ/int8 静态区间数值顺序 | 2026-07-22 |
| official | Softmax | static classification vector, int8 output path | cmsis-nn | cmsis_softmax_s8 | SOFTMAX_001 | core/softmax | 10 分类 Softmax int8 代码生成 | 2026-07-22 |
| official | Softmax | static classification vector, int8 output path | cmsis-nn | cmsis_softmax_s8 | SOFTMAX_002 | core/softmax | SqueezeNet 末端 float output Softmax int8 C 输出 | 2026-07-22 |
| official | Squeeze | static QDQ/int8 tensor, explicit axes removing dimensions of size 1 | cmsis-nn | generation_time_shape_alias_or_layout_copy | SQUEEZE_001 | core/squeeze | 官方 Squeeze QDQ/int8 静态去 1 维数值顺序 | 2026-07-22 |
| official | Sub | same-shape QDQ/int8 tensors, second input may be constant | cmsis-nn | generated_c_sub_s8 | SUB_001 | core/sub | 官方 Sub QDQ/int8 同形状常量分支数值精度 | 2026-07-22 |
| official | Transpose | static QDQ/int8 tensor, rank=4 explicit perm | cmsis-nn | cmsis_transpose_s8 | TRANSPOSE_001 | core/transpose | 官方 Transpose QDQ/int8 rank4 显式 perm 数值顺序 | 2026-07-22 |
| mixed | multiple | internal float32 graph without QDQ quantization section | reject | reject_without_quantization | NEG_001 | negative | float32 无 Q/DQ 模型正确拒绝 | 2026-07-22 |
| mixed | multiple | SqueezeNet int8 with QLinearConv, MaxPool, DQ/Concat/Q, QLinearGlobalAveragePool and terminal Softmax | cmsis-nn | cmsis_nn_squeezenet_int8 | NET_001 | networks | 真实 SqueezeNet 1.0 int8 图像分类网络导入评估 | 2026-07-22 |
| official | Conv | rank=4 NCHW, group=1, static shape, QDQ/int8 parameters | cmsis-nn | cmsis_conv2d_s8 | NET_002 | networks | 真实 MobileNetV2 int8/QLinear 图像分类网络导入评估 | 2026-07-22 |
| official | Conv | rank=4, group=input_channels, depthwise multiplier, QDQ/int8 parameters | cmsis-nn | cmsis_depthwise_conv2d_s8 | NET_002 | networks | 真实 MobileNetV2 int8/QLinear 图像分类网络导入评估 | 2026-07-22 |
| extension | QLinearAdd | two static int8 tensors or activation plus bias-like tensor | cmsis-nn | cmsis_elementwise_add_s8 | NET_002 | networks | 真实 MobileNetV2 int8/QLinear 图像分类网络导入评估 | 2026-07-22 |
| extension | QLinearGlobalAveragePool | static rank=4 quantized global average pool | cmsis-nn | cmsis_avgpool_s8 | NET_002 | networks | 真实 MobileNetV2 int8/QLinear 图像分类网络导入评估 | 2026-07-22 |
| official | QLinearMatMul | static fully connected compatible quantized matmul | cmsis-nn | cmsis_fully_connected_per_channel_s8 | NET_002 | networks | 真实 MobileNetV2 int8/QLinear 图像分类网络导入评估 | 2026-07-22 |
| mixed | multiple | SSD-MobileNet detection graph with multi-output heads/dynamic or postprocess ops | reject | unsupported_detection_boundary | NET_003 | networks | 真实 SSD-MobileNet int8 目标检测网络导入评估 | 2026-07-22 |
| official | Conv | rank=4 NCHW, group=1, static shape, QDQ/int8 parameters | cmsis-nn | cmsis_conv2d_s8 | NET_004 | networks | KWS DS-CNN-style int8 网络导入评估 | 2026-07-22 |
| official | Conv | rank=4, group=input_channels, depthwise multiplier, QDQ/int8 parameters | cmsis-nn | cmsis_depthwise_conv2d_s8 | NET_004 | networks | KWS DS-CNN-style int8 网络导入评估 | 2026-07-22 |
| official | MaxPool | rank=4 static NCHW, QDQ/int8 tensor, CMSIS-compatible pool window | cmsis-nn | cmsis_maxpool_s8 | NET_004 | networks | KWS DS-CNN-style int8 网络导入评估 | 2026-07-22 |
| official | Gemm | static 2-D fully connected form, optional bias, QDQ/int8 parameters | cmsis-nn | cmsis_fully_connected_s8 | NET_004 | networks | KWS DS-CNN-style int8 网络导入评估 | 2026-07-22 |
| official | Conv | rank=3 NCW, group=1, static shape, QDQ/int8 parameters | cmsis-nn | cmsis_conv1d_as_2d_s8 | NET_005 | networks | Tiny signal jump int8 时序分类网络导入评估 | 2026-07-22 |
| official | Gemm | static 2-D fully connected form, optional bias, QDQ/int8 parameters | cmsis-nn | cmsis_fully_connected_s8 | NET_005 | networks | Tiny signal jump int8 时序分类网络导入评估 | 2026-07-22 |
| mixed | multiple | EfficientNet-Lite quantized graph forms outside current layout/op whitelist | reject | unsupported_import_boundary | NET_006 | networks | EfficientNet-Lite4 int8 小型图像分类网络导入评估 | 2026-07-22 |
| mixed | multiple | static-shape float or QDQ graph executed by generated C reference backend | c-reference | c99_reference_runtime | TS_001 | networks | CWRU bearing vibration MLP float C reference 数值回归 | 2026-07-22 |
| mixed | multiple | static-shape float or QDQ graph executed by generated C reference backend | c-reference | c99_reference_runtime | TS_002 | networks | KWS DS-CNN PTQ INT8 Q/DQ float C reference 数值回归 | 2026-07-22 |
| mixed | multiple | static-shape float or QDQ graph executed by generated C reference backend | c-reference | c99_reference_runtime | TS_003 | networks | KWS DS-CNN QAT INT8 Q/DQ float C reference 数值回归 | 2026-07-22 |
| mixed | multiple | static-shape float or QDQ graph executed by generated C reference backend | c-reference | c99_reference_runtime | TS_004 | networks | MET hybrid 双输入 float C reference 数值回归 | 2026-07-22 |
| mixed | multiple | static-shape float or QDQ graph executed by generated C reference backend | c-reference | c99_reference_runtime | TS_005 | networks | STWIN vowel IMU CNN float C reference 数值回归 | 2026-07-22 |
| official | Conv | rank=4 NCHW, group=1, static shape, QDQ/int8 parameters | cmsis-nn | cmsis_conv2d_s8 | TOPO_001 | topology | Conv->Relu->Pool->Flatten->FC 组合链路 | 2026-07-22 |
| official | MaxPool | rank=4 static NCHW, QDQ/int8 tensor, CMSIS-compatible pool window | cmsis-nn | cmsis_maxpool_s8 | TOPO_001 | topology | Conv->Relu->Pool->Flatten->FC 组合链路 | 2026-07-22 |
| official | Gemm | static 2-D fully connected form, optional bias, QDQ/int8 parameters | cmsis-nn | cmsis_fully_connected_s8 | TOPO_001 | topology | Conv->Relu->Pool->Flatten->FC 组合链路 | 2026-07-22 |
| official | Conv | rank=3 NCW, group=1, static shape, QDQ/int8 parameters | cmsis-nn | cmsis_conv1d_as_2d_s8 | TOPO_002 | topology | Conv1d->Relu->Conv1d->Relu->Flatten->FC 时序分类链路 | 2026-07-22 |
| official | Gemm | static 2-D fully connected form, optional bias, QDQ/int8 parameters | cmsis-nn | cmsis_fully_connected_s8 | TOPO_002 | topology | Conv1d->Relu->Conv1d->Relu->Flatten->FC 时序分类链路 | 2026-07-22 |
| official | QLinearConv | static rank=4 convolution with per-channel weight scale | cmsis-nn | cmsis_qlinearconv_per_channel_s8 | TOPO_003 | topology | 真实 MNIST QLinear int8 分类链路 | 2026-07-22 |
| official | MaxPool | rank=4 static NCHW, QDQ/int8 tensor, CMSIS-compatible pool window | cmsis-nn | cmsis_maxpool_s8 | TOPO_003 | topology | 真实 MNIST QLinear int8 分类链路 | 2026-07-22 |
| extension | QLinearAdd | two static int8 tensors or activation plus bias-like tensor | cmsis-nn | cmsis_elementwise_add_s8 | TOPO_003 | topology | 真实 MNIST QLinear int8 分类链路 | 2026-07-22 |
| official | QLinearMatMul | static fully connected compatible quantized matmul | cmsis-nn | cmsis_fully_connected_per_channel_s8 | TOPO_003 | topology | 真实 MNIST QLinear int8 分类链路 | 2026-07-22 |

## 未通过用例 (FAIL)

以下用例代表目标能力或拒绝边界，但本轮执行未达到 registry 预期：

*所有用例均已通过。*

## 已规划但未验证

以下 support matrix 行尚未绑定 PASS 用例，不能作为当前能力声明：

*暂无未覆盖 planned 行。*
