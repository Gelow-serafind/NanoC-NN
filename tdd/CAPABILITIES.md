# NanoC-NN 能力集

> 本文件由 `python tdd/scripts/run_tests.py --mode target` 自动生成，禁止手动编辑。
> 最后更新: 2026-08-11T23:56:40.790923
> Git commit: `92c93df`

**能力集大小: 92 / 92 (100%)**

## 已验证能力 (PASS)

以下能力按 `ONNX op -> schema 子形态 -> backend -> case` 展示。能力必须同时存在 support matrix 行和 PASS 用例，才可视为已确认。

| schema source | ONNX op | schema 子形态 | backend | lowering | 用例 ID | 分类 | 能力描述 | 验证日期 |
|---------------|---------|--------------|---------|----------|---------|------|---------|---------|
| official | Abs | static QDQ/int8 tensor, same input/output quantization | cmsis-nn | generated_c_abs_s8 | ABS_001 | core/abs | 官方 Abs QDQ/int8 同量化数值精度 | 2026-08-11 |
| official | Add | same-shape QDQ/int8 tensors, second input may be constant | cmsis-nn | cmsis_elementwise_add_s8 | ADD_001 | core/add | 官方 Add QDQ/int8 同形状常量分支数值精度 | 2026-08-11 |
| official | And | two bool intermediate tensors (compare outputs), bool output consumed by Where | cmsis-nn | generated_c_and_bool | AND_001 | core/and | 官方 And bool 逻辑中间张量驱动 Where 数值精度 | 2026-08-11 |
| official | Where | static bool condition, same-shape QDQ/int8 then/else tensors | cmsis-nn | generated_c_where_s8 | AND_001 | core/and | 官方 And bool 逻辑中间张量驱动 Where 数值精度 | 2026-08-11 |
| extension | QLinearGlobalAveragePool | static rank=4 quantized global average pool | cmsis-nn | cmsis_avgpool_s8 | AVGPOOL_001 | core/avgpool | QLinearGlobalAveragePool int8 全局池化代码生成 | 2026-08-11 |
| official | GlobalAveragePool | rank=4 static NCHW, QDQ/int8 tensor, full spatial H/W average | cmsis-nn | cmsis_global_avgpool_s8 | AVGPOOL_002 | core/avgpool | 官方 GlobalAveragePool QDQ/int8 全局池化数值精度 | 2026-08-11 |
| official | AveragePool | rank=4 static NCHW, QDQ/int8 tensor, CMSIS-compatible average pool window | cmsis-nn | cmsis_avgpool_s8 | AVGPOOL_003 | core/avgpool | 官方 AveragePool QDQ/int8 2x2 stride=2 数值精度 | 2026-08-11 |
| official | Ceil | static QDQ/int8 tensor | cmsis-nn | generated_c_ceil_s8 | CEIL_001 | core/ceil | 官方 Ceil QDQ/int8 静态张量数值精度 | 2026-08-11 |
| official | Celu | static QDQ/int8 tensor, alpha attribute | cmsis-nn | generated_c_celu_s8 | CELU_001 | core/celu | 官方 Celu QDQ/int8 alpha 属性数值精度 | 2026-08-11 |
| official | Clip | static QDQ/int8 tensor with constant min/max inputs or legacy attributes | cmsis-nn | generated_c_clip_s8 | CLIP_001 | core/clip | 官方 Clip QDQ/int8 常量 min/max 数值精度 | 2026-08-11 |
| official | Concat | rank=4 NCHW, axis=1 channel concat, same-quantization QDQ/int8 tensors | cmsis-nn | cmsis_concatenation_s8_z | CONCAT_001 | core/concat | rank=4 channel 维 int8 Concat 数值精度 | 2026-08-11 |
| official | Concat | rank=4 NCHW, axis=1 channel concat, same-quantization QDQ/int8 tensors | cmsis-nn | cmsis_concatenation_s8_z | CONCAT_002 | core/concat | SqueezeNet 风格不同量化分支 Concat 数值精度 | 2026-08-11 |
| official | Conv | rank=4 NCHW, group=1, static shape, QDQ/int8 parameters | cmsis-nn | cmsis_conv2d_s8 | CONV_001 | core/conv | 1x1 pointwise 单通道 | 2026-08-11 |
| official | Conv | rank=4 NCHW, group=1, static shape, QDQ/int8 parameters | cmsis-nn | cmsis_conv2d_s8 | CONV_002 | core/conv | 3x3 标准 SAME padding | 2026-08-11 |
| official | Conv | rank=4 NCHW, group=1, static shape, QDQ/int8 parameters | cmsis-nn | cmsis_conv2d_s8 | CONV_003 | core/conv | stride=2 下采样 | 2026-08-11 |
| official | Conv | rank=4 NCHW, group=1, static shape, QDQ/int8 parameters | cmsis-nn | cmsis_conv2d_s8 | CONV_004 | core/conv | 非对称输入 zp!=0 | 2026-08-11 |
| official | Div | same-shape QDQ/int8 tensors, non-zero second input may be constant | cmsis-nn | generated_c_div_s8 | DIV_001 | core/div | 官方 Div QDQ/int8 同形状非零常量分支数值精度 | 2026-08-11 |
| official | Elu | static QDQ/int8 tensor, alpha attribute | cmsis-nn | generated_c_elu_s8 | ELU_001 | core/elu | 官方 Elu QDQ/int8 alpha 属性数值精度 | 2026-08-11 |
| official | Equal | same-shape QDQ/int8 tensors, bool output consumed by Where | cmsis-nn | generated_c_equal_bool | EQUAL_001 | core/equal | 官方 Equal QDQ/int8 bool 结果驱动 Where 数值精度 | 2026-08-11 |
| official | Where | static bool condition, same-shape QDQ/int8 then/else tensors | cmsis-nn | generated_c_where_s8 | EQUAL_001 | core/equal | 官方 Equal QDQ/int8 bool 结果驱动 Where 数值精度 | 2026-08-11 |
| official | Erf | static QDQ/int8 tensor | cmsis-nn | generated_c_erf_s8 | ERF_001 | core/erf | 官方 Erf QDQ/int8 静态张量数值精度 | 2026-08-11 |
| official | Exp | static QDQ/int8 tensor, bounded input domain | cmsis-nn | generated_c_exp_s8 | EXP_001 | core/exp | 官方 Exp QDQ/int8 静态张量受控输入域数值精度 | 2026-08-11 |
| official | Flatten | static QDQ/int8 tensor, axis=1, preserves ONNX row-major flatten order | cmsis-nn | generation_time_shape_alias_or_layout_copy | FLATTEN_001 | core/flatten | 官方 Flatten QDQ/int8 axis=1 数值顺序 | 2026-08-11 |
| official | Floor | static QDQ/int8 tensor | cmsis-nn | generated_c_floor_s8 | FLOOR_001 | core/floor | 官方 Floor QDQ/int8 静态张量数值精度 | 2026-08-11 |
| official | Gather | static QDQ/int8 tensor, constant indices, one axis | cmsis-nn | generated_c_gather_s8 | GATHER_001 | core/gather | 官方 Gather QDQ/int8 静态 indices 数值顺序 | 2026-08-11 |
| official | Gemm | static 2-D fully connected form, optional bias, QDQ/int8 parameters | cmsis-nn | cmsis_fully_connected_s8 | GEMM_001 | core/gemm | 最小对称 FC 无 bias | 2026-08-11 |
| official | Gemm | static 2-D fully connected form, optional bias, QDQ/int8 parameters | cmsis-nn | cmsis_fully_connected_s8 | GEMM_002 | core/gemm | 非对称输入 zp!=0 含 bias | 2026-08-11 |
| official | Gemm | static 2-D fully connected form, optional bias, QDQ/int8 parameters | cmsis-nn | cmsis_fully_connected_s8 | GEMM_003 | core/gemm | 非 4 对齐维度 13->7 | 2026-08-11 |
| official | Gemm | static 2-D fully connected form, optional bias, QDQ/int8 parameters | cmsis-nn | cmsis_fully_connected_s8 | GEMM_004 | core/gemm | 中等规模 64->32 | 2026-08-11 |
| official | Greater | same-shape QDQ/int8 tensors, bool output consumed by Where | cmsis-nn | generated_c_greater_bool | GREATER_001 | core/greater | 官方 Greater QDQ/int8 bool 结果驱动 Where 数值精度 | 2026-08-11 |
| official | Where | static bool condition, same-shape QDQ/int8 then/else tensors | cmsis-nn | generated_c_where_s8 | GREATER_001 | core/greater | 官方 Greater QDQ/int8 bool 结果驱动 Where 数值精度 | 2026-08-11 |
| official | GreaterOrEqual | same-shape QDQ/int8 tensors, bool output consumed by Where | cmsis-nn | generated_c_greaterorequal_bool | GREATEROREQUAL_001 | core/greaterorequal | 官方 GreaterOrEqual QDQ/int8 bool 结果驱动 Where 数值精度 | 2026-08-11 |
| official | Where | static bool condition, same-shape QDQ/int8 then/else tensors | cmsis-nn | generated_c_where_s8 | GREATEROREQUAL_001 | core/greaterorequal | 官方 GreaterOrEqual QDQ/int8 bool 结果驱动 Where 数值精度 | 2026-08-11 |
| official | HardSigmoid | static QDQ/int8 tensor, alpha/beta attributes | cmsis-nn | generated_c_hardsigmoid_s8 | HARDSIGMOID_001 | core/hardsigmoid | 官方 HardSigmoid QDQ/int8 alpha/beta 属性数值精度 | 2026-08-11 |
| official | HardSwish | static QDQ/int8 tensor | cmsis-nn | generated_c_hardswish_s8 | HARDSWISH_001 | core/hardswish | 官方 HardSwish QDQ/int8 静态张量数值精度 | 2026-08-11 |
| official | LeakyRelu | static QDQ/int8 tensor, scalar alpha attribute | cmsis-nn | generated_c_leakyrelu_s8 | LEAKYRELU_001 | core/leakyrelu | 官方 LeakyRelu QDQ/int8 alpha=0.1 数值精度 | 2026-08-11 |
| official | Less | same-shape QDQ/int8 tensors, bool output consumed by Where | cmsis-nn | generated_c_less_bool | LESS_001 | core/less | 官方 Less QDQ/int8 bool 结果驱动 Where 数值精度 | 2026-08-11 |
| official | Where | static bool condition, same-shape QDQ/int8 then/else tensors | cmsis-nn | generated_c_where_s8 | LESS_001 | core/less | 官方 Less QDQ/int8 bool 结果驱动 Where 数值精度 | 2026-08-11 |
| official | LessOrEqual | same-shape QDQ/int8 tensors, bool output consumed by Where | cmsis-nn | generated_c_lessorequal_bool | LESSOREQUAL_001 | core/lessorequal | 官方 LessOrEqual QDQ/int8 bool 结果驱动 Where 数值精度 | 2026-08-11 |
| official | Where | static bool condition, same-shape QDQ/int8 then/else tensors | cmsis-nn | generated_c_where_s8 | LESSOREQUAL_001 | core/lessorequal | 官方 LessOrEqual QDQ/int8 bool 结果驱动 Where 数值精度 | 2026-08-11 |
| official | Log | static QDQ/int8 tensor, positive input domain | cmsis-nn | generated_c_log_s8 | LOG_001 | core/log | 官方 Log QDQ/int8 正输入域数值精度 | 2026-08-11 |
| official | MatMul | static fully connected compatible form, QDQ/int8 parameters | cmsis-nn | cmsis_fully_connected_s8 | MATMUL_001 | core/matmul | 官方 MatMul QDQ/int8 FC-compatible 形态 | 2026-08-11 |
| official | Max | same-shape QDQ/int8 tensors, second input may be constant | cmsis-nn | generated_c_max_s8 | MAX_001 | core/max | 官方 Max QDQ/int8 同形状双输入数值精度 | 2026-08-11 |
| official | MaxPool | rank=4 static NCHW, QDQ/int8 tensor, CMSIS-compatible pool window | cmsis-nn | cmsis_maxpool_s8 | MAXPOOL_001 | core/maxpool | 标准 2x2 stride=2 MaxPool int8 代码生成 | 2026-08-11 |
| official | MaxPool | rank=4 static NCHW, QDQ/int8 tensor, CMSIS-compatible pool window | cmsis-nn | cmsis_maxpool_s8 | MAXPOOL_002 | core/maxpool | SqueezeNet 风格 DQ/MaxPool/Q int8 边界代码生成 | 2026-08-11 |
| official | Min | same-shape QDQ/int8 tensors, second input may be constant | cmsis-nn | generated_c_min_s8 | MIN_001 | core/min | 官方 Min QDQ/int8 同形状双输入数值精度 | 2026-08-11 |
| official | Mul | same-shape QDQ/int8 tensors, second input may be constant | cmsis-nn | cmsis_elementwise_mul_s8 | MUL_001 | core/mul | 官方 Mul QDQ/int8 同形状常量分支数值精度 | 2026-08-11 |
| official | Neg | static QDQ/int8 tensor | cmsis-nn | generated_c_neg_s8 | NEGOP_001 | core/neg | 官方 Neg QDQ/int8 静态张量数值精度 | 2026-08-11 |
| official | Not | single bool intermediate tensor (compare output), bool output consumed by Where | cmsis-nn | generated_c_not_bool | NOT_001 | core/not | 官方 Not bool 逻辑中间张量驱动 Where 数值精度 | 2026-08-11 |
| official | Where | static bool condition, same-shape QDQ/int8 then/else tensors | cmsis-nn | generated_c_where_s8 | NOT_001 | core/not | 官方 Not bool 逻辑中间张量驱动 Where 数值精度 | 2026-08-11 |
| official | Or | two bool intermediate tensors (compare outputs), bool output consumed by Where | cmsis-nn | generated_c_or_bool | OR_001 | core/or | 官方 Or bool 逻辑中间张量驱动 Where 数值精度 | 2026-08-11 |
| official | Where | static bool condition, same-shape QDQ/int8 then/else tensors | cmsis-nn | generated_c_where_s8 | OR_001 | core/or | 官方 Or bool 逻辑中间张量驱动 Where 数值精度 | 2026-08-11 |
| official | Pad | static rank=4 QDQ/int8 tensor, constant mode, constant pads initializer | cmsis-nn | generated_c_pad_s8 | PAD_001 | core/pad | 官方 Pad QDQ/int8 rank4 constant mode 数值顺序 | 2026-08-11 |
| official | Pow | same-shape QDQ/int8 tensors, finite value domain | cmsis-nn | generated_c_pow_s8 | POW_001 | core/pow | 官方 Pow QDQ/int8 同形状有限输入域数值精度 | 2026-08-11 |
| official | PRelu | same-shape QDQ/int8 tensor, constant quantized slope input | cmsis-nn | generated_c_prelu_s8 | PRELU_001 | core/prelu | 官方 PRelu QDQ/int8 常量 slope 数值精度 | 2026-08-11 |
| official | QLinearConv | uint8 activation, static rank=4, CMSIS-compatible convolution | cmsis-nn | cmsis_qlinearconv_s8 | QLINEAR_NUM_001 | core/qlinear | QLinearConv uint8 输入数值精度 (CMSIS-NN vs ONNX) | 2026-08-11 |
| official | QLinearConv | static rank=4 convolution with per-channel weight scale | cmsis-nn | cmsis_qlinearconv_per_channel_s8 | QLINEAR_NUM_002 | core/qlinear | 多通道 QLinearConv + bias + per-channel scale | 2026-08-11 |
| official | QLinearConv | static rank=4 convolution with per-channel weight scale | cmsis-nn | cmsis_qlinearconv_per_channel_s8 | QLINEAR_NUM_003 | core/qlinear | 高通道 QLinearConv 5×5 per-channel (仿 MNIST Conv1) | 2026-08-11 |
| official | Reciprocal | static QDQ/int8 tensor, non-zero finite value domain | cmsis-nn | generated_c_reciprocal_s8 | RECIPROCAL_001 | core/reciprocal | 官方 Reciprocal QDQ/int8 非零输入数值精度 | 2026-08-11 |
| official | ReduceL1 | rank=2 static QDQ/int8 tensor, axes=1, keepdims=1 | cmsis-nn | generated_c_reducel1_s8 | REDUCEL1_001 | core/reducel1 | 官方 ReduceL1 QDQ/int8 rank2 axes=1 keepdims=1 数值精度 | 2026-08-11 |
| official | ReduceL2 | rank=2 static QDQ/int8 tensor, axes=1, keepdims=1 | cmsis-nn | generated_c_reducel2_s8 | REDUCEL2_001 | core/reducel2 | 官方 ReduceL2 QDQ/int8 rank2 axes=1 keepdims=1 数值精度 | 2026-08-11 |
| official | ReduceMax | rank=2 static QDQ/int8 tensor, axes=1, keepdims=1 | cmsis-nn | generated_c_reducemax_s8 | REDUCEMAX_001 | core/reducemax | 官方 ReduceMax QDQ/int8 rank2 axes=1 keepdims=1 数值精度 | 2026-08-11 |
| official | ReduceMean | rank=2 static QDQ/int8 tensor, axes=1, keepdims=1 | cmsis-nn | generated_c_reducemean_s8 | REDUCEMEAN_001 | core/reducemean | 官方 ReduceMean QDQ/int8 rank2 axes=1 keepdims=1 数值精度 | 2026-08-11 |
| official | ReduceMin | rank=2 static QDQ/int8 tensor, axes=1, keepdims=1 | cmsis-nn | generated_c_reducemin_s8 | REDUCEMIN_001 | core/reducemin | 官方 ReduceMin QDQ/int8 rank2 axes=1 keepdims=1 数值精度 | 2026-08-11 |
| official | ReduceProd | rank=2 static QDQ/int8 tensor, axes=1, keepdims=1 | cmsis-nn | generated_c_reduceprod_s8 | REDUCEPROD_001 | core/reduceprod | 官方 ReduceProd QDQ/int8 rank2 axes=1 keepdims=1 数值精度 | 2026-08-11 |
| official | ReduceSum | rank=2 static QDQ/int8 tensor, axes=1, keepdims=1 | cmsis-nn | generated_c_reducesum_s8 | REDUCESUM_001 | core/reducesum | 官方 ReduceSum QDQ/int8 rank2 axes=1 keepdims=1 数值精度 | 2026-08-11 |
| official | Reshape | static QDQ/int8 tensor, constant shape, same element count | cmsis-nn | generation_time_shape_alias_or_layout_copy | RESHAPE_001 | core/reshape | 官方 Reshape QDQ/int8 静态形状数值顺序 | 2026-08-11 |
| official | Round | static QDQ/int8 tensor | cmsis-nn | generated_c_round_s8 | ROUND_001 | core/round | 官方 Round QDQ/int8 静态张量数值精度 | 2026-08-11 |
| official | Selu | static QDQ/int8 tensor, alpha/gamma attributes | cmsis-nn | generated_c_selu_s8 | SELU_001 | core/selu | 官方 Selu QDQ/int8 alpha/gamma 属性数值精度 | 2026-08-11 |
| official | Sigmoid | static QDQ/int8 tensor | cmsis-nn | generated_c_sigmoid_s8 | SIGMOID_001 | core/sigmoid | 官方 Sigmoid QDQ/int8 静态张量数值精度 | 2026-08-11 |
| official | Sign | static QDQ/int8 tensor | cmsis-nn | generated_c_sign_s8 | SIGN_001 | core/sign | 官方 Sign QDQ/int8 静态张量数值精度 | 2026-08-11 |
| official | Slice | static QDQ/int8 tensor, constant starts/ends/axes/steps | cmsis-nn | generated_c_slice_s8 | SLICE_001 | core/slice | 官方 Slice QDQ/int8 静态区间数值顺序 | 2026-08-11 |
| official | Softmax | static classification vector, int8 output path | cmsis-nn | cmsis_softmax_s8 | SOFTMAX_001 | core/softmax | 10 分类 Softmax int8 代码生成 | 2026-08-11 |
| official | Softmax | static classification vector, int8 output path | cmsis-nn | cmsis_softmax_s8 | SOFTMAX_002 | core/softmax | SqueezeNet 末端 float output Softmax int8 C 输出 | 2026-08-11 |
| official | Softplus | static QDQ/int8 tensor | cmsis-nn | generated_c_softplus_s8 | SOFTPLUS_001 | core/softplus | 官方 Softplus QDQ/int8 静态张量数值精度 | 2026-08-11 |
| official | Softsign | static QDQ/int8 tensor | cmsis-nn | generated_c_softsign_s8 | SOFTSIGN_001 | core/softsign | 官方 Softsign QDQ/int8 静态张量数值精度 | 2026-08-11 |
| official | Sqrt | static QDQ/int8 tensor, non-negative value domain | cmsis-nn | generated_c_sqrt_s8 | SQRT_001 | core/sqrt | 官方 Sqrt QDQ/int8 非负输入数值精度 | 2026-08-11 |
| official | Squeeze | static QDQ/int8 tensor, explicit axes removing dimensions of size 1 | cmsis-nn | generation_time_shape_alias_or_layout_copy | SQUEEZE_001 | core/squeeze | 官方 Squeeze QDQ/int8 静态去 1 维数值顺序 | 2026-08-11 |
| official | Sub | same-shape QDQ/int8 tensors, second input may be constant | cmsis-nn | generated_c_sub_s8 | SUB_001 | core/sub | 官方 Sub QDQ/int8 同形状常量分支数值精度 | 2026-08-11 |
| official | Tanh | static QDQ/int8 tensor | cmsis-nn | generated_c_tanh_s8 | TANH_001 | core/tanh | 官方 Tanh QDQ/int8 静态张量数值精度 | 2026-08-11 |
| official | ThresholdedRelu | static QDQ/int8 tensor, alpha attribute | cmsis-nn | generated_c_thresholdedrelu_s8 | THRESHOLDEDRELU_001 | core/thresholdedrelu | 官方 ThresholdedRelu QDQ/int8 alpha 属性数值精度 | 2026-08-11 |
| official | Transpose | static QDQ/int8 tensor, rank=4 explicit perm | cmsis-nn | cmsis_transpose_s8 | TRANSPOSE_001 | core/transpose | 官方 Transpose QDQ/int8 rank4 显式 perm 数值顺序 | 2026-08-11 |
| official | Unsqueeze | static data-path QDQ/int8 tensor, constant axes, element count unchanged | cmsis-nn | generated_c_unsqueeze_s8 | UNSQUEEZE_001 | core/unsqueeze | 官方 Unsqueeze QDQ/int8 数据路径静态升维数值顺序 | 2026-08-11 |
| official | Where | static bool condition, same-shape QDQ/int8 then/else tensors | cmsis-nn | generated_c_where_s8 | WHERE_001 | core/where | 官方 Where 静态 bool mask 选择 QDQ/int8 数据数值精度 | 2026-08-11 |
| official | Xor | two bool intermediate tensors (compare outputs), bool output consumed by Where | cmsis-nn | generated_c_xor_bool | XOR_001 | core/xor | 官方 Xor bool 逻辑中间张量驱动 Where 数值精度 | 2026-08-11 |
| official | Where | static bool condition, same-shape QDQ/int8 then/else tensors | cmsis-nn | generated_c_where_s8 | XOR_001 | core/xor | 官方 Xor bool 逻辑中间张量驱动 Where 数值精度 | 2026-08-11 |
| mixed | multiple | internal float32 graph without QDQ quantization section | reject | reject_without_quantization | NEG_001 | negative | float32 无 Q/DQ 模型正确拒绝 | 2026-08-11 |
| mixed | multiple | SqueezeNet int8 with QLinearConv, MaxPool, DQ/Concat/Q, QLinearGlobalAveragePool and terminal Softmax | cmsis-nn | cmsis_nn_squeezenet_int8 | NET_001 | networks | 真实 SqueezeNet 1.0 int8 图像分类网络导入评估 | 2026-08-11 |
| official | Conv | rank=4 NCHW, group=1, static shape, QDQ/int8 parameters | cmsis-nn | cmsis_conv2d_s8 | NET_002 | networks | 真实 MobileNetV2 int8/QLinear 图像分类网络导入评估 | 2026-08-11 |
| official | Conv | rank=4, group=input_channels, depthwise multiplier, QDQ/int8 parameters | cmsis-nn | cmsis_depthwise_conv2d_s8 | NET_002 | networks | 真实 MobileNetV2 int8/QLinear 图像分类网络导入评估 | 2026-08-11 |
| extension | QLinearAdd | two static int8 tensors or activation plus bias-like tensor | cmsis-nn | cmsis_elementwise_add_s8 | NET_002 | networks | 真实 MobileNetV2 int8/QLinear 图像分类网络导入评估 | 2026-08-11 |
| extension | QLinearGlobalAveragePool | static rank=4 quantized global average pool | cmsis-nn | cmsis_avgpool_s8 | NET_002 | networks | 真实 MobileNetV2 int8/QLinear 图像分类网络导入评估 | 2026-08-11 |
| official | QLinearMatMul | static fully connected compatible quantized matmul | cmsis-nn | cmsis_fully_connected_per_channel_s8 | NET_002 | networks | 真实 MobileNetV2 int8/QLinear 图像分类网络导入评估 | 2026-08-11 |
| mixed | multiple | SSD-MobileNet detection graph with multi-output heads/dynamic or postprocess ops | reject | unsupported_detection_boundary | NET_003 | networks | 真实 SSD-MobileNet int8 目标检测网络导入评估 | 2026-08-11 |
| official | Conv | rank=4 NCHW, group=1, static shape, QDQ/int8 parameters | cmsis-nn | cmsis_conv2d_s8 | NET_004 | networks | KWS DS-CNN-style int8 网络导入评估 | 2026-08-11 |
| official | Conv | rank=4, group=input_channels, depthwise multiplier, QDQ/int8 parameters | cmsis-nn | cmsis_depthwise_conv2d_s8 | NET_004 | networks | KWS DS-CNN-style int8 网络导入评估 | 2026-08-11 |
| official | MaxPool | rank=4 static NCHW, QDQ/int8 tensor, CMSIS-compatible pool window | cmsis-nn | cmsis_maxpool_s8 | NET_004 | networks | KWS DS-CNN-style int8 网络导入评估 | 2026-08-11 |
| official | Gemm | static 2-D fully connected form, optional bias, QDQ/int8 parameters | cmsis-nn | cmsis_fully_connected_s8 | NET_004 | networks | KWS DS-CNN-style int8 网络导入评估 | 2026-08-11 |
| official | Conv | rank=3 NCW, group=1, static shape, QDQ/int8 parameters | cmsis-nn | cmsis_conv1d_as_2d_s8 | NET_005 | networks | Tiny signal jump int8 时序分类网络导入评估 | 2026-08-11 |
| official | Gemm | static 2-D fully connected form, optional bias, QDQ/int8 parameters | cmsis-nn | cmsis_fully_connected_s8 | NET_005 | networks | Tiny signal jump int8 时序分类网络导入评估 | 2026-08-11 |
| mixed | multiple | EfficientNet-Lite quantized graph forms outside current layout/op whitelist | reject | unsupported_import_boundary | NET_006 | networks | EfficientNet-Lite4 int8 小型图像分类网络导入评估 | 2026-08-11 |
| mixed | multiple | static-shape float or QDQ graph executed by generated C reference backend | c-reference | c99_reference_runtime | TS_001 | networks | CWRU bearing vibration MLP float C reference 数值回归 | 2026-08-11 |
| mixed | multiple | static-shape float or QDQ graph executed by generated C reference backend | c-reference | c99_reference_runtime | TS_002 | networks | KWS DS-CNN PTQ INT8 Q/DQ float C reference 数值回归 | 2026-08-11 |
| mixed | multiple | static-shape float or QDQ graph executed by generated C reference backend | c-reference | c99_reference_runtime | TS_003 | networks | KWS DS-CNN QAT INT8 Q/DQ float C reference 数值回归 | 2026-08-11 |
| mixed | multiple | static-shape float or QDQ graph executed by generated C reference backend | c-reference | c99_reference_runtime | TS_004 | networks | MET hybrid 双输入 float C reference 数值回归 | 2026-08-11 |
| mixed | multiple | static-shape float or QDQ graph executed by generated C reference backend | c-reference | c99_reference_runtime | TS_005 | networks | STWIN vowel IMU CNN float C reference 数值回归 | 2026-08-11 |
| official | Conv | rank=4 NCHW, group=1, static shape, QDQ/int8 parameters | cmsis-nn | cmsis_conv2d_s8 | TOPO_001 | topology | Conv->Relu->Pool->Flatten->FC 组合链路 | 2026-08-11 |
| official | MaxPool | rank=4 static NCHW, QDQ/int8 tensor, CMSIS-compatible pool window | cmsis-nn | cmsis_maxpool_s8 | TOPO_001 | topology | Conv->Relu->Pool->Flatten->FC 组合链路 | 2026-08-11 |
| official | Gemm | static 2-D fully connected form, optional bias, QDQ/int8 parameters | cmsis-nn | cmsis_fully_connected_s8 | TOPO_001 | topology | Conv->Relu->Pool->Flatten->FC 组合链路 | 2026-08-11 |
| official | Conv | rank=3 NCW, group=1, static shape, QDQ/int8 parameters | cmsis-nn | cmsis_conv1d_as_2d_s8 | TOPO_002 | topology | Conv1d->Relu->Conv1d->Relu->Flatten->FC 时序分类链路 | 2026-08-11 |
| official | Gemm | static 2-D fully connected form, optional bias, QDQ/int8 parameters | cmsis-nn | cmsis_fully_connected_s8 | TOPO_002 | topology | Conv1d->Relu->Conv1d->Relu->Flatten->FC 时序分类链路 | 2026-08-11 |
| official | QLinearConv | static rank=4 convolution with per-channel weight scale | cmsis-nn | cmsis_qlinearconv_per_channel_s8 | TOPO_003 | topology | 真实 MNIST QLinear int8 分类链路 | 2026-08-11 |
| official | MaxPool | rank=4 static NCHW, QDQ/int8 tensor, CMSIS-compatible pool window | cmsis-nn | cmsis_maxpool_s8 | TOPO_003 | topology | 真实 MNIST QLinear int8 分类链路 | 2026-08-11 |
| extension | QLinearAdd | two static int8 tensors or activation plus bias-like tensor | cmsis-nn | cmsis_elementwise_add_s8 | TOPO_003 | topology | 真实 MNIST QLinear int8 分类链路 | 2026-08-11 |
| official | QLinearMatMul | static fully connected compatible quantized matmul | cmsis-nn | cmsis_fully_connected_per_channel_s8 | TOPO_003 | topology | 真实 MNIST QLinear int8 分类链路 | 2026-08-11 |

## 未通过用例 (FAIL)

以下用例代表目标能力或拒绝边界，但本轮执行未达到 registry 预期：

*所有用例均已通过。*

## 已规划但未验证

以下 support matrix 行尚未绑定 PASS 用例，不能作为当前能力声明：

*暂无未覆盖 planned 行。*
