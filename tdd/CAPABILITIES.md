# NanoC-NN 能力集

> 本文件由 `python tdd/scripts/run_tests.py --mode target` 自动生成，禁止手动编辑。
> 最后更新: 2026-07-03T22:42:57.275678
> Git commit: `eeda9a3`

**能力集大小: 12 / 12 (100%)**

## 已验证能力 (PASS)

以下用例通过测试，代表代码生成器已确认支持的能力：

| 用例 ID | 分类 | 能力描述 | 验证日期 |
|---------|------|---------|---------|
| CONV_001 | core/conv | 1x1 pointwise 单通道 | 2026-07-03 |
| CONV_002 | core/conv | 3x3 标准 SAME padding | 2026-07-03 |
| CONV_003 | core/conv | stride=2 下采样 | 2026-07-03 |
| CONV_004 | core/conv | 非对称输入 zp!=0 | 2026-07-03 |
| GEMM_001 | core/gemm | 最小对称 FC 无 bias | 2026-07-03 |
| GEMM_002 | core/gemm | 非对称输入 zp!=0 含 bias | 2026-07-03 |
| GEMM_003 | core/gemm | 非 4 对齐维度 13->7 | 2026-07-03 |
| GEMM_004 | core/gemm | 中等规模 64->32 | 2026-07-03 |
| MAXPOOL_001 | core/maxpool | 标准 2x2 stride=2 MaxPool int8 代码生成 | 2026-07-03 |
| SOFTMAX_001 | core/softmax | 10 分类 Softmax int8 代码生成 | 2026-07-03 |
| NEG_001 | negative | float32 无 Q/DQ 模型正确拒绝 | 2026-07-03 |
| TOPO_001 | topology | Conv->Relu->Pool->Flatten->FC 组合链路 | 2026-07-03 |

## 未通过用例 (FAIL)

以下用例代表目标能力但尚未实现：

*所有用例均已通过。*
