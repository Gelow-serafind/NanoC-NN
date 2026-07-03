# NanoC-NN 能力集

> 本文件由 `python tdd/scripts/run_tests.py` 自动生成，禁止手动编辑。
> 最后更新: 2026-07-03T16:19:28.126394

**能力集大小: 2 / 12 (16%)**

## 已验证能力 (PASS)

以下用例通过测试，代表代码生成器已确认支持的能力：

| 用例 ID | 分类 | 能力描述 | 验证日期 |
|---------|------|---------|---------|
| MAXPOOL_001 | core/maxpool | 标准 2x2 stride=2 MaxPool int8 代码生成 | 2026-07-03 |
| SOFTMAX_001 | core/softmax | 10 分类 Softmax int8 代码生成 | 2026-07-03 |

## 未通过用例 (FAIL)

以下用例代表目标能力但尚未实现：

| 用例 ID | 分类 | 目标能力 | 阻塞原因 |
|---------|------|---------|---------|
| CONV_001 | core/conv | 1x1 pointwise 单通道 | converter 未提取节点级量化字段 |
| CONV_002 | core/conv | 3x3 标准 SAME padding | converter 未提取节点级量化字段 |
| CONV_003 | core/conv | stride=2 下采样 | converter 未提取节点级量化字段 |
| CONV_004 | core/conv | 非对称输入 zp!=0 | converter 未提取节点级量化字段 |
| GEMM_001 | core/gemm | 最小对称 FC 无 bias | converter 未提取节点级量化字段 |
| GEMM_002 | core/gemm | 非对称输入 zp!=0 含 bias | converter 未提取节点级量化字段 |
| GEMM_003 | core/gemm | 非 4 对齐维度 13->7 | converter 未提取节点级量化字段 |
| GEMM_004 | core/gemm | 中等规模 64->32 | converter 未提取节点级量化字段 |
| NEG_001 | negative | float32 无 Q/DQ 模型正确拒绝 | 预期 unsupported，实际 blocked |
| TOPO_001 | topology | Conv->Relu->Pool->Flatten->FC 组合链路 | converter 未提取节点级量化字段 |
