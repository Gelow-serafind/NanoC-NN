# 032: 官方池化/Lp/变体批量 TDD 迭代（100% 计划 W1d-b）

## 基本信息

- **日期**: 2026-08-12
- **前置迭代**: `031_graph_reduce_ext_batch`
- **触发来源**: `PLAN_100_SUPPORT.md` 图谱 100% 收敛计划 W1d-b

## 目标

按收获队列执行 W1d-b。本轮选择：

- `GlobalMaxPool`、`GlobalLpPool`、`LpPool`（池化族）
- `LpNormalization`（行 Lp 归一化）
- `CumSum`（轴累计和）
- `Mean`、`Sum`（双输入逐元素）

## 新增/修改的测试用例

| case | ONNX op | 支持形态 |
|------|---------|----------|
| `GLOBALMAXPOOL_001` | `GlobalMaxPool` | rank=4 NCHW 全局空间最大池化 |
| `GLOBALLPPOOL_001` | `GlobalLpPool` | rank=4 NCHW p=2 全局 Lp 池化 |
| `LPPOOL_001` | `LpPool` | rank=4 kernel=2 stride=1 窗口 Lp 池化 |
| `LPNORMALIZATION_001` | `LpNormalization` | rank=2 axis=1 p=2 行 Lp 归一化 |
| `CUMSUM_001` | `CumSum` | rank=2 axis=1 累计和 |
| `MEAN_001` | `Mean` | 双输入同形状逐元素均值 |
| `SUM_001` | `Sum` | 双输入同形状逐元素和 |

## 代码修改

1. converter：7 个算子白名单；`Mean/Sum` 复用 elementwise 量化；5 个空间/行
   归约算子新增 `_generated_spatial_quant_info`（静态 shape + p/axis/kernel 属性）。
2. codegen：
   - 新增 `_generated_spatial_layer` / `_generated_spatial_call`，覆盖
     globalmaxpool/globallppool/lppool/lpnormalization/cumsum 的 C99 路径。
   - `_generated_binary_call` 增加 mean/sum 数学分支。
   - `Mean/Sum` 需加入 `_quantized_weight_declarations` 的 constant-input 集合与
     tensor run body 的二元分派（否则落 `bias_values`/`input_size` KeyError）。

## 执行结果

首次执行 7 个新 case 全部失败（unsupported）。实现后：

- validate-only: `102` 个用例
- 新增 7 个 case target: 全部 PASS
- 新增 7 个 case numeric: 全部 PASS，`max_abs=0.0`
- regression: PASS（84/84 结构、80/80 numeric）
- target: `102/102 PASS`

## 发现与后续

- 池化/空间归约新增独立 `_generated_spatial_call` 路径，与既有 reduce/pool 框架分离，
  7 个算子一次性收敛。
- 100% 计划下一批（W0）：`ArgMax`/`ArgMin`，处理 index 输出 ABI 策略。
