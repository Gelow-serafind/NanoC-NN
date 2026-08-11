# 031: 官方归约扩展批量 TDD 迭代（100% 计划 W1d-a）

## 基本信息

- **日期**: 2026-08-12
- **前置迭代**: `030_graph_attr_activation_batch`
- **触发来源**: `PLAN_100_SUPPORT.md` 图谱 100% 收敛计划 W1d-a

## 目标

按收获队列执行 W1d-a：最贴近既有 reduce 框架的归约扩展。本轮选择：

- `ReduceLogSum`
- `ReduceLogSumExp`
- `ReduceSumSquare`

（W1c 折叠批因需 empty-runtime 路径延后；W1d-b 池化/Lp/变体留给下一批）

## 新增/修改的测试用例

| case | ONNX op | 支持形态 | 数值数据集 |
|------|---------|----------|------------|
| `REDUCELOGSUM_001` | `ReduceLogSum` | rank=2、`axes=[1]`、`keepdims=1` | `reducelogsum_qdq_smoke` |
| `REDUCELOGSUMEXP_001` | `ReduceLogSumExp` | rank=2、`axes=[1]`、`keepdims=1` | `reducelogsumexp_qdq_smoke` |
| `REDUCESUMSQUARE_001` | `ReduceSumSquare` | rank=2、`axes=[1]`、`keepdims=1` | `reducesumsquare_qdq_smoke` |

## 执行结果

首次执行 3 个新 case 全部失败（unsupported），符合 TDD 预期。

## 代码修改

1. converter：`SUPPORTED_OPS` / `_int8_contract` / reduce 量化集合增加 3 个算子，
   复用 `_reduce_mean_quant_info`（rank=2 axes=1 keepdims=1 约束）。
2. codegen：`_generated_reduce_call` 增加 3 个归约数学分支（sum+logf、
   sum(expf)+logf、sum(x²)），各白名单/分派点同步注册。

## 关键发现：pipeline 假阳性

实测 `Cast`/`Constant` 的 pipeline CLI 报告 `codegen status: ok`，但生成的
`model.c` 是 `NANOC_STATUS_BLOCKED` 空 stub——违反 AGENTS.md
"生成物优先于报告状态" 的规则。TDD target 的 smoke run 能抓到（runtime_ok=False），
说明测试侧已有护栏，但 CLI 侧报告不可靠。已记录到 `PLAN_100_SUPPORT.md`，
折叠批实现 empty-runtime 拷贝路径时一并修复。

## 最终结果

执行命令：

```bash
PYTHONPATH=src python tdd/scripts/run_tests.py --validate-only
PYTHONPATH=src python tdd/scripts/run_tests.py --mode target --case <3 个新 case> --generate
PYTHONPATH=src python tdd/scripts/run_numeric_tests.py --case <3 个新 case> --generate
PYTHONPATH=src python tdd/scripts/run_regression.py --generate
PYTHONPATH=src python tdd/scripts/generate_support_map.py
PYTHONPATH=src python tdd/scripts/run_tests.py --mode target --generate
```

结果：

- validate-only: `95` 个用例规格校验通过
- 新增 3 个 case target: 全部 PASS
- 新增 3 个 case numeric: 全部 PASS，均为 `top1=2/2`、饱和率 `0.00`、最大绝对误差 `0.0`
- regression: PASS（73/73）
- target: `95/95 PASS`

## 发现与后续

- 归约扩展完全复用既有框架，3 个算子一次性收敛，无卡点。
- 100% 计划下一批（W1d-b）：`GlobalMaxPool` `GlobalLpPool` `LpPool` `LpNormalization`
  `CumSum` `Mean` `Sum`，其中 `Mean`/`Sum` 是变参逐元素算子、Lp 是池化族，需按各自
  框架接入。
