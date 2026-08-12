# 035: ArgMax/ArgMin 索引 ABI 转正迭代

## 基本信息

- **日期**: 2026-08-12
- **前置迭代**: `034_graph_convergence_final`
- **触发来源**: 100% 收敛计划终态后，将高价值拒绝边界拉回能力集的第一批
  （PLAN 批 11 / W0 索引 ABI）。

## 目标

把 `ArgMax`/`ArgMin` 从 `unsupported` 拒绝边界转成 PASS 能力，打通**非 int8 外部
ABI** 这条横切能力：索引输出模型经新增入口 `nanoc_model_run_index(const int8_t*, int32_t*)`
暴露 int32 index，numeric runner 增加精确整数比较模式。

## 架构决策（用户确认）

非 int8 外部 ABI 采用**独立入口 + int32**：
- 索引模型新增 `nanoc_model_run_index(const int8_t *input, int32_t *output)`；
- 现有 int8 模型 `nanoc_model_run(const int8_t*, int8_t*)` 契约完全不变（零回归）；
- numeric runner 按 `NumericCheck.index_compare` 分流，ONNX int64 index 与 C int32
  index 做精确相等比较（非 float 反量化）。

## 本批次内容

1. **TDD 资产**：`ARGSMAX_001`/`ARGSMIN_001` case（rank2 axis=1 keepdims=1，
   `[1,4]` int8 → int64[1] index）；registry 新增 `NumericCheck.index_compare`；
   support matrix 用 `ONNX_ARGMAX_QDQ_INT8`/`ONNX_ARGMIN_QDQ_INT8` ok 行替换
   `_NON_INT8_UNSUPPORTED` 拒绝行；生成器 `_gen_arg_case` 转正；两个非退化 smoke
   数据集（4 样本覆盖不同 index {0,1,2,3}）。

2. **converter**：`SUPPORTED_OPS`/`_int8_contract` 白名单加 ArgMax/ArgMin；
   `_argmax_quant_info` 仅提取输入侧 int8 量化（输出 int64 index 无量化段），
   cmsis_nn 字段含 `index_count`/`axis`/`keepdims`/`select_last_index`。

3. **codegen**：`_arg_layer`/`_generated_argmax_call`（直接对 int8 q-value 行扫描，
   scale>0 保序故与反量化一致；`select_last_index=1` 时用 `>=`/`<=` 取末极值）；
   `_cmsis_tensor_runtime_run_body` 新增 argmax/argmin 分发；`_model_h`/`_model_c`/
   `_main_c` 对索引模型发射 `nanoc_model_run_index` + `NANOC_MODEL_OUTPUT_INDEX_COUNT`；
   mapper `RUNTIME_ACTIONS`/`RENDERED_RUNTIME_OPS` 与 quantization
   `REQUIRED_NODE_QUANT_FIELDS` 登记。

4. **numeric runner**：`_write_index_c_runner`（int32 buffer + index 入口）、
   `_run_index_c_runner`（int32 解析）、`_compare_index_outputs`（逐元素精确相等）；
   `_run_numeric_case` 按 `index_compare` 分流，跳过输出反量化。

5. **NEG 边界退役**：NEG_002/003（ArgMax/ArgMin 非 int8 ABI 拒绝）随能力转正移除，
   原模型结构转为 ARGSMAX_001/ARGSMIN_001 正例。

## 最终结果

- validate-only: `106` 个用例（106 - 2 NEG + 2 正例）
- target: `106/106 PASS`
- numeric: `84/84 PASS`（原 82 + ARGSMAX_001/ARGSMIN_001）
- regression: PASS（84/84 numeric）
- 能力集文件 `CAPABILITIES.md`、支持图谱同步

## 结论

W0（ABI 基础）从拒绝边界转正：`ArgMax`/`ArgMin` 进入 PASS 能力集，同时建立
**非 int8 索引输出 ABI** 这条横切能力，为后续 TopK（共享 index ABI）铺路。
分类头（backbone → argmax）成为可生成、可数值验收的真实 MCU 形态。

## 遗留（非阻塞）

- `select_last_index=1`（取末极值）在 C 侧已按 `>=`/`<=` 实现，但无独立数值用例
  保护（当前用例默认 0）。
- `TopK`（int64 index + values 双输出）可复用本批 index ABI，列入后续批次。
- 更一般 axis（如轴 0）的 ArgMax/ArgMin 需扩展 codegen 行扫描方向。
