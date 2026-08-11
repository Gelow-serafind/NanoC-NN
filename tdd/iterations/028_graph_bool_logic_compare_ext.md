# 028: 官方 bool 逻辑与比较扩展批量 TDD 迭代

## 基本信息

- **日期**: 2026-08-11
- **前置迭代**: `027_graph_compare_select_reduce_batch`
- **触发来源**: 027 后续建议，继续补比较扩展与 bool 逻辑算子

## 目标

沿 ONNX 官方图谱继续扩展 bool 中间张量子图能力。027 建立了 compare 产 bool、`Where` 消费 bool 的框架，本轮把 bool 子图从"单个比较条件"扩展到"多条件组合"与"取反"，覆盖阈值窗口、互斥、反向选择等 MCU 常见图结构。本轮选择：

- `GreaterOrEqual`
- `LessOrEqual`
- `And`
- `Or`
- `Not`
- `Xor`

本轮刻意没有纳入 `ArgMax`。`ArgMax` 输出 index，涉及模型最终输出非 int8 的 ABI 和 numeric runner 扩展，延续 027 的决定，留给单独迭代。

## 新增/修改的测试用例

| case | ONNX op | 支持形态 | 数值数据集 |
|------|---------|----------|------------|
| `GREATEROREQUAL_001` | `GreaterOrEqual` | 同形状 QDQ/int8 比较，bool 中间输出由 `Where` 消费 | `greaterorequal_where_qdq_smoke` |
| `LESSOREQUAL_001` | `LessOrEqual` | 同形状 QDQ/int8 比较，bool 中间输出由 `Where` 消费 | `lessorequal_where_qdq_smoke` |
| `AND_001` | `And` | 两个 compare 产出的 bool 输入，bool 输出由 `Where` 消费（阈值窗口） | `and_where_qdq_smoke` |
| `OR_001` | `Or` | 两个 compare 产出的 bool 输入，bool 输出由 `Where` 消费（双上界） | `or_where_qdq_smoke` |
| `NOT_001` | `Not` | 单个 compare 产出的 bool 输入，bool 输出由 `Where` 消费（取反） | `not_where_qdq_smoke` |
| `XOR_001` | `Xor` | 两个 compare 产出的 bool 输入，bool 输出由 `Where` 消费（窗口互斥） | `xor_where_qdq_smoke` |

## 支持形态边界

- bool 只作为内部中间张量，由 compare 算子产出，以 `int8_t` 的 `0/1` buffer 承载，最终由 `Where` 消费；不暴露为模型最终输出，不覆盖运行时 bool 外部输入。
- `And/Or/Xor/Not` 的输入必须是已识别的 bool 生产者（compare 或逻辑算子）输出；bool initializer 直接作为逻辑算子输入的首个形态暂不纳入，留给后续。

## 执行结果

首次执行 6 个新 case 全部失败，符合 TDD 预期：

```text
GREATEROREQUAL_001 expected=ok actual=unsupported
LESSOREQUAL_001     expected=ok actual=unsupported
AND_001             expected=ok actual=unsupported
OR_001              expected=ok actual=unsupported
NOT_001             expected=ok actual=unsupported
XOR_001             expected=ok actual=unsupported
```

失败根因均为 converter 白名单未包含新算子。

## 代码修改

1. converter:
   - `SUPPORTED_OPS` / `_int8_contract` 增加 6 个官方算子白名单。
   - `GreaterOrEqual/LessOrEqual` 复用 `_compare_quant_info`，输出 `generated_c_{op}_bool`。
   - 新增 `_bool_logic_quant_info`：校验逻辑算子输入全部为已识别的 bool 生产者、同形状，输出 `generated_c_{op}_bool`。
   - 新增 `_find_bool_producer`：在已解析 node_quant 中按 BOOL elem_type 定位 bool 生产者。

2. codegen:
   - mapper：`RUNTIME_ACTIONS` / `RENDERED_RUNTIME_OPS` 注册 6 个算子。
   - quantization：`REQUIRED_NODE_QUANT_FIELDS` 覆盖 compare 扩展与 bool 逻辑算子。
   - generator：
     - `_generated_compare_call` 增加 `>=`、`<=` 分支。
     - 新增 `_generated_bool_logic_call`：逐元素 `&&` / `||` / `!` / 异或，输入以 `!=0` 归一为 0/1。
     - 新增 `_bool_logic_layer`，注册到 `_runtime_layers` 与两个 run body 分派。
     - tensor-aware run body 增加 bool 逻辑输入的表达式解析（含单输入 `Not`）。
     - 顺带合并 027 遗留的 compare/where 重复分派分支。

## 最终结果

执行命令：

```bash
PYTHONPATH=src python tdd/scripts/run_tests.py --validate-only
PYTHONPATH=src python tdd/scripts/run_tests.py --mode target --case <6 个新 case> --generate
PYTHONPATH=src python tdd/scripts/run_numeric_tests.py --case <6 个新 case> --generate
PYTHONPATH=src python tdd/scripts/run_regression.py --generate
PYTHONPATH=src python tdd/scripts/generate_support_map.py
PYTHONPATH=src python tdd/scripts/run_tests.py --mode target --generate
```

结果：

- validate-only: `82` 个用例规格校验通过
- 新增 6 个 case target: 全部 PASS（api/compile/run 全 OK）
- 新增 6 个 case numeric: 全部 PASS，均为 `top1=2/2`、饱和率 `0.00`、最大绝对误差 `0.0`
- regression: `60/60` numeric PASS（含 6 新增 + 54 既有，无退化）
- target: `82/82 PASS`，`CAPABILITIES.md` 与支持图谱同步

## 设计要点

- 数据集刻意构造了**非退化、可判别**的混值输出：每个 case 的 bool 组合都同时含 T 和 F 元素，且样本跨越阈值/窗口边界，避免"条件全真/全假导致结构 PASS 但逻辑错误被隐藏"（对应 026 迭代对 ReduceSum/ReduceMean 误用的警告）。
- `And/Xor` 用 `Greater(>下界) + Less(<上界)` 窗口结构，`Or` 用 `Less + Less` 双上界结构，确保 OR 组合不塌缩为恒真。
- bool 逻辑算子不携带量化参数、无权重；输入 buffer 通过 `_tensor_symbol_map` 由 compare 层输出符号自然解析，无需额外 ABI。

## 发现与后续

本轮确认 bool 中间张量子图框架可自然扩展：从"单比较→Where"到"多比较→逻辑组合→Where"，converter 只需识别 bool 生产者，codegen 的 buffer 符号机制自动承接。

后续建议：

- 单独迭代 `ArgMax/ArgMin`，正式处理 index 输出和 numeric runner 的非 int8 输出验收。
- 支持 bool initializer 直接作为逻辑算子输入，覆盖静态 mask 组合。
- 支持 bool 逻辑算子级联（如 `Not(And(...))` 多层嵌套），当前首个形态覆盖单层逻辑。
- 从 `NET_004` KWS-style 或其他真实时序网络中继续提炼真实模型驱动 case。
