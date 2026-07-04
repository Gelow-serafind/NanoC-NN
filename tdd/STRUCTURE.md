# TDD 目录结构与运行规则

本文件定义 TDD 资产与运行产物的边界。核心原则：**人维护的东西永远不放在可清理目录里，脚本可清理目录必须显式标记为 runner-owned workdir。**

## 顶层目录

```text
tdd/
├── cases/                 # 永久：测试需求规格
├── fixtures/              # 永久：真实 ONNX、数据集、期望输出、外部问题样本
├── scripts/               # 永久：生成、结构测试、数值测试脚本
├── tools/                 # 永久：TDD 辅助工具，如数据集采集 UI
├── iterations/            # 永久：每轮 TDD 迭代记录
├── results/               # 半永久：测试报告、历史摘要，不放人工数据集
└── work/                  # 临时：脚本生成的模型、C 工程、runner、scratch，可整体删除
```

## 永久资产

`tdd/cases/` 是需求规格。它描述“我们承诺支持什么输入形态、什么行为是正确的”。

`tdd/fixtures/` 是测试资产。任何人工整理的数据、真实用户模型、完整网络、手写样本、期望输出，都必须放在这里：

```text
tdd/fixtures/
├── onnx/                  # 完整或真实来源 ONNX fixture
├── datasets/              # 输入样本数据集
│   └── <dataset_id>/
│       ├── dataset.json   # 样本、标签、输入缩放说明
│       └── README.md
└── expected/              # 可选：预先固化的参考输出
    └── <case_id>/
```

规则：

- `fixtures/` 下的内容默认是永久资产，不允许 runner 自动删除。
- 真实完整网络必须放在 `fixtures/onnx/`，并在对应 case 中说明来源。
- 数据集必须放在 `fixtures/datasets/`，不能放在 `results/` 或 `work/`。
- 如果样本由脚本生成，也应保留生成脚本或在 README 中说明生成规则。

## 临时工作区

`tdd/work/` 是唯一允许测试脚本清理的目录：

```text
tdd/work/
├── models/                # generate_models.py 生成的 ONNX 测试模型
├── structural/<case_id>/  # run_tests.py 结构测试工作区
└── numeric/<case_id>/     # run_numeric_tests.py 数值测试工作区
```

规则：

- `work/` 不进入版本管理。
- runner 只能删除带 `.nanoc_tdd_workdir` marker 的目录。
- 开发者临时实验可以放 `work/manual/`，但不要依赖其长期存在。
- 任何需要保留的实验结果，必须提升为 `fixtures/`、`cases/` 或 `iterations/`。

## 测试分层

TDD 测试分三层：

| 层级 | 入口 | 验收目标 |
|------|------|----------|
| 规格校验 | `run_tests.py --validate-only` | case、registry、模型生成器一致 |
| 结构验收 | `run_tests.py --mode target --generate` | converter/codegen、CMSIS-NN API、C99 smoke compile/run |
| 数值验收 | `run_numeric_tests.py --case <ID> --generate` | ONNX Runtime 输出与生成 C 输出对比 |

结构验收只能说明“生成物存在并可启动”，不能说明“推理结果正确”。完整网络能力必须进入数值验收。

## 完整网络测试

完整网络测试用于覆盖真实用户可能输入的 ONNX，而不是只覆盖单个算子。

完整网络 case 必须满足：

- ONNX fixture 固化在 `fixtures/onnx/`。
- 至少一个数据集固化在 `fixtures/datasets/`。
- case 规格说明网络结构、真实来源、输入/输出语义。
- 若期望 `ok`，必须逐步补充数值验收，不能只依赖 C99 smoke run。

当前第一条完整网络数值验收目标是 `TOPO_003`：真实 MNIST QLinear int8 分类链路。

## 数据集采集 UI

MNIST 手写样本采集 UI 位于：

```bash
python tdd/tools/mnist_capture/server.py
```

它会将样本写入：

```text
tdd/fixtures/datasets/mnist_hand_drawn/dataset.json
```

采集完成后，可以分别执行两个数据集的 ONNX-vs-C 对比：

```bash
python tdd/scripts/run_numeric_tests.py --case TOPO_003 --dataset mnist_synthetic_smoke --generate
python tdd/scripts/run_numeric_tests.py --case TOPO_003 --dataset mnist_hand_drawn --generate
```
