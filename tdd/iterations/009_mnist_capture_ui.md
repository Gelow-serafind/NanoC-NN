# 迭代 009: MNIST 手写样本采集 UI

## 基本信息

- **日期**: 2026-07-04
- **前置迭代**: 008
- **触发来源**: 人类需求 / 需要采集真实手写数字样本，并用于 ONNX-vs-C 数值对比

## 目标

在 TDD 永久资产目录下加入一个本地 UI，用于采集手写数字图片，并自动记录到数据集 fixture。采集后，数值测试可以分别对 synthetic smoke 数据集和手写数据集执行 ONNX Runtime 与生成 C 推理对比。

## 新增内容

- `tdd/tools/mnist_capture/server.py`
  - 本地 HTTP 服务。
  - 提供静态采集 UI。
  - 接收 `float_pixels` 和 `label`，追加写入 fixture 数据集。

- `tdd/tools/mnist_capture/index.html`
  - 280x280 手写画板。
  - 标签选择。
  - 28x28 预览。
  - 保存样本到后端。

- `tdd/fixtures/datasets/mnist_hand_drawn/`
  - 新增持久数据集目录。
  - 初始 `dataset.json` 为空样本集。

## 数值测试扩展

`run_numeric_tests.py` 新增：

```bash
--dataset <dataset_id>
```

用于覆盖 registry 中默认数据集，方便同一 `TOPO_003` case 分别跑：

```bash
python tdd/scripts/run_numeric_tests.py --case TOPO_003 --dataset mnist_synthetic_smoke --generate
python tdd/scripts/run_numeric_tests.py --case TOPO_003 --dataset mnist_hand_drawn --generate
```

## 目录规则

采集出来的数据是人工资产，必须写入 `tdd/fixtures/datasets/`，不能写入 `tdd/work/` 或 `tdd/results/`。

## 下一步

采集若干带标签手写样本后，执行两个数据集的数值测试，并比较：

- ONNX-vs-C top1 一致率。
- C 输出饱和率。
- 若有 label，进一步比较 ONNX label accuracy 与 C label accuracy 的差异。
