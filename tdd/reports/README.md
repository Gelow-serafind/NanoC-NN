# TDD 可视化报告

本目录存放给人类阅读和开发跟踪使用的 TDD 可视化产物。这里的文件可以纳入版本管理，用于观察能力地图随迭代增长的过程。

## 当前产物

| 文件 | 说明 | 生成命令 |
|------|------|----------|
| `onnx_support_map.html` | 可拖动、可缩放的 ONNX 官方算子全集支持思维导图 | `python tdd/scripts/generate_support_map.py` |

## 目录边界

- `tdd/reports/`：版本化的人类报告和能力地图。
- `tdd/results/`：测试脚本输出的机器结果，例如 `latest.json`、`numeric_latest.json`。
- `tdd/work/`：测试和调试临时产物，不得作为人工数据集或长期报告来源。

## 使用规则

1. 新增或修改 `support_matrix.py`、`cases_registry.py`、ONNX schema catalog 或 `tdd/results/latest.json` 后，应重新生成 `onnx_support_map.html`。
2. 完成一个新算子或新 case 后，必须检查图谱中对应节点是否从“尚未开始/已登记但未完成”移动到正确状态。
3. 规划下一轮测试时，优先从图谱的“已登记但未完成”和“尚未开始”节点中选择目标，再反向创建最小 TDD case。
4. 图谱是开发导航，不是能力权威声明；能力权威仍是 `tdd/CAPABILITIES.md` 和全量回归结果。

## 重新生成

```bash
PYTHONPATH=tdd/work/python_deps:src /Users/tiedan/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 tdd/scripts/generate_support_map.py
```
