# NanoC-NN Desktop UI (PySide6)

这是 NanoC-NN 的桌面 UI 原型，目标风格接近 STM32CubeMX 一类跨平台工程工具，而不是网页玩具。

## 安装依赖

当前项目把 PySide6 放在可选依赖 `ui` 中：

```bash
conda run -n nanoc-onnx-examples python -m pip install -e ".[ui]"
```

## 启动

```bash
conda run -n nanoc-onnx-examples python ui/desktop_pyside6/main.py
```

## 当前功能

- 选择本地 ONNX 文件。
- 一键加载仓库内 MNIST / SqueezeNet fixture。
- 解析 ONNX 输入、输出、opset、producer、节点数量和算子统计。
- 配置 target、backend、SRAM、Flash、输出目录、C symbol prefix。
- 调用现有 `nanoc onnx-to-cmsis` pipeline。
- 显示 `ok` / `blocked` / `unsupported` / `oversize` 状态。
- 浏览生成报告和生成 C 文件。
- 底部显示命令输出日志。

## 设计约束

- UI 只调用现有 converter/codegen/pipeline，不复制转换逻辑。
- 运行产物写入 `ui/work/desktop/`，不进入版本管理。
- 常规转换不默认设置 SRAM/Flash 预算；只有用户填写预算时才触发 `oversize`。
