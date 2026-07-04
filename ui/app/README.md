# NanoC-NN UI App

这是 NanoC-NN UI 的第一版可运行网页。它使用一个本地 Python server 包装现有 CLI，前端为无框架静态页面。

## 启动

推荐在已有 `nanoc-onnx-examples` 环境中启动：

```bash
conda run -n nanoc-onnx-examples python ui/app/server.py --port 8766
```

打开：

```text
http://127.0.0.1:8766/
```

## 当前已打通功能

- 选择本地 ONNX 文件并上传。
- 使用仓库内 MNIST int8 fixture 作为示例模型。
- 解析 ONNX 输入/输出、opset、节点数量、算子统计。
- 配置 target/backend/SRAM/Flash/out dir/prefix。
- 调用现有 `nanoc onnx-to-cmsis` pipeline。
- 展示转换状态、命令、日志、报告路径和生成物文件。
- 浏览生成文件内容。
- 运行 TDD 稳定回归。

## 设计约束

- UI 后端只调用现有 converter/codegen/pipeline，不复制转换逻辑。
- 上传模型和转换输出写入 `ui/work/`，不进入版本管理。
- 常规转换不默认设置 SRAM/Flash 预算；只有用户填写预算时才触发 `oversize` 平台拦截。
