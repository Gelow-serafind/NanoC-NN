# Desktop Packaging

本目录保存 NanoC-NN Desktop 的 PyInstaller 打包配置。

## macOS

安装 UI 打包依赖：

```bash
conda run -n nanoc-onnx-examples python -m pip install -e ".[ui]"
```

构建：

```bash
conda run -n nanoc-onnx-examples bash ui/desktop_pyside6/packaging/build_macos.sh
```

生成物：

```text
dist/NanoC-NN.app
```

## Windows

Windows 包需要在 Windows 上构建：

```powershell
python -m pip install -e ".[ui]"
.\ui\desktop_pyside6\packaging\build_windows.ps1
```

生成物：

```text
dist\NanoC-NN
```

## 当前策略

- 使用 PyInstaller 打包 PySide6 桌面程序。
- 桌面图标源文件为 `ui/desktop_pyside6/assets/app_icon.png`，macOS bundle 使用 `ui/desktop_pyside6/packaging/app_icon.icns`。
- 将 `src/` 放入 app bundle，用于继续调用现有 `nanoc_nn` pipeline。
- 将 `tdd/fixtures/` 放入 app bundle，用于桌面 UI 的 MNIST / SqueezeNet 示例。
- 运行期输出写入用户目录：

```text
~/Library/Application Support/NanoC-NN/work
```

## 后续事项

- macOS 正式分发前需要 codesign 和 notarization。
- Windows 正式分发前需要图标、版本信息和安装包脚本。
- 后续可以把 UI 入口纳入 Python package，减少 spec 中的数据路径配置。

## 当前验证记录

2026-07-05 在 macOS / arm64 / conda `nanoc-onnx-examples` 环境下已验证：

```text
dist/NanoC-NN.app
size: ~117 MB
open dist/NanoC-NN.app: can launch
```

PyInstaller 构建末尾出现 ad-hoc codesign 警告：

```text
resource fork, Finder information, or similar detritus not allowed
```

当前作为本机开发包可以继续验证；正式分发前需要单独处理签名、清理扩展属性和 notarization。
