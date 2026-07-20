# TDD 终端实机验收

`tdd/terminal/` 是 NanoC-NN TDD 的最后一环，用于把已经通过结构验收和
Host 数值验收的模型，继续放到真实 Arm Cortex-M 开发板上做 ONNX / Host C /
ARM C 三路对比。

这层测试是可选门禁：没有接入 STM32 时，常规全量测试仍以
`run_tests.py` 和 `run_numeric_tests.py` 为基础；接入 STM32 后，终端测试会在
同一套 case / dataset 上追加 ARM C 推理结果，并把结果写入支持图谱。

## 分层关系

```text
run_tests.py
  结构验收：ONNX -> converter -> codegen -> C99 smoke

run_numeric_tests.py
  Host 数值验收：ONNX Runtime -> generated Host C

tdd/terminal/scripts/run_terminal_tests.py
  终端实机验收：ONNX Runtime -> generated Host C -> ARM C
```

终端实机验收通过后，才表示该 case 完成了真实 MCU 运行闭环。

## 目录结构

```text
tdd/terminal/
  boards/       板卡和连接配置
  cases/        终端实机 case 配置，绑定已有 TDD case
  reports/      终端测试报告，latest + history
  scripts/      串口、烧录、三路对比脚本
```

`boards/` 和 `cases/` 使用 JSON，是为了避免给 TDD runner 引入额外运行时依赖。

## 执行方式

列出可执行终端 case：

```bash
python tdd/terminal/scripts/run_terminal_tests.py --list
```

自动检测板卡；检测不到时跳过，不影响普通回归：

```bash
python tdd/terminal/scripts/run_terminal_tests.py --case TOPO_003 --generate --auto
```

要求必须有板卡并烧录执行：

```bash
python tdd/terminal/scripts/run_terminal_tests.py \
  --case TOPO_003 \
  --board stm32f103vetx_uart_tim6 \
  --generate \
  --flash \
  --require-board
```

如果固件已经烧录，可以只跑串口推理：

```bash
python tdd/terminal/scripts/run_terminal_tests.py \
  --case TOPO_003 \
  --board stm32f103vetx_uart_tim6 \
  --no-flash \
  --require-board
```

## 判定规则

每个终端 case 必须至少检查：

- ONNX top1 与 Host C top1 是否一致。
- ONNX top1 与 ARM C top1 是否一致。
- Host C int8 输出与 ARM C int8 输出是否一致，或满足 case 配置的误差阈值。
- ARM response checksum、status、case id、output size 是否正确。
- 板端 `elapsed_us` 是否被记录。

默认常规全量测试不强制接板。发布、板级验证或硬件在位 CI 可以开启
`--terminal auto|required`，把 ARM C 作为最后一环。

## 支持图谱

`tdd/scripts/generate_support_map.py` 会读取
`tdd/terminal/reports/terminal_latest.json`。如果某个 TDD case 有终端报告，图谱
case 节点会出现 `terminal:` 明细，便于从 ONNX 支持图谱上直接看到哪些能力已经
完成 ARM C 闭环。
