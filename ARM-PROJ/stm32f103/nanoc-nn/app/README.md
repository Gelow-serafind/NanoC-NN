# NanoC-NN ARM TDD runner

`app/` is the board-side runner used by host TDD scripts to execute one model
case on STM32F103 hardware. The host sends a binary request over USART1, the
board dispatches the selected case, measures model execution time with the
continuous TIM6 microsecond clock, and returns status, elapsed time, and output
bytes.

## Time base

The project uses an 8 MHz HSE and PLL x9, so `SYSCLK` is 72 MHz. APB1 is
divided by 2, and STM32F1 timers on a divided APB bus receive `PCLK1 x 2`, so
TIM6 also runs from 72 MHz.

TIM6 is configured as:

| Field | Value |
| --- | --- |
| Input clock | 72 MHz |
| Prescaler | 71 |
| Counter frequency | 1 MHz |
| Tick period | 1 us |
| Auto reload | 65535 |
| Overflow period | 65.536 ms |

`nanoc_time_now_us()` combines the 16-bit TIM6 counter and overflow count into a
continuous microsecond timestamp. Case inference timing is measured by taking
two timestamps around the case `run` function, so UART transfer time is not
included.

## Frame format

Request header is 16 bytes followed by `input_size` payload bytes:

| Offset | Size | Field |
| --- | --- | --- |
| 0 | 4 | magic `NCTD` little-endian (`0x4454434E`) |
| 4 | 1 | protocol version, currently `1` |
| 5 | 1 | command, currently `1` = run case |
| 6 | 2 | case id, little-endian |
| 8 | 4 | input size, little-endian |
| 12 | 4 | additive checksum of header bytes `0..11` and payload |

Response header is 20 bytes followed by `output_size` payload bytes:

| Offset | Size | Field |
| --- | --- | --- |
| 0 | 4 | magic `NCTR` little-endian (`0x5254434E`) |
| 4 | 1 | protocol version, currently `1` |
| 5 | 1 | status |
| 6 | 2 | case id, little-endian |
| 8 | 4 | elapsed inference time in microseconds |
| 12 | 4 | output size, little-endian |
| 16 | 4 | additive checksum of header bytes `0..15` and payload |

## Case rule

Each case file only exposes a narrow `run` function. It receives an already
shaped input byte array and an output byte buffer. UART framing, protocol
parsing, dispatch, and timing stay outside the case implementation.

To add a case:

1. Add a value to `nanoc_case_id_t` in `nanoc_case.h`.
2. Add `app/cases/nanoc_case_<name>.c/.h`.
3. Register the descriptor in `nanoc_case.c`.
4. Add the new source file to the root `CMakeLists.txt`.

The initial `NANOC_CASE_DUMMY_ECHO` case echoes input to output and exists only
to verify UART framing, dispatch, and timing before real generated models are
linked in.
