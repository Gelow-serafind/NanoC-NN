# mnist_hand_drawn

This dataset stores hand-drawn MNIST-style samples collected through the local TDD capture UI.

Each sample is stored as a 28x28 grayscale `float_pixels` array in `[0, 1]`, with an optional label from `0` to `9`.
It is a persistent fixture and must not be placed under `tdd/work/` or `tdd/results/`.

Capture UI:

```bash
python tdd/tools/mnist_capture/server.py
```

Numeric check after collecting samples:

```bash
python tdd/scripts/run_numeric_tests.py --case TOPO_003 --dataset mnist_hand_drawn --generate
```
