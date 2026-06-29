from __future__ import annotations

import torch
from torch import nn


class SignalJumpNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=8, out_channels=8, kernel_size=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(8 * 8, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

