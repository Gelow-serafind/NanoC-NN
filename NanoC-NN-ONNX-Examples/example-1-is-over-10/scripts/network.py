from __future__ import annotations

import torch
from torch import nn


class IsOver10Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)
