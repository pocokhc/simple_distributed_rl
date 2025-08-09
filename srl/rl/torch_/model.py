import torch
import torch.nn as nn


class SequentialModel(nn.Module):
    def __init__(self, layers: list):
        super().__init__()
        self.h_layers = layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for h in self.h_layers:
            x = h(x)
        return x
