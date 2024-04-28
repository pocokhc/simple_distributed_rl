import functools
import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from srl.rl.torch_.converter import convert_activation_torch


class CategoricalDist:
    def __init__(self, logits, unimix: float = 0):
        self.classes = logits.shape[-1]
        self.logits = logits
        self.unimix = unimix

    @functools.lru_cache
    def mean(self):
        return torch.full(
            self.logits.shape(),
            torch.nan,
            dtype=self.logits.dtype,
            device=self.logits.device,
        )

    @functools.lru_cache
    def mode(self):
        return torch.argmax(self.logits, -1)

    @functools.lru_cache
    def variance(self):
        return torch.full(
            self.logits.shape(),
            torch.nan,
            dtype=self.logits.dtype,
            device=self.logits.device,
        )

    @functools.lru_cache
    def probs(self):
        probs = torch.softmax(self.logits, -1)
        if self.unimix > 0:
            uniform = torch.ones_like(probs) / probs.shape[-1]
            probs = (1 - self.unimix) * probs + self.unimix * uniform
        return probs

    def sample(self, onehot: bool = False):
        a = torch.multinomial(self.probs(), num_samples=1)
        if onehot:
            a = torch.squeeze(a, dim=1)
            a = torch.nn.functional.one_hot(a, num_classes=self.classes).float()
        return a

    def rsample(self, onehot: bool = True):
        a = torch.multinomial(self.probs(), num_samples=1)
        if onehot:
            a = torch.squeeze(a, dim=1)
            a = torch.nn.functional.one_hot(a, num_classes=self.classes).float()
        return (a - self.probs()).detach() + self.probs()

    @functools.lru_cache
    def log_probs(self):
        probs = torch.clamp(self.probs(), 1e-10, 1)
        return torch.log(probs)

    def log_prob(self, a, onehot: bool = False):
        if onehot:
            a = torch.squeeze(a, dim=-1)
            a = torch.nn.functional.one_hot(a, self.classes).float()
        a = (self.log_probs() * a).sum(dim=-1)
        return torch.unsqueeze(a, axis=-1)


class CategoricalDistBlock(nn.Module):
    def __init__(
        self,
        in_size: int,
        classes: int,
        hidden_layer_sizes: Tuple[int, ...] = (),
        activation="ReLU",
        unimix: float = 0,
    ):
        super().__init__()
        self.classes = classes
        self.unimix = unimix
        activation = convert_activation_torch(activation)

        in_size = in_size
        self.h_layers = nn.ModuleList()
        for size in hidden_layer_sizes:
            self.h_layers.append(nn.Linear(in_size, size))
            self.h_layers.append(activation(inplace=True))
            in_size = size
        self.h_layers.append(nn.Linear(in_size, classes))

    def forward(self, x):
        for layer in self.h_layers:
            x = layer(x)
        return CategoricalDist(x, self.unimix)

    def compute_train_loss(self, x, y):
        dist = self(x)

        # 対数尤度の最大化
        log_likelihood = dist.log_prob(y)
        return -log_likelihood.mean()
