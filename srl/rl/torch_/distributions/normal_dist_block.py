import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from srl.rl.torch_.converter import convert_activation_torch


def compute_normal_logprob(x, loc, log_scale):
    """
    log π(a|s) when the policy is normally distributed
    https://ja.wolframalpha.com/input?i2d=true&i=Log%5BDivide%5B1%2C+%5C%2840%29Sqrt%5B2+*+Pi+*+Power%5B%CF%83%2C2%5D%5D%5C%2841%29%5D+*+Exp%5B-+Divide%5BPower%5B%5C%2840%29x+-+%CE%BC%5C%2841%29%2C2%5D%2C+2+*+Power%5B%CF%83%2C2%5D%5D%5D%5D
    -0.5 * log(2*pi) - log(stddev) - 0.5 * ((x - mean) / stddev)^2
    """
    return -0.5 * math.log(2 * math.pi) - log_scale - 0.5 * (((x - loc) / torch.exp(log_scale)) ** 2)


def compute_normal_logprob_sgp(x, loc, log_scale, epsilon: float = 1e-10):
    """
    Squashed Gaussian Policy log π(a|s)
    Paper: https://arxiv.org/abs/1801.01290
    """
    # xはsquashed前の値
    logmu = compute_normal_logprob(x, loc, log_scale)
    x = 1.0 - torch.tanh(x) ** 2
    x = torch.clamp(x, epsilon, 1.0)
    return logmu - torch.sum(torch.log(x), dim=-1, keepdim=True)


class NormalDist:
    def __init__(self, loc: torch.Tensor, log_scale: torch.Tensor):
        self._loc = loc
        self._log_scale = log_scale

    def mean(self):
        return self._loc

    def mode(self):
        return self._loc

    def stddev(self):
        return torch.exp(self._log_scale)

    def variance(self):
        return self.stddev() ** 2

    def sample(self):
        return torch.normal(self._loc, torch.exp(self._log_scale))

    def rsample(self):
        e = torch.normal(torch.zeros_like(self._loc), torch.ones_like(self._loc))
        return self._loc + torch.exp(self._log_scale) * e

    def log_prob(self, x):
        return compute_normal_logprob(x, self._loc, self._log_scale)

    def log_prob_sgp(self, x):
        return compute_normal_logprob_sgp(x, self._loc, self._log_scale)

    def entropy(self):
        return 0.5 + 0.5 * math.log(2 * math.pi) + self._log_scale


class NormalDistBlock(nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        hidden_layer_sizes: Tuple[int, ...] = (),
        loc_layer_sizes: Tuple[int, ...] = (),
        scale_layer_sizes: Tuple[int, ...] = (),
        activation="ReLU",
        fixed_scale: float = -1,
        enable_stable_gradients: bool = True,
        stable_gradients_scale_range: tuple = (1e-10, 10),
    ):
        super().__init__()
        self.out_size = out_size
        self.enable_stable_gradients = enable_stable_gradients
        activation = convert_activation_torch(activation)

        if enable_stable_gradients:
            self.stable_gradients_scale_range = (
                np.log(stable_gradients_scale_range[0]),
                np.log(stable_gradients_scale_range[1]),
            )

        in_size = in_size
        self.h_layers = nn.ModuleList()
        for size in hidden_layer_sizes:
            self.h_layers.append(nn.Linear(in_size, size))
            self.h_layers.append(activation())
            in_size = size
        h_out_size = in_size

        # --- loc
        in_size = h_out_size
        self.loc_layers = nn.ModuleList()
        for size in loc_layer_sizes:
            self.loc_layers.append(nn.Linear(in_size, size))
            self.loc_layers.append(activation())
            in_size = size
        self.loc_layers.append(nn.Linear(in_size, out_size))
        in_size = out_size

        # --- scale
        if fixed_scale > 0:
            self.fixed_log_scale = np.log(fixed_scale)
        else:
            self.fixed_log_scale = None
            in_size = h_out_size
            self.log_scale_layers = nn.ModuleList()
            for size in scale_layer_sizes:
                self.log_scale_layers.append(nn.Linear(in_size, size))
                self.log_scale_layers.append(activation())
                in_size = size
            self.log_scale_layers.append(nn.Linear(in_size, out_size))

    def forward(self, x):
        for layer in self.h_layers:
            x = layer(x)

        # --- loc
        loc = x
        for layer in self.loc_layers:
            loc = layer(loc)

        # --- scale
        if self.fixed_log_scale is not None:
            log_scale = torch.ones_like(loc) * self.fixed_log_scale
        else:
            log_scale = x
            for layer in self.log_scale_layers:
                log_scale = layer(log_scale)

        if self.enable_stable_gradients:
            log_scale = torch.clamp(
                log_scale,
                self.stable_gradients_scale_range[0],
                self.stable_gradients_scale_range[1],
            )

        return NormalDist(loc, log_scale)

    def compute_train_loss(self, x, y):
        dist = self(x)

        # 対数尤度の最大化
        log_likelihood = dist.log_prob(y)
        return -log_likelihood.mean()
