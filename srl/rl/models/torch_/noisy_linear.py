import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        sigma: float = 0.5,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma = sigma

        self.w_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.w_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.b_mu = nn.Parameter(torch.Tensor(out_features))
        self.b_sigma = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.w_mu.size(1))
        self.w_mu.data.uniform_(-stdv, stdv)
        self.b_mu.data.uniform_(-stdv, stdv)

        initial_sigma = self.sigma * stdv
        self.w_sigma.data.fill_(initial_sigma)
        self.b_sigma.data.fill_(initial_sigma)

    def forward(self, x):
        w_noise = torch.randn(
            self.w_mu.size(),
            dtype=self.w_mu.dtype,
            layout=self.w_mu.layout,
            device=self.b_mu.device,
        )
        b_noise = torch.randn(
            self.b_mu.size(),
            dtype=self.b_mu.dtype,
            layout=self.b_mu.layout,
            device=self.b_mu.device,
        )

        w = self.w_mu + self.w_sigma * w_noise
        b = self.b_mu + self.b_sigma * b_noise

        return F.linear(x, w, b)
