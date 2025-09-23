from typing import Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from srl.rl.torch_.converter import convert_activation_torch


class CategoricalDist:
    def __init__(self, logits):
        self.classes = logits.shape[-1]
        self._dist = torch.distributions.Categorical(logits=logits)
        self._logits = logits

    def set_unimix(self, unimix: float):
        if unimix == 0:
            return
        probs = torch.softmax(self._logits, -1)
        uniform = torch.ones_like(probs) / probs.shape[-1]
        probs = (1 - unimix) * probs + unimix * uniform
        self._dist = torch.distributions.Categorical(probs=probs)

    def logits(self):
        return self._dist.logits

    def mean(self):
        return self._dist.mean

    def mode(self):
        return self._dist.mode

    def variance(self):
        return self._dist.variance

    def probs(self):
        return cast(torch.Tensor, self._dist.probs)

    def sample(self, onehot: bool = True):
        a = self._dist.sample()
        if onehot:
            a = torch.nn.functional.one_hot(a, num_classes=self.classes).float()
        else:
            a = torch.unsqueeze(a, -1)
        return a

    def rsample(self):
        a = self._dist.sample()
        a = torch.nn.functional.one_hot(a, num_classes=self.classes).float()
        probs = torch.clamp(self.probs(), 1e-8, 1.0 - 1e-8)
        return (a - probs).detach() + probs

    def log_probs(self):
        return torch.log(torch.clamp(self._dist.probs, 10e-8, 1))

    def log_prob(self, a: torch.Tensor, onehot: bool = False, keepdims: bool = True, **kwargs):
        if onehot:
            if a.ndim == 2:  # (batch, 1) の場合 squeeze
                a = a.squeeze(1)
            a = F.one_hot(a, num_classes=self.classes).to(self._logits.dtype)

        log_probs = self.log_probs()  # (batch, classes)
        log_prob = torch.sum(log_probs * a, dim=-1)  # (batch,)

        if keepdims:
            log_prob = log_prob.unsqueeze(-1)  # (batch, 1)

        return log_prob

    def entropy(self):
        return torch.unsqueeze(self._dist.entropy(), dim=-1)


class CategoricalDistBlock(nn.Module):
    def __init__(
        self,
        in_size: int,
        classes: int,
        hidden_layer_sizes: Tuple[int, ...] = (),
        activation="ReLU",
    ):
        super().__init__()
        self.classes = classes
        activation = convert_activation_torch(activation)

        in_size = in_size
        self.h_layers = nn.ModuleList()
        for size in hidden_layer_sizes:
            self.h_layers.append(nn.Linear(in_size, size))
            self.h_layers.append(activation())
            in_size = size
        self.h_layers.append(nn.Linear(in_size, classes))

    def forward(self, x):
        for layer in self.h_layers:
            x = layer(x)
        return CategoricalDist(x)

    def compute_train_loss(self, x, y, unimix: float = 0):
        dist = self(x)
        dist.set_unimix(unimix)

        # 対数尤度の最大化
        log_likelihood = dist.log_prob(y)
        return -log_likelihood.mean()
