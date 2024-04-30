from typing import Tuple, cast

import torch
import torch.nn as nn

from srl.rl.torch_.converter import convert_activation_torch


class CategoricalDist:
    def __init__(self, logits):
        self.classes = logits.shape[-1]
        self._dist = torch.distributions.Categorical(logits=logits)
        self._logits = logits

    def set_unimix(self, unimix: float):
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

    def sample(self, onehot: bool = False):
        a = self._dist.sample()
        if onehot:
            a = torch.nn.functional.one_hot(a, num_classes=self.classes).float()
        else:
            a = torch.unsqueeze(a, -1)
        return a

    def rsample(self, onehot: bool = True):
        a = self._dist.sample()
        if onehot:
            a = torch.nn.functional.one_hot(a, num_classes=self.classes).float()
        else:
            a = torch.unsqueeze(a, -1)
        probs = self.probs()
        return (a - probs).detach() + probs

    def log_probs(self):
        return self._dist.log_prob

    def log_prob(self, a, onehot: bool = False):
        if onehot:
            a = torch.squeeze(a, dim=-1)
            a = torch.nn.functional.one_hot(a, self.classes).float()
        return self._dist.log_prob(a)

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
            self.h_layers.append(activation(inplace=True))
            in_size = size
        self.h_layers.append(nn.Linear(in_size, classes))

    def forward(self, x):
        for layer in self.h_layers:
            x = layer(x)
        return CategoricalDist(x)

    def compute_train_loss(self, x, y):
        dist = self(x)

        # 対数尤度の最大化
        log_likelihood = dist.log_prob(y)
        return -log_likelihood.mean()
