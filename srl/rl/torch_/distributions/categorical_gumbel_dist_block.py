from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from srl.rl.torch_.converter import convert_activation_torch


def gumbel_inverse(u: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Gumbel逆変換サンプリング"""
    return -torch.log(-torch.log(u + eps) + eps)


class CategoricalGumbelDist:
    def __init__(self, logits: torch.Tensor):
        self.classes = logits.shape[-1]
        self.logits = logits

    def mean(self):
        raise NotImplementedError()

    def mode(self):
        return torch.argmax(self.logits, dim=-1)

    def variance(self):
        raise NotImplementedError()

    def probs(self, temperature: float = 1.0, **kwargs):
        return torch.softmax(self.logits / temperature, dim=-1)

    def sample_topk(self, k: int, temperature: float = 1, onehot: bool = False):
        batch_size = self.logits.size(0)
        noise = torch.rand((batch_size, k, self.classes), device=self.logits.device)
        gumbel_noise = gumbel_inverse(noise)

        logits_expanded = self.logits.unsqueeze(1).expand(-1, k, -1)  # (batch, k, num_classes)
        noisy_logits = (logits_expanded + gumbel_noise) / temperature
        topk = torch.argmax(noisy_logits, dim=-1)  # (batch, k)
        if onehot:
            topk = F.one_hot(topk, num_classes=self.classes).float()

        return topk

    def sample(self, temperature: float = 1.0, onehot: bool = False, **kwargs):
        noise = torch.rand_like(self.logits, device=self.logits.device)
        logits_noisy = (self.logits + gumbel_inverse(noise)) / temperature

        # --- argmaxでアクション決定
        act = torch.argmax(logits_noisy, dim=-1)

        if onehot:
            return F.one_hot(act, num_classes=self.classes).float()  # (batch, classes)
        else:
            return act.unsqueeze(-1)  # (batch, 1)

    def rsample(self, temperature: float = 1.0, **kwargs) -> torch.Tensor:
        rnd = torch.rand_like(self.logits, device=self.logits.device)
        logits_noisy = self.logits + gumbel_inverse(rnd)
        return F.softmax(logits_noisy / temperature, dim=-1)

    def log_probs(self, temperature: float = 1, **kwargs):
        probs = self.probs(temperature)
        probs = torch.clamp(probs, 1e-10, 1)  # log(0)回避用
        return torch.log(probs)

    def log_prob(self, a: torch.Tensor, temperature: float = 1, onehot: bool = False, keepdims: bool = True, **kwargs):
        if onehot:
            if a.ndim == 2:  # (batch, 1) の場合 squeeze
                a = a.squeeze(1)
            a = F.one_hot(a, num_classes=self.classes).to(self.logits.dtype)  # (batch, classes)

        # --- log π(a|s) を計算
        log_probs = self.log_probs(temperature)  # (batch, classes)
        log_prob = torch.sum(log_probs * a, dim=-1)  # (batch,)

        if keepdims:
            log_prob = log_prob.unsqueeze(-1)  # (batch, 1)

        return log_prob

    def entropy(self, temperature: float = 1):
        probs = self.probs(temperature=temperature)
        log_probs = torch.log(probs)
        return -torch.sum(probs * log_probs, dim=-1)

    def compute_train_loss(self, y, temperature: float = 1):
        # クロスエントロピーの最小化
        # -Σ p * log(q)
        return -torch.sum(y * self.log_probs(temperature), dim=-1)


class CategoricalGumbelDistBlock(nn.Module):
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
        return CategoricalGumbelDist(x)

    def compute_train_loss(self, x, y, temperature: float = 1):
        dist = self(x)
        return torch.mean(dist.compute_train_loss(y, temperature))
