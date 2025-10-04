from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from srl.rl.torch_.distributions.normal_dist_block import NormalDistBlock

from .config import Config


class RunningNorm(nn.Module):
    """観測や報酬のRMS正規化を行う簡易クラス。

    Note:
    - 数値安定性のために分散の下限を設定。
    - 学習時のみ統計を更新し、評価時は固定。
    """

    def __init__(self, eps: float = 1e-10, momentum: float = 0.1) -> None:
        super().__init__()
        self.register_buffer("min", torch.zeros(1))
        self.register_buffer("var", torch.ones(1))
        self.eps = eps
        self.momentum = momentum
        self.initialized = False

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        if x.numel() == 0:
            return
        m = x.min()
        v = (x - m).var(unbiased=False)
        if not self.initialized:
            self.min.copy_(m)  # type: ignore
            self.var.copy_(v.clamp_max(1.0**2))  # type: ignore
            self.initialized = True
        else:
            self.min.lerp_(m, self.momentum)  # type: ignore
            self.var.lerp_(v, self.momentum)  # type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        std = self.var.sqrt()  # type: ignore
        return (x - self.min) / std.clamp_min(self.eps)  # type: ignore

    def get_min(self) -> float:
        return self.min.item()  # type: ignore

    def get_var(self) -> float:
        return self.var.item()  # type: ignore


class BYOLNetwork(nn.Module):
    def __init__(self, in_out_size: int, config: Config):
        super().__init__()
        self.in_out_size = in_out_size
        self.units = config.base_units
        self.int_reward_clip = config.int_reward_clip
        self.int_reward_scale = config.int_reward_byol_scale

        self.act_block = nn.Embedding(config.action_space.n, self.units)

        self.trans_block = nn.Sequential(
            nn.Linear(in_out_size + self.units, self.units),
            nn.BatchNorm1d(self.units),
            nn.SiLU(),
            nn.Linear(self.units, in_out_size),
            nn.BatchNorm1d(in_out_size),
            nn.SiLU(),
        )

        self.proj_block = self.create_projection()
        self.pred_block = nn.Sequential(
            nn.Linear(self.units, self.units),
            nn.BatchNorm1d(self.units),
            nn.SiLU(),
            nn.Linear(self.units, self.units),
        )

        self.reward_norm = RunningNorm(momentum=config.int_norm_momentum)

    def create_projection(self):
        return nn.Sequential(
            nn.Linear(self.in_out_size, self.units),
            nn.BatchNorm1d(self.units),
            nn.SiLU(),
            nn.Linear(self.units, self.units),
            nn.BatchNorm1d(self.units),
            nn.SiLU(),
            nn.Linear(self.units, self.units),
            nn.BatchNorm1d(self.units),
        )

    def forward(self, oe: torch.Tensor, action_indices: torch.Tensor) -> torch.Tensor:
        # --- trans
        ae = self.act_block(action_indices)
        x = torch.cat([oe, ae], dim=-1)
        x = self.trans_block(x)

        # proj
        x = self.proj_block(x)
        return self.pred_block(x)

    def compute_loss_and_reward(self, y_target: torch.Tensor, y_hat: torch.Tensor):
        loss_vec = self.byol_loss(y_target, y_hat)
        rew = loss_vec.detach() * self.int_reward_scale
        if self.training:
            self.reward_norm.update(rew)
        rew = self.reward_norm(rew)
        return loss_vec, torch.clamp(rew, -self.int_reward_clip, self.int_reward_clip)

    @staticmethod
    def byol_loss(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """BYOLで用いられる予測-ターゲット表現間の二乗距離（cos類似正規化）。

        Note:
        - ベクトルを正規化してからL2距離を取る。Why not: 角度ベースの安定性を優先。
        """
        p = F.normalize(p, dim=-1)
        z = F.normalize(z, dim=-1)
        return 2 - 2 * (p * z).sum(dim=-1)  # = ||p - z||^2 / 2 のスケーリング版


class RNDModel(nn.Module):
    def __init__(self, input_dim: int, config: Config) -> None:
        super().__init__()
        self.int_reward_clip = config.int_reward_clip
        self.int_reward_scale = config.int_reward_rnd_scale
        hidden_dim = config.base_units
        momentum = config.int_norm_momentum

        # Target: 固定（学習しない）
        self.target = self._make_model(input_dim, hidden_dim)
        for p in self.target.parameters():
            p.requires_grad = False

        # Predictor: 学習対象
        self.predictor = self._make_model(input_dim, hidden_dim)

        self.error_norm = RunningNorm(momentum=momentum)

    def _make_model(self, input_dim, hidden_dim):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, oe: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        target_out = self.target(oe)
        pred_out = self.predictor(oe)
        return pred_out, target_out

    def compute_intrinsic_reward(self, oe: torch.Tensor, update: bool = False, norm: bool = True) -> torch.Tensor:
        pred_out, target_out = self.forward(oe)
        error = F.l1_loss(pred_out, target_out, reduction="none").mean(dim=1)
        error *= self.int_reward_scale
        if update:
            self.error_norm.update(error)
        if norm:
            error = self.error_norm(error)
            error = torch.clamp(error, -self.int_reward_clip, self.int_reward_clip)
        return error

    def norm(self, error: torch.Tensor):
        return self.error_norm(error)


class QNetwork(nn.Module):
    def __init__(self, in_size: int, config: Config):
        super().__init__()
        action_num = config.action_space.n
        units = config.base_units
        self.distribution = config.enable_q_distribution

        self.hidden_block = nn.Sequential(
            nn.Linear(in_size, units),
            nn.SiLU(),
        )

        # --- dueling network
        self.value_block = nn.Sequential(
            nn.Linear(units, units),
            nn.SiLU(),
            nn.Linear(units, 1),
        )
        self.adv_block = nn.Sequential(
            nn.Linear(units, units),
            nn.SiLU(),
            NormalDistBlock(units, action_num) if self.distribution else nn.Linear(units, action_num),
        )

    def forward(self, x: torch.Tensor):
        x = self.hidden_block(x)
        v = self.value_block(x)
        if self.distribution:
            adv_dist = self.adv_block(x)
            adv = adv_dist.rsample()
        else:
            adv = self.adv_block(x)
        x = v + adv - torch.mean(adv, dim=-1, keepdim=True)
        return x, v

    def forward_mean(self, x: torch.Tensor):
        x = self.hidden_block(x)
        v = self.value_block(x)
        if self.distribution:
            adv_dist = self.adv_block(x)
            adv = adv_dist.mean()
        else:
            adv = self.adv_block(x)
        x = v + adv - torch.mean(adv, dim=-1, keepdim=True)
        return x, v

    def get_distribution(self, x: torch.Tensor):
        x = self.hidden_block(x)
        adv_dist = self.adv_block(x)
        return adv_dist


class EmbeddingNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, action_num: int):
        super().__init__()

        self.in_block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.out_block = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_num),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        x1 = self.in_block(x1)
        x2 = self.in_block(x2)
        x = torch.cat([x1, x2], dim=1)
        return self.out_block(x)

    def predict(self, x: torch.Tensor):
        return self.in_block(x)
