import torch
import torch.nn as nn
import torch.nn.functional as F


class RunningNorm(nn.Module):
    """観測や報酬のRMS正規化を行う簡易クラス。


    Note:
    - 数値安定性のために分散の下限を設定。
    - 学習時のみ統計を更新し、評価時は固定。
    """

    def __init__(self, eps: float = 1e-6, momentum: float = 0.01) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(1))
        self.register_buffer("var", torch.ones(1))
        self.eps = eps
        self.momentum = momentum
        self.initialized = False

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        if x.numel() == 0:
            return
        m = x.mean()
        v = x.var(unbiased=False)
        if not self.initialized:
            self.mean.copy_(m)  # type: ignore
            self.var.copy_(v.clamp_min(self.eps))  # type: ignore
            self.initialized = True
        else:
            self.mean.lerp_(m, self.momentum)  # type: ignore
            self.var.lerp_(v, self.momentum)  # type: ignore

    def forward(self, x: torch.Tensor, update: bool = True) -> torch.Tensor:
        """入力xを(z-score風に)正規化して返す。


        Args:
        x: 任意shapeのテンソル
        update: Trueなら統計を更新
        """
        if self.training and update:
            with torch.no_grad():
                self.update(x)
        std = (self.var + self.eps).sqrt()  # type: ignore
        return (x - self.mean) / std  # type: ignore


class ProjectorNetwork(nn.Module):
    def __init__(self, units: int, z_size: int, action_num: int):
        super().__init__()
        self.units = units
        self.z_size = z_size

        self.act_block = nn.Embedding(action_num, units)

        # projector で BN を使うと学習が安定し、collapse せずに済む。LayerNorm では collapse する傾向がある
        self.trans_block = nn.Sequential(
            nn.LazyLinear(units),
            nn.BatchNorm1d(units),
            nn.ReLU(),
            nn.Linear(units, z_size),
            nn.BatchNorm1d(z_size),
            nn.ReLU(),
        )
        self.proj_block = nn.Sequential(
            nn.Linear(self.z_size, self.units),
            nn.BatchNorm1d(self.units),
            nn.ReLU(),
            nn.Linear(self.units, self.units),
            nn.BatchNorm1d(self.units),
            nn.ReLU(),
            nn.Linear(self.units, self.units),
            nn.BatchNorm1d(self.units),
        )
        self.pred_block = nn.Sequential(
            nn.Linear(units, units),
            nn.BatchNorm1d(units),
            nn.ReLU(),
            nn.Linear(units, units),
        )
        self.reward_norm = RunningNorm()

    def forward(self, oe: torch.Tensor, action_indices: torch.Tensor) -> torch.Tensor:
        # --- trans
        ae = self.act_block(action_indices)
        z = torch.cat([oe, ae], dim=-1)
        n_oe = self.trans_block(z)

        # projection & pred
        y = self.proj_block(n_oe)
        y = self.pred_block(y)
        return y

    def projection(self, oe: torch.Tensor) -> torch.Tensor:
        return self.proj_block(oe)

    def compute_loss_and_reward(self, y_target: torch.Tensor, y_hat: torch.Tensor):
        loss_vec = self.byol_loss(y_target, y_hat)
        rew_raw = loss_vec.detach()
        rew_norm = self.reward_norm(rew_raw, update=self.training)
        rew = torch.tanh(rew_norm)
        return loss_vec, rew

    @staticmethod
    def byol_loss(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """BYOLで用いられる予測-ターゲット表現間の二乗距離（cos類似正規化）。

        Note:
        - ベクトルを正規化してからL2距離を取る。Why not: 角度ベースの安定性を優先。
        """
        p = F.normalize(p, dim=-1)
        z = F.normalize(z, dim=-1)
        return 2 - 2 * (p * z).sum(dim=-1)  # = ||p - z||^2 / 2 のスケーリング版


class BYOLNetwork(nn.Module):
    def __init__(self, units: int, z_size: int, action_num: int):
        super().__init__()
        self.units = units
        self.z_size = z_size

        self.act_block = nn.Embedding(action_num, units)

        self.trans_block = nn.Sequential(
            nn.LazyLinear(units),
            nn.BatchNorm1d(units),
            nn.SiLU(),
            nn.Linear(units, z_size),
            nn.BatchNorm1d(z_size),
            nn.SiLU(),
        )

        self.proj_block = self.create_projection()
        self.pred_block = nn.Sequential(
            nn.Linear(units, units),
            nn.BatchNorm1d(units),
            nn.SiLU(),
            nn.Linear(units, units),
        )

        self.reward_norm = RunningNorm()

    def create_projection(self):
        return nn.Sequential(
            nn.Linear(self.z_size, self.units),
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
        rew_raw = loss_vec.detach()
        rew_norm = self.reward_norm(rew_raw, update=self.training)
        rew = torch.tanh(rew_norm)
        # rew = (rew_norm * self.cfg.reward_scale).clamp(max=self.cfg.reward_clip)
        return loss_vec, rew

    @staticmethod
    def byol_loss(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """BYOLで用いられる予測-ターゲット表現間の二乗距離（cos類似正規化）。

        Note:
        - ベクトルを正規化してからL2距離を取る。Why not: 角度ベースの安定性を優先。
        """
        p = F.normalize(p, dim=-1)
        z = F.normalize(z, dim=-1)
        return 2 - 2 * (p * z).sum(dim=-1)  # = ||p - z||^2 / 2 のスケーリング版
