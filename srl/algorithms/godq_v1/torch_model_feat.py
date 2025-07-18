import torch
import torch.nn as nn


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
        self.project_block = nn.Sequential(
            nn.Linear(self.z_size, self.units),
            nn.BatchNorm1d(self.units),
            nn.ReLU(),
            nn.Linear(self.units, self.units),
            nn.BatchNorm1d(self.units),
            nn.ReLU(),
            nn.Linear(self.units, self.units),
            nn.BatchNorm1d(self.units),
        )
        self.project_head_block = nn.Sequential(
            nn.Linear(units, units),
            nn.BatchNorm1d(units),
            nn.ReLU(),
            nn.Linear(units, units),
        )

    def trans(self, z: torch.Tensor, action_indices: torch.Tensor) -> torch.Tensor:
        ae = self.act_block(action_indices)
        z = torch.cat([z, ae], dim=-1)
        return self.trans_block(z)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        return self.project_block(z)

    def projection_and_head(self, z: torch.Tensor) -> torch.Tensor:
        z = self.project_block(z)
        return self.project_head_block(z)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        z = self.trans(x, a)
        z = self.projection_and_head(z)
        return z


class SPRNetwork(nn.Module):
    def __init__(self, units: int, z_size: int, action_num: int):
        super().__init__()
        self.units = units
        self.z_size = z_size

        self.act_block = nn.Embedding(action_num, units)

        self.trans_block = nn.Sequential(
            nn.LazyLinear(units),
            nn.ReLU(),
            nn.Linear(units, z_size),
        )
        self.project_block = self.create_projection()
        self.project_head_block = nn.Sequential(
            nn.Linear(units, units),
            nn.BatchNorm1d(units),
            nn.ReLU(),
            nn.Linear(units, units),
        )

    def create_projection(self):
        return nn.Sequential(
            nn.Linear(self.z_size, self.units),
            nn.BatchNorm1d(self.units),
            nn.ReLU(),
            nn.Linear(self.units, self.units),
            nn.BatchNorm1d(self.units),
        )

    def trans(self, oe: torch.Tensor, action_indices: torch.Tensor) -> torch.Tensor:
        ae = self.act_block(action_indices)
        x = torch.cat([oe, ae], dim=-1)
        return self.trans_block(x)

    def projection_and_head(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_block(x)
        return self.project_head_block(x)

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        z = self.trans(z, a)
        z = self.projection_and_head(z)
        return z
