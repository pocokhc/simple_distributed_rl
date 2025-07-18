import torch
import torch.nn as nn
import torch.nn.functional as F

from srl.rl.torch_.blocks.dueling_network import DuelingNetworkBlock

from .config import Config
from .torch_model_encoder import create_encoder_block


class QNetwork(nn.Module):
    def __init__(self, config: Config, units: int, action_num: int, device):
        super().__init__()

        self.encoder = create_encoder_block(config, device)
        self.hidden_block = nn.Sequential(
            nn.LazyLinear(units),
            nn.SiLU(),
        )
        self.duel_block = DuelingNetworkBlock(in_size=units, hidden_units=units, out_layer_units=action_num)

        # --- out size & init weights
        self.to(device)
        x = torch.zeros((1, *config.observation_space.shape), dtype=config.get_dtype("torch"), device=device)
        oe, x = self(x)
        self.enc_out_size: int = oe.shape[-1]

    def forward(self, x: torch.Tensor):
        oe = self.encoder(x)
        x = self.hidden_block(oe)
        return oe, self.duel_block(x)


class AE(nn.Module):
    def __init__(self, in_size: int, units: int, latent_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_size, units)
        self.fc2 = nn.Linear(units, latent_dim)
        self.fc3 = nn.Linear(latent_dim, units)
        self.fc4 = nn.Linear(units, in_size)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h1 = F.relu(self.fc1(x))
        return self.fc2(h1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        return self.decode(z), z


class RNDNetwork(nn.Module):
    def __init__(self, in_size: int, units: int):
        super().__init__()
        self.h_block = nn.Sequential(
            nn.Linear(in_size, units),
            nn.ReLU(),
            nn.Linear(units, units),
            nn.ReLU(),
            nn.LayerNorm(units),
        )

    def forward(self, x):
        return self.h_block(x)
