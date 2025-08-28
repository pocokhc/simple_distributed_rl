import torch
import torch.nn as nn

from srl.rl.torch_.blocks.dueling_network import DuelingNetworkBlock

from .config import Config
from .torch_model_encoder import create_encoder_block


class QNetwork(nn.Module):
    def __init__(self, config: Config, units: int, action_num: int, device):
        super().__init__()

        self.encoder = create_encoder_block(config)
        self.hidden_block = nn.Sequential(
            nn.LazyLinear(units),
            nn.SiLU(),
        )
        self.duel_block = DuelingNetworkBlock(
            in_size=units,
            hidden_units=units,
            out_layer_units=action_num,
        )

        # --- out size & init weights
        self.to(device)
        x = torch.zeros((1, *config.observation_space.shape), dtype=config.get_dtype("torch"), device=device)
        oe, x = self(x)
        self.enc_out_size: int = oe.shape[-1]

    def forward(self, x: torch.Tensor):
        oe = self.encoder(x)
        x = self.hidden_block(oe)
        x = self.duel_block(x)
        return oe, x

    def forward_q(self, oe: torch.Tensor):
        x = self.hidden_block(oe)
        x = self.duel_block(x)
        return x


class QIntNetwork(nn.Module):
    def __init__(self, in_size: int, units: int, action_num: int):
        super().__init__()

        self.hidden_block = nn.Sequential(
            nn.Linear(in_size, units),
            nn.SiLU(),
        )
        self.duel_block = DuelingNetworkBlock(
            in_size=units,
            hidden_units=units,
            out_layer_units=action_num,
        )

    def forward(self, x: torch.Tensor):
        x = self.hidden_block(x)
        return self.duel_block(x)
