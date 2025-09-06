import torch
import torch.nn as nn

from srl.rl.torch_.distributions.normal_dist_block import NormalDist, NormalDistBlock

from .config import Config


class QNetwork(nn.Module):
    def __init__(self, in_size: int, config: Config, distribution: bool, duel_net: bool):
        super().__init__()
        action_num = config.action_space.n
        units = config.base_units
        self.distribution = distribution
        self.duel_net = duel_net

        self.hidden_block = nn.Sequential(
            nn.Linear(in_size, units),
            nn.SiLU(),
        )

        # --- dueling network
        if duel_net:
            self.value_block = nn.Sequential(
                nn.Linear(units, units),
                nn.SiLU(),
                NormalDistBlock(units, 1) if distribution else nn.Linear(units, 1),
            )
            self.adv_block = nn.Sequential(
                nn.Linear(units, units),
                nn.SiLU(),
                NormalDistBlock(units, action_num) if distribution else nn.Linear(units, action_num),
            )
        else:
            self.out_block = nn.Sequential(
                nn.Linear(units, units),
                nn.SiLU(),
                NormalDistBlock(units, action_num) if distribution else nn.Linear(units, action_num),
            )

    def forward(self, x: torch.Tensor):
        x = self.hidden_block(x)
        if self.duel_net:
            # --- duel(ave)
            if self.distribution:
                v_dist = self.value_block(x)
                adv_dist = self.adv_block(x)
                v = v_dist.rsample()
                adv = adv_dist.rsample()
            else:
                v = self.value_block(x)
                adv = self.adv_block(x)
            x = v + adv - torch.mean(adv, dim=-1, keepdim=True)
        else:
            if self.distribution:
                dist = self.out_block(x)
                x = dist.rsample()
            else:
                x = self.out_block(x)
        return x

    def forward_mean(self, x: torch.Tensor):
        x = self.hidden_block(x)
        if self.duel_net:
            # --- duel(ave)
            if self.distribution:
                v_dist = self.value_block(x)
                adv_dist = self.adv_block(x)
                v = v_dist.mean()
                adv = adv_dist.mean()
            else:
                v = self.value_block(x)
                adv = self.adv_block(x)
            x = v + adv - torch.mean(adv, dim=-1, keepdim=True)
        else:
            if self.distribution:
                dist = self.out_block(x)
                x = dist.mean()
            else:
                x = self.out_block(x)
        return x

    def get_distribution(self, x: torch.Tensor):
        assert self.distribution
        x = self.hidden_block(x)
        v_dist = self.value_block(x)
        adv_dist = self.adv_block(x)
        mu = v_dist.mean() + adv_dist.mean()
        log_sigma = torch.log(torch.sqrt(v_dist.stddev() ** 2 + adv_dist.stddev() ** 2))
        return NormalDist(mu, log_sigma), v_dist, adv_dist
