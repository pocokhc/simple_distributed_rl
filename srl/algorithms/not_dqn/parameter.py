import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from srl.base.rl.parameter import RLParameter
from srl.rl.torch_.helper import model_backup, model_restore

from .config import Config

logger = logging.getLogger(__name__)


class QNetwork(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.in_block = config.input_block.create_torch_block(config)
        self.hidden_block = config.hidden_block.create_torch_block(self.in_block.out_size, config.action_space.n)

    def forward(self, x):
        x = self.in_block(x)
        x = self.hidden_block(x)
        return x


class Parameter(RLParameter[Config]):
    def setup(self):
        super().setup()
        self.device = torch.device(self.config.used_device_torch)
        self.np_dtype = self.config.get_dtype("np")

        self.q_online = QNetwork(self.config).to(self.device)
        self.q_online.train()

    def call_restore(self, dat: dict, from_serialized: bool = False, from_worker: bool = False, **kwargs) -> None:
        model_restore(self.q_online, dat, from_serialized)

    def call_backup(self, serialized: bool = False, to_worker=False, **kwargs) -> Any:
        return model_backup(self.q_online, serialized)

    def summary(self, **kwargs):
        print(self.q_online)

    def pred_q(self, state: np.ndarray) -> np.ndarray:
        state_torch = torch.tensor(np.asarray(state, dtype=self.np_dtype), device=self.device)
        with torch.no_grad():
            self.q_online.eval()
            q = self.q_online(state_torch)
            self.q_online.train()  # 常にtrain
        return q.detach().cpu().numpy()
