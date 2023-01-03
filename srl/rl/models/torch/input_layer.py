import logging
from typing import Tuple

import torch
import torch.nn as nn

from srl.base.define import EnvObservationType

logger = logging.getLogger(__name__)


class InputLayer(nn.Module):
    def __init__(
        self,
        observation_shape: Tuple[int, ...],
        observation_type: EnvObservationType,
    ):
        """状態の入力レイヤーを作成して返します

        Args:
            observation_shape (Tuple[int, ...]): 状態の入力shape
            observation_type (EnvObservationType): 状態が何かを表すEnvObservationType

        Returns:
            use_image_head (bool):
                Falseの時 out_layer は flatten、
                Trueの時 out_layer は CNN の形式で返ります。
        """
        super().__init__()
        self.err_msg = f"unknown observation_type: {observation_type}"

        self.observation_shape = observation_shape
        self.observation_type = observation_type
        self.flatten = nn.Flatten()

    def is_image_head(self) -> bool:
        return not (
            self.observation_type == EnvObservationType.DISCRETE
            or self.observation_type == EnvObservationType.CONTINUOUS
            or self.observation_type == EnvObservationType.UNKNOWN
        )

    def forward(self, x: torch.Tensor):
        # --- value head
        if (
            self.observation_type == EnvObservationType.DISCRETE
            or self.observation_type == EnvObservationType.CONTINUOUS
            or self.observation_type == EnvObservationType.UNKNOWN
        ):
            return self.flatten(x)

        # --- image head
        if self.observation_type == EnvObservationType.GRAY_2ch:
            if len(self.observation_shape) == 2:
                # (batch, w, h) -> (batch, w, h, 1)
                x = x.reshape(x.shape + (1,))
            elif len(self.observation_shape) == 3:
                # (batch, len, w, h) -> (batch, w, h, len)
                x = x.permute((0, 2, 3, 1))
            else:
                raise ValueError(self.err_msg)

        elif self.observation_type == EnvObservationType.GRAY_3ch:
            assert self.observation_shape[-1] == 1
            if len(self.observation_shape) == 3:
                # (batch, w, h, 1)
                pass
            elif len(self.observation_shape) == 4:
                # (batch, len, w, h, 1) -> (batch, len, w, h)
                # (batch, len, w, h) -> (batch, w, h, len)
                x = x.reshape(x.shape[:4])
                x = x.permute((0, 2, 3, 1))
            else:
                raise ValueError(self.err_msg)

        elif self.observation_type == EnvObservationType.COLOR:
            if len(self.observation_shape) == 3:
                # (batch, w, h, ch)
                pass
            else:
                raise ValueError(self.err_msg)

        elif self.observation_type == EnvObservationType.SHAPE2:
            if len(self.observation_shape) == 2:
                # (batch, w, h) -> (batch, w, h, 1)
                x = x.reshape(x.shape + (1,))
            elif len(self.observation_shape) == 3:
                # (batch, len, w, h) -> (batch, w, h, len)
                x = x.permute((0, 2, 3, 1))
            else:
                raise ValueError(self.err_msg)

        elif self.observation_type == EnvObservationType.SHAPE3:
            if len(self.observation_shape) == 3:
                # (batch, n, w, h) -> (batch, w, h, n)
                x = x.permute((0, 2, 3, 1))
            else:
                raise ValueError(self.err_msg)

        else:
            raise ValueError(self.err_msg)

        return x
