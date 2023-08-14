import logging
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from srl.base.define import EnvObservationTypes

logger = logging.getLogger(__name__)


class InputBlock(nn.Module):
    def __init__(
        self,
        observation_shape: Tuple[int, ...],
        observation_type: EnvObservationTypes,
        enable_time_distributed_layer: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.observation_type = observation_type
        self.observation_shape = observation_shape
        self.enable_time_distributed_layer = enable_time_distributed_layer

        self.error_msg = f"unknown observation_type: {self.observation_type}"

        self.flatten = nn.Flatten()

        self.use_image_layer = not (
            self.observation_type == EnvObservationTypes.DISCRETE
            or self.observation_type == EnvObservationTypes.CONTINUOUS
            or self.observation_type == EnvObservationTypes.UNKNOWN
        )

        # --- out shape
        if enable_time_distributed_layer:
            x = np.ones((1, 1) + observation_shape, dtype=np.float32)
            y = self.forward(torch.tensor(x))
            self.out_shape = y.shape[2:]
        else:
            x = np.ones((1,) + observation_shape, dtype=np.float32)
            y = self.forward(torch.tensor(x))
            self.out_shape = y.shape[1:]

    def forward(self, x: torch.Tensor):
        batch_size = 0
        seq = 0
        if self.enable_time_distributed_layer:
            # (batch, seq, shape) -> (batch*seq, shape)
            size = x.size()
            batch_size = size[0]
            seq = size[1]
            shape = size[2:]
            x = x.reshape((batch_size * seq, *shape))

        if not self.use_image_layer:
            # --- value head
            x = self.flatten(x)
        elif self.observation_type == EnvObservationTypes.GRAY_2ch:
            # --- image head

            if len(self.observation_shape) == 2:
                # (batch, h, w) -> (batch, h, w, 1)
                # (batch, h, w, 1) -> (batch, 1, h, w)
                x = x.reshape(x.shape + (1,))
                x = x.permute((0, 3, 1, 2))
            elif len(self.observation_shape) == 3:
                # (batch, len, h, w)
                pass
            else:
                raise ValueError(self.error_msg)

        elif self.observation_type == EnvObservationTypes.GRAY_3ch:
            assert self.observation_shape[-1] == 1
            if len(self.observation_shape) == 3:
                # (batch, h, w, 1) -> (batch, 1, h, w)
                x = x.permute((0, 3, 1, 2))
            elif len(self.observation_shape) == 4:
                # (batch, len, h, w, 1) -> (batch, len, h, w)
                x = x.reshape(x.shape[:4])
            else:
                raise ValueError(self.error_msg)

        elif self.observation_type == EnvObservationTypes.COLOR:
            if len(self.observation_shape) == 3:
                # (batch, h, w, ch) -> (batch, ch, h, w)
                x = x.permute((0, 3, 1, 2))
            else:
                raise ValueError(self.error_msg)

        elif self.observation_type == EnvObservationTypes.SHAPE2:
            if len(self.observation_shape) == 2:
                # (batch, h, w) -> (batch, h, w, 1)
                # (batch, h, w, 1) -> (batch, 1, h, w)
                x = x.reshape(x.shape + (1,))
                x = x.permute((0, 3, 1, 2))
            elif len(self.observation_shape) == 3:
                # (batch, len, h, w)
                pass
            else:
                raise ValueError(self.error_msg)

        elif self.observation_type == EnvObservationTypes.SHAPE3:
            if len(self.observation_shape) == 3:
                # (batch, n, h, w)
                pass
            else:
                raise ValueError(self.error_msg)

        else:
            raise ValueError(self.error_msg)

        if self.enable_time_distributed_layer:
            # (batch*seq, shape) -> (batch, seq, shape)
            size = x.size()
            shape = size[1:]
            x = x.view(batch_size, seq, *shape)

        return x


if __name__ == "__main__":
    m = InputBlock((1, 2, 3), EnvObservationTypes.COLOR)
    print(m)
