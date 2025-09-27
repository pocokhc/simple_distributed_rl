import logging
from typing import Tuple, cast

import numpy as np
import torch
import torch.nn as nn

from srl.base.define import SpaceTypes
from srl.base.spaces.box import BoxSpace

logger = logging.getLogger(__name__)


class InputImageReshapeBlock(nn.Module):
    def __init__(self, space: BoxSpace):
        super().__init__()
        self.space = space

        x = np.zeros((1,) + space.shape)
        y = self.forward(torch.tensor(x))
        self.out_shape = cast(Tuple[int, int, int], y.shape[1:])

    def forward(self, x: torch.Tensor):
        err_msg = f"unknown space_type: {self.space}"

        if self.space.stype == SpaceTypes.GRAY_HW:
            if len(self.space.shape) == 2:
                # (batch, h, w) -> (batch, 1, h, w)
                # (batch, h, w, 1) -> (batch, 1, h, w)
                x = x.reshape(x.shape + (1,))
                x = x.permute((0, 3, 1, 2))
            elif len(self.space.shape) == 3:
                # (batch, len, h, w)
                pass
            else:
                raise ValueError(err_msg)

        elif self.space.stype == SpaceTypes.GRAY_HW1:
            assert self.space.shape[-1] == 1
            if len(self.space.shape) == 3:
                # (batch, h, w, 1) -> (batch, 1, h, w)
                x = x.permute((0, 3, 1, 2))
            elif len(self.space.shape) == 4:
                # (batch, len, h, w, 1) -> (batch, len, h, w)
                x = x.reshape(x.shape[:4])
            else:
                raise ValueError(err_msg)

        elif self.space.stype == SpaceTypes.RGB:
            if len(self.space.shape) == 3:
                # (batch, h, w, ch) -> (batch, ch, h, w)
                x = x.permute((0, 3, 1, 2))
            else:
                raise ValueError(err_msg)

        elif self.space.stype == SpaceTypes.FEATURE_MAP:
            if len(self.space.shape) == 3:
                # (batch, h, w, ch) -> (batch, ch, h, w)
                x = x.permute((0, 3, 1, 2))
            else:
                raise ValueError(err_msg)

        else:
            raise ValueError(err_msg)

        return x
