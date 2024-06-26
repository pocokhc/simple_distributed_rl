import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from srl.base.define import SpaceTypes
from srl.base.exception import NotSupportedError
from srl.base.spaces.box import BoxSpace
from srl.base.spaces.multi import MultiSpace
from srl.base.spaces.space import SpaceBase
from srl.rl.models.config.image_block import ImageBlockConfig
from srl.rl.models.config.mlp_block import MLPBlockConfig

logger = logging.getLogger(__name__)


def create_in_block_out_multi(
    value_block_config: MLPBlockConfig,
    image_block_config: ImageBlockConfig,
    observation_space: SpaceBase,
):
    if isinstance(observation_space, MultiSpace):
        o = InputMultiBlock(
            value_block_config,
            image_block_config,
            observation_space,
            is_concat=False,
        )
        return o, o.out_types
    assert isinstance(observation_space, BoxSpace)
    if SpaceTypes.is_image(observation_space.stype):
        return InputImageBlock(
            image_block_config,
            observation_space,
            is_flatten=False,
            out_multi=True,
        ), [observation_space.stype]
    else:
        return InputValueBlock(
            value_block_config,
            observation_space,
            out_multi=True,
        ), [observation_space.stype]


def create_in_block_out_value(
    value_block_config: MLPBlockConfig,
    image_block_config: ImageBlockConfig,
    observation_space: SpaceBase,
):
    if isinstance(observation_space, MultiSpace):
        return InputMultiBlock(
            value_block_config,
            image_block_config,
            observation_space,
            is_concat=True,
        )
    assert isinstance(observation_space, BoxSpace)
    if SpaceTypes.is_image(observation_space.stype):
        return InputImageBlock(image_block_config, observation_space)
    else:
        return InputValueBlock(value_block_config, observation_space)


def create_in_block_out_image(
    image_block_config: ImageBlockConfig,
    observation_space: SpaceBase,
):
    if not isinstance(observation_space, BoxSpace):
        raise NotSupportedError(observation_space)
    if SpaceTypes.is_image(observation_space.stype):
        return InputImageBlock(
            image_block_config,
            observation_space,
            is_flatten=False,
        )
    else:
        raise NotSupportedError(observation_space)


# -------------


class InputValueBlock(nn.Module):
    def __init__(
        self,
        value_block_config: MLPBlockConfig,
        observation_space: BoxSpace,
        out_multi: bool = False,
    ):
        super().__init__()
        self.out_multi = out_multi
        self.in_shape = observation_space.shape

        self.in_layers = nn.ModuleList()
        self.in_layers.append(nn.Flatten())
        flat_size = np.zeros(observation_space.shape).flatten().shape[0]
        val_block = value_block_config.create_block_torch(flat_size)
        self.in_layers.append(val_block)
        self.out_size = val_block.out_size

    def forward(self, x: torch.Tensor):
        for h in self.in_layers:
            x = h(x)
        if self.out_multi:
            return [x]
        else:
            return x

    def create_batch_shape(self, prefix_shape: Tuple[Optional[int], ...]):
        return prefix_shape + self.in_shape

    def create_batch_single_data(self, data: np.ndarray, device):
        return torch.tensor(data[np.newaxis, ...]).to(device)

    def create_batch_stack_data(self, data: np.ndarray, device):
        # [batch_list, shape], stackはnpが早い
        return torch.tensor(np.asarray(data)).to(device)


class _InputImageBlock(nn.Module):
    def __init__(self, obs_space: BoxSpace):
        super().__init__()
        self.obs_space = obs_space

        x = np.zeros((1,) + obs_space.shape, dtype=np.float32)
        y = self.forward(torch.tensor(x))
        self.out_shape = y.shape[1:]

    def forward(self, x: torch.Tensor):
        err_msg = f"unknown observation_type: {self.obs_space}"

        if self.obs_space.stype == SpaceTypes.GRAY_2ch:
            if len(self.obs_space.shape) == 2:
                # (batch, h, w) -> (batch, h, w, 1)
                # (batch, h, w, 1) -> (batch, 1, h, w)
                x = x.reshape(x.shape + (1,))
                x = x.permute((0, 3, 1, 2))
            elif len(self.obs_space.shape) == 3:
                # (batch, len, h, w)
                pass
            else:
                raise ValueError(err_msg)

        elif self.obs_space.stype == SpaceTypes.GRAY_3ch:
            assert self.obs_space.shape[-1] == 1
            if len(self.obs_space.shape) == 3:
                # (batch, h, w, 1) -> (batch, 1, h, w)
                x = x.permute((0, 3, 1, 2))
            elif len(self.obs_space.shape) == 4:
                # (batch, len, h, w, 1) -> (batch, len, h, w)
                x = x.reshape(x.shape[:4])
            else:
                raise ValueError(err_msg)

        elif self.obs_space.stype == SpaceTypes.COLOR:
            if len(self.obs_space.shape) == 3:
                # (batch, h, w, ch) -> (batch, ch, h, w)
                x = x.permute((0, 3, 1, 2))
            else:
                raise ValueError(err_msg)

        elif self.obs_space.stype == SpaceTypes.IMAGE:
            if len(self.obs_space.shape) == 3:
                # (batch, h, w, ch) -> (batch, ch, h, w)
                x = x.permute((0, 3, 1, 2))
            else:
                raise ValueError(err_msg)

        else:
            raise ValueError(err_msg)

        return x


class InputImageBlock(nn.Module):
    def __init__(
        self,
        image_block_config: ImageBlockConfig,
        observation_space: BoxSpace,
        is_flatten: bool = True,
        out_multi: bool = False,
    ):
        super().__init__()
        self.out_multi = out_multi
        self.in_shape = observation_space.shape

        self.in_layers = nn.ModuleList()
        in_img_block = _InputImageBlock(observation_space)
        self.in_layers.append(in_img_block)
        img_block = image_block_config.create_block_torch(in_img_block.out_shape)
        self.in_layers.append(img_block)
        if is_flatten:
            self.in_layers.append(nn.Flatten())
            self.out_size = np.zeros(img_block.out_shape).flatten().shape[0]
        else:
            self.out_shape = img_block.out_shape

    def forward(self, x: torch.Tensor):
        for h in self.in_layers:
            x = h(x)
        if self.out_multi:
            return [x]
        else:
            return x

    def create_batch_shape(self, prefix_shape: Tuple[Optional[int], ...]):
        return prefix_shape + self.in_shape

    def create_batch_single_data(self, data: np.ndarray, device):
        return torch.tensor(data[np.newaxis, ...]).to(device)

    def create_batch_stack_data(self, data: np.ndarray, device):
        # [batch_list, shape], stackはnpが早い
        return torch.tensor(np.asarray(data)).to(device)


class InputMultiBlock(nn.Module):
    def __init__(
        self,
        value_block_config: Optional[MLPBlockConfig],
        image_block_config: Optional[ImageBlockConfig],
        observation_space: MultiSpace,
        is_concat: bool = True,
    ):
        super().__init__()
        self.is_concat = is_concat

        self.in_indices = []
        self.in_layers = nn.ModuleList()
        self.in_shapes = []
        self.out_types = []
        if is_concat:
            self.out_size = 0
        else:
            self.out_shapes = []
        for i, space in enumerate(observation_space.spaces):
            if not isinstance(space, BoxSpace):
                continue
            if SpaceTypes.is_image(space.stype):
                if image_block_config is None:
                    logger.info("image space is skip")
                    continue
                b = InputImageBlock(
                    image_block_config,
                    space,
                    is_flatten=is_concat,
                    out_multi=False,
                )
                self.in_indices.append(i)
                self.in_layers.append(b)
                self.in_shapes.append(space.shape)
                self.out_types.append(space.stype)
                if is_concat:
                    self.out_size += b.out_size
                else:
                    self.out_shapes.append(b.out_shape)
            else:
                if value_block_config is None:
                    logger.info("value space is skip")
                    continue
                b = InputValueBlock(
                    value_block_config,
                    space,
                    out_multi=False,
                )
                self.in_indices.append(i)
                self.in_layers.append(b)
                self.in_shapes.append(space.shape)
                self.out_types.append(space.stype)
                if is_concat:
                    self.out_size += b.out_size
                else:
                    self.out_shapes.append(b.out_size)
        assert len(self.in_indices) > 0

    def forward(self, x: torch.Tensor):
        x_arr = []
        i = -1
        for idx in self.in_indices:
            i += 1
            _x = x[idx]
            _x = self.in_layers[i](_x)
            x_arr.append(_x)
        if self.is_concat:
            x_arr = torch.cat(x_arr, dim=-1)
        return x_arr

    def create_batch_shape(self, prefix_shape: Tuple[Optional[int], ...]):
        return [prefix_shape + s for s in self.in_shapes]

    def create_batch_single_data(self, data: List[np.ndarray], device):
        return [torch.tensor(d[np.newaxis, ...]).to(device) for d in data]

    def create_batch_stack_data(self, data: np.ndarray, device):
        # [batch_list, multi_list, shape], stackはnpが早い
        arr = []
        for idx in self.in_indices:
            arr.append(torch.tensor(np.asarray([d[idx] for d in data])).to(device))
        return arr
