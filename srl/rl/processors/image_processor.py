import logging
from dataclasses import dataclass
from typing import Literal, Optional, Tuple, cast

import numpy as np

from srl.base.define import EnvObservationType, SpaceTypes
from srl.base.env.processor import EnvProcessor
from srl.base.rl.processor import RLProcessor
from srl.base.spaces.box import BoxSpace
from srl.base.spaces.space import SpaceBase
from srl.utils.common import is_package_installed

logger = logging.getLogger(__name__)


@dataclass
class ImageProcessor(RLProcessor, EnvProcessor):
    """
    EnvProcessorとしても使える
    """

    image_type: SpaceTypes = SpaceTypes.GRAY_2ch
    resize: Optional[Tuple[int, int]] = None  # (w, h)
    normalize_type: Literal["", "0to1", "-1to1"] = ""
    trimming: Optional[Tuple[int, int, int, int]] = None  # (top, left, bottom, right)

    def remap_observation_space(self, prev_space: SpaceBase, **kwargs) -> Optional[SpaceBase]:
        # パッケージがなければ何もしない
        if not is_package_installed("cv2"):
            return None

        if not isinstance(prev_space, BoxSpace):
            return None

        # Imageのみ対象
        assert self.image_type in [
            SpaceTypes.GRAY_2ch,
            SpaceTypes.GRAY_3ch,
            SpaceTypes.COLOR,
        ]
        if prev_space.stype not in [
            SpaceTypes.GRAY_2ch,
            SpaceTypes.GRAY_3ch,
            SpaceTypes.COLOR,
        ]:
            return None

        shape = prev_space.shape
        low = prev_space.low
        high = prev_space.high
        new_shape = (shape[0], shape[1])

        # trimming
        if self.trimming is not None:
            self.top, self.left, self.bottom, self.right = self.trimming
            assert self.top < self.bottom
            assert self.left < self.right
            w = shape[1]
            h = shape[0]
            if self.top < 0:
                self.top = 0
            if self.left < 0:
                self.left = 0
            if self.bottom > h:
                self.bottom = h
            if self.right > w:
                self.right = w
            new_shape = (self.right - self.left, self.bottom - self.top)

        # resize
        if self.resize is not None and new_shape != self.resize:
            new_shape = self.resize

        # norm
        new_dtype = prev_space.dtype
        if "float" in str(prev_space.dtype):
            self.normalize_type = ""
            logger.info("normalize disable")
        self.max_val = np.max(high)
        self.min_val = np.min(low)
        if self.normalize_type == "0to1":
            low = 0
            high = 1
            new_dtype = np.float32
        elif self.normalize_type == "-1to1":
            low = -1
            high = 1
            new_dtype = np.float32
        else:
            low = self.min_val
            high = self.max_val

        # shape
        if self.image_type == SpaceTypes.GRAY_3ch:
            new_shape = new_shape + (1,)
        elif self.image_type == SpaceTypes.COLOR:
            new_shape = new_shape + (3,)

        new_space = BoxSpace(new_shape, low, high, new_dtype, self.image_type)
        return new_space

    def remap_observation(self, state: EnvObservationType, prev_space: SpaceBase, new_space: SpaceBase, **kwargs) -> EnvObservationType:
        import cv2

        state = cast(np.ndarray, state)
        prev_stype = prev_space.stype

        if self.image_type == SpaceTypes.COLOR and (prev_stype == SpaceTypes.GRAY_2ch or prev_stype == SpaceTypes.GRAY_3ch):
            # gray -> color
            if prev_stype == SpaceTypes.GRAY_2ch:
                state = state[..., np.newaxis]
            state = np.tile(state, (1, 1, 3))

            # if "float" in str(state.dtype):
            #    if prev_stype == SpaceTypes.GRAY_2ch:
            #        state = state[..., np.newaxis]
            #    state = np.tile(state, (1, 1, 3))
            # else:
            #    state = np.asarray(state).astype(np.uint8)
            #    state = cv2.applyColorMap(state, cv2.COLORMAP_HOT)
        elif prev_stype == SpaceTypes.COLOR and (self.image_type == SpaceTypes.GRAY_2ch or self.image_type == SpaceTypes.GRAY_3ch):
            # color -> gray
            if "float" in str(state.dtype):
                pass
            else:
                state = np.asarray(state).astype(np.uint8)
                state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)

        if self.trimming is not None:
            state = state[self.top : self.bottom, self.left : self.right]

        if self.resize is not None:
            if "float" in str(state.dtype):
                pass  # warning normされたものはresizeできない
            else:
                state = cv2.resize(state, self.resize)
            assert state.shape[0] == self.resize[0] and state.shape[1] == self.resize[1]

        if self.normalize_type == "0to1":
            state = state.astype(np.float32)
            state /= self.max_val
        elif self.normalize_type == "-1to1":
            state = state.astype(np.float32)
            state = (state / self.max_val) * 2 - 1

        state = cast(np.ndarray, state)
        if len(state.shape) == 2 and self.image_type == SpaceTypes.GRAY_3ch:
            state = state[..., np.newaxis]

        return state
