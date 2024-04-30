import logging
from dataclasses import dataclass
from typing import Optional, Tuple, cast

import numpy as np

from srl.base.define import EnvObservationType, SpaceTypes
from srl.base.env.env_run import EnvRun, SpaceBase
from srl.base.rl.config import RLConfig
from srl.base.rl.processor import Processor
from srl.base.rl.worker_run import WorkerRun
from srl.base.spaces.box import BoxSpace
from srl.utils.common import is_package_installed

logger = logging.getLogger(__name__)


@dataclass
class ImageProcessor(Processor):
    image_type: SpaceTypes = SpaceTypes.GRAY_2ch
    resize: Optional[Tuple[int, int]] = None  # (w, h)
    enable_norm: bool = False
    trimming: Optional[Tuple[int, int, int, int]] = None  # (top, left, bottom, right)

    def __post_init__(self):
        self.before_observation_type = SpaceTypes.UNKNOWN
        self.max_val = 0

    def remap_observation_space(self, env_observation_space: SpaceBase, env: EnvRun, rl_config: RLConfig) -> SpaceBase:
        self.before_observation_space = env_observation_space
        self.is_valid = False

        # パッケージがなければ何もしない
        if not is_package_installed("cv2"):
            return env_observation_space

        if not isinstance(env_observation_space, BoxSpace):
            return env_observation_space

        # Imageのみ対象
        assert self.image_type in [
            SpaceTypes.GRAY_2ch,
            SpaceTypes.GRAY_3ch,
            SpaceTypes.COLOR,
        ]
        if env_observation_space.stype not in [
            SpaceTypes.GRAY_2ch,
            SpaceTypes.GRAY_3ch,
            SpaceTypes.COLOR,
        ]:
            return env_observation_space

        # 予測する
        if self.before_observation_type == SpaceTypes.UNKNOWN:
            if len(env_observation_space.shape) == 2:
                self.before_observation_type = SpaceTypes.GRAY_2ch
            elif len(env_observation_space.shape) == 3:
                # w,h,ch 想定
                ch = env_observation_space.shape[-1]
                if ch == 1:
                    self.before_observation_type = SpaceTypes.GRAY_3ch
                elif ch == 3:
                    self.before_observation_type = SpaceTypes.COLOR

        shape = env_observation_space.shape
        low = env_observation_space.low
        high = env_observation_space.high
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
        if self.resize is not None:
            new_shape = self.resize

        # norm
        new_dtype = env_observation_space.dtype
        if "float" in str(env_observation_space.dtype):
            self.enable_norm = False
            logger.info("norm disable")
        self.max_val = np.max(high)
        if self.enable_norm:
            high = 1
            new_dtype = np.float32
        else:
            high = self.max_val
        low = np.min(low)

        # shape
        if self.image_type == SpaceTypes.GRAY_3ch:
            new_shape = new_shape + (1,)
        elif self.image_type == SpaceTypes.COLOR:
            new_shape = new_shape + (3,)

        self.is_valid = True
        new_space = BoxSpace(new_shape, low, high, new_dtype, self.image_type)
        return new_space

    def remap_observation(self, state: EnvObservationType, worker: WorkerRun, env: EnvRun) -> EnvObservationType:
        if not self.is_valid:
            return state
        import cv2

        state = np.asarray(state).astype(np.uint8)

        if self.image_type == SpaceTypes.COLOR and (
            self.before_observation_type == SpaceTypes.GRAY_2ch or self.before_observation_type == SpaceTypes.GRAY_3ch
        ):
            # gray -> color
            state = cv2.applyColorMap(state, cv2.COLORMAP_HOT)
        elif self.before_observation_type == SpaceTypes.COLOR and (
            self.image_type == SpaceTypes.GRAY_2ch or self.image_type == SpaceTypes.GRAY_3ch
        ):
            # color -> gray
            state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)

        if self.trimming is not None:
            state = cast(np.ndarray, state)
            state = state[self.top : self.bottom, self.left : self.right]

        if self.resize is not None:
            state = cv2.resize(state, self.resize)

        state = cast(np.ndarray, state)
        if self.enable_norm:
            state = state.astype(np.float32)
            state /= self.max_val

        if len(state.shape) == 2 and self.image_type == SpaceTypes.GRAY_3ch:
            state = state[..., np.newaxis]

        return state
