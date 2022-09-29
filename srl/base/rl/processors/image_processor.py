import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from srl.base.define import EnvObservation, EnvObservationType, RLObservationType
from srl.base.env.base import EnvRun, SpaceBase
from srl.base.env.spaces.box import BoxSpace
from srl.base.rl.processor import Processor
from srl.utils.common import is_package_installed

try:
    import cv2
except ImportError:
    pass

logger = logging.getLogger(__name__)


@dataclass
class ImageProcessor(Processor):

    image_type: EnvObservationType = EnvObservationType.GRAY_2ch
    resize: Optional[Tuple[int, int]] = None
    enable_norm: bool = False
    trimming: Optional[Tuple[int, int, int, int]] = None  # (top, left, bottom, right)

    def __post_init__(self):
        self.before_observation_type = EnvObservationType.UNKNOWN
        self.max_val = 0

        assert is_package_installed(
            "cv2"
        ), "To use ImageProcessor you need to install the 'cv2'. (pip install opencv-python)"
        assert EnvObservationType.is_image(self.image_type)

    def change_observation_info(
        self,
        env_observation_space: SpaceBase,
        env_observation_type: EnvObservationType,
        rl_observation_type: RLObservationType,
        env: EnvRun,
    ) -> Tuple[SpaceBase, EnvObservationType]:
        self.before_observation_type = env_observation_type
        self.is_valid = False

        # BoxSpace のみ対象
        if not isinstance(env_observation_space, BoxSpace):
            return env_observation_space, env_observation_type

        # 予測する
        if self.before_observation_type == EnvObservationType.UNKNOWN:
            if len(env_observation_space.shape) == 2:
                self.before_observation_type = EnvObservationType.GRAY_2ch
            elif len(env_observation_space.shape) == 3:
                # w,h,ch 想定
                ch = env_observation_space.shape[-1]
                if ch == 1:
                    self.before_observation_type = EnvObservationType.GRAY_3ch
                elif ch == 3:
                    self.before_observation_type = EnvObservationType.COLOR

        # 画像のみ対象
        if not EnvObservationType.is_image(self.before_observation_type):
            return env_observation_space, env_observation_type

        shape = env_observation_space.shape
        low = env_observation_space.low
        high = env_observation_space.high
        new_shape = (shape[0], shape[1])

        # trimming
        if self.trimming is not None:
            self.top = self.trimming[0]
            self.left = self.trimming[1]
            self.bottom = self.trimming[2]
            self.right = self.trimming[3]
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
        self.max_val = np.max(high)
        if self.enable_norm:
            high = 1
        else:
            high = self.max_val
        low = np.min(low)

        # type
        new_type = self.image_type

        # shape
        if self.image_type == EnvObservationType.GRAY_3ch:
            new_shape = new_shape + (1,)
        elif self.image_type == EnvObservationType.COLOR:
            new_shape = new_shape + (3,)

        self.is_valid = True
        new_space = BoxSpace(new_shape, low, high)
        return new_space, new_type

    def process_observation(
        self,
        observation: EnvObservation,
        env: EnvRun,
    ) -> EnvObservation:
        if not self.is_valid:
            return observation
        observation = np.asarray(observation).astype(np.uint8)

        if self.image_type == EnvObservationType.COLOR and (
            self.before_observation_type == EnvObservationType.GRAY_2ch
            or self.before_observation_type == EnvObservationType.GRAY_3ch
        ):
            # gray -> color
            observation = cv2.applyColorMap(observation, cv2.COLORMAP_HOT)
        elif self.before_observation_type == EnvObservationType.COLOR and (
            self.image_type == EnvObservationType.GRAY_2ch or self.image_type == EnvObservationType.GRAY_3ch
        ):
            # color -> gray
            observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)

        if self.trimming is not None:
            observation = observation[self.top : self.bottom, self.left : self.right]

        if self.resize is not None:
            observation = cv2.resize(observation, self.resize)

        if self.enable_norm:
            observation = observation.astype(np.float32)
            observation /= self.max_val

        if len(observation.shape) == 2 and self.image_type == EnvObservationType.GRAY_3ch:
            observation = observation[..., np.newaxis]

        return observation
