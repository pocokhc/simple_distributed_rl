import logging
from dataclasses import dataclass
from typing import Optional, Tuple

from srl.base.env.base import SpaceBase
from srl.base.env.processor import Processor
from srl.base.env.spaces.box import BoxSpace

try:
    import cv2
except ImportError:
    pass

import numpy as np
from srl.base.define import EnvObservation, EnvObservationType, RLObservationType

logger = logging.getLogger(__name__)


@dataclass
class ImageProcessor(Processor):

    gray: bool = True
    resize: Optional[Tuple[int, int]] = None
    enable_norm: bool = False

    def __post_init__(self):
        self.before_observation_type = EnvObservationType.UNKNOWN
        self.max_val = 0

        self.image_types = [
            EnvObservationType.GRAY_2ch,
            EnvObservationType.GRAY_3ch,
            EnvObservationType.COLOR,
        ]

    def change_observation_info(
        self,
        env_observation_space: SpaceBase,
        env_observation_type: EnvObservationType,
        rl_observation_type: RLObservationType,
        original_env: object,
    ) -> Tuple[SpaceBase, EnvObservationType]:
        self.before_observation_type = env_observation_type

        # 画像で BoxSpace のみ対象
        assert self.before_observation_type in self.image_types
        assert isinstance(env_observation_space, BoxSpace)

        shape = env_observation_space.shape
        low = env_observation_space.low
        high = env_observation_space.high

        # resize
        if self.resize is None:
            new_shape = (shape[0], shape[1])
        else:
            new_shape = self.resize

        # norm
        self.max_val = np.max(high)
        if self.enable_norm:
            high = 1
        else:
            high = self.max_val
        low = np.min(low)

        # gray
        new_type = EnvObservationType.GRAY_2ch
        if not self.gray and env_observation_type == EnvObservationType.COLOR:
            new_type = EnvObservationType.COLOR

        new_space = BoxSpace(new_shape, low, high)
        return new_space, new_type

    def process_observation(
        self,
        observation: EnvObservation,
        original_env: object,
    ) -> EnvObservation:
        observation = np.asarray(observation)
        if self.before_observation_type == EnvObservationType.GRAY_3ch:
            # (w,h,1) -> (w,h)
            observation = np.squeeze(observation, -1)
        elif self.before_observation_type == EnvObservationType.COLOR:
            if self.gray:
                # (w,h,ch) -> (w,h)
                observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)

        if self.resize is not None:
            observation = cv2.resize(observation, self.resize)

        if self.enable_norm:
            observation = observation.astype(np.float32)
            observation /= self.max_val

        return observation
