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
from srl.base.define import EnvObservationType, RLObservationType

logger = logging.getLogger(__name__)


@dataclass
class ImageProcessor(Processor):

    gray: bool = True
    resize: Optional[Tuple[int, int]] = None
    enable_norm: bool = False

    def __post_init__(self):
        self.before_observation_type = EnvObservationType.UNKNOWN
        self.max_val = 0

    def change_observation_info(
        self,
        env_observation_space: SpaceBase,
        env_observation_type: EnvObservationType,
        rl_observation_type: RLObservationType,
        original_env: object,
    ) -> Tuple[SpaceBase, EnvObservationType]:

        if env_observation_type not in [
            EnvObservationType.GRAY_3ch,
            EnvObservationType.COLOR,
        ]:
            return env_observation_space, env_observation_type

        if not isinstance(env_observation_space, BoxSpace):
            return env_observation_space, env_observation_type
        shape = env_observation_space.shape
        low = env_observation_space.low
        high = env_observation_space.high

        assert len(shape) == 3
        self.before_observation_type = env_observation_type
        new_observation_type = EnvObservationType.GRAY_2ch
        if self.resize is None:
            new_shape = (shape[0], shape[1])
        else:
            new_shape = self.resize
        self.max_val = np.max(high)
        if self.enable_norm:
            high = 1
        else:
            high = self.max_val
        low = np.min(low)
        new_observation_space = BoxSpace(low, high, new_shape)

        return new_observation_space, new_observation_type

    def process_observation(
        self,
        observation: np.ndarray,
        original_env: object,
    ) -> np.ndarray:
        if self.before_observation_type == EnvObservationType.GRAY_3ch:
            # (w,h,1) -> (w,h)
            observation = np.squeeze(observation, -1)
        elif self.before_observation_type == EnvObservationType.COLOR:
            # (w,h,ch) -> (w,h)
            observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)

        if self.before_observation_type in [
            EnvObservationType.GRAY_2ch,
            EnvObservationType.GRAY_3ch,
            EnvObservationType.COLOR,
        ]:
            if self.resize is not None:
                observation = cv2.resize(observation, self.resize)

            if self.enable_norm:
                observation = observation.astype(np.float32)
                observation /= self.max_val

        return observation
