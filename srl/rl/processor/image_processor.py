import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import gym
import gym.spaces
import numpy as np
from srl.base.define import EnvObservationType, RLObservationType
from srl.base.rl.processor import Processor

logger = logging.getLogger(__name__)


@dataclass
class ImageProcessor(Processor):

    change_gray: bool = True
    resize: Optional[Tuple[int, int]] = None
    enable_norm: bool = False

    def __post_init__(self):
        self.before_observation_type = EnvObservationType.UNKOWN
        self.max_val = 0

    def change_observation_info(
        self,
        observation_space: gym.spaces.Box,
        observation_type: EnvObservationType,
        rl_observation_type: RLObservationType,
    ) -> Tuple[gym.spaces.Box, EnvObservationType]:
        if observation_type not in [
            EnvObservationType.GRAY_3ch,
            EnvObservationType.COLOR,
        ]:
            return observation_space, observation_type

        assert len(observation_space.shape) == 3
        self.before_observation_type = observation_type
        new_observation_type = EnvObservationType.GRAY_2ch
        if self.resize is None:
            new_shape = (observation_space.shape[0], observation_space.shape[1])
        else:
            new_shape = self.resize
        self.max_val = np.max(observation_space.high)
        if self.enable_norm:
            high = 1
        else:
            high = self.max_val
        new_observation_space = gym.spaces.Box(
            low=np.min(observation_space.low),
            high=high,
            shape=new_shape,
        )

        return new_observation_space, new_observation_type

    def observation_encode(self, observation: np.ndarray) -> np.ndarray:
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


if __name__ == "__main__":
    pass
