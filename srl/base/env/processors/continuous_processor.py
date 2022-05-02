import logging
from dataclasses import dataclass
from typing import Any, Tuple

import gym
import gym.spaces
import numpy as np
from srl.base.define import EnvActionType, EnvObservationType, RLActionType, RLObservationType
from srl.base.env.base import EnvBase
from srl.base.env.processor import Processor

from .common import tuple_to_box

logger = logging.getLogger(__name__)


@dataclass
class ContinuousProcessor(Processor):
    def __post_init__(self):
        self.action_tbl = None
        self.change_type = ""

    # --- action

    def change_action_info(
        self,
        action_space: gym.spaces.Space,
        action_type: EnvActionType,
        rl_action_type: RLActionType,
        env: EnvBase,
    ) -> Tuple[gym.spaces.Space, EnvActionType]:

        if rl_action_type != RLActionType.CONTINUOUS:
            return action_space, action_type

        if isinstance(action_space, gym.spaces.Discrete):
            new_space = gym.spaces.Box(low=0, high=action_space.n - 1, shape=(1,))
            self.change_type = "Discrete->Box"
            return new_space, EnvActionType.CONTINUOUS

        if isinstance(action_space, gym.spaces.Tuple):
            new_space = tuple_to_box(action_space)
            self.change_type = "Tuple->Box"
            return new_space, EnvActionType.CONTINUOUS

        if isinstance(action_space, gym.spaces.Box):
            return action_space, EnvActionType.CONTINUOUS

        raise ValueError(f"Unimplemented: {action_space.__class__.__name__}")

    def action_decode(self, action: Any, env: EnvBase) -> Any:
        if self.change_type == "":
            return action
        if self.change_type == "Discrete->Box":
            return np.round(action[0])
        if self.change_type == "Tuple->Box":
            return [np.round(a) for a in action]
        raise ValueError()

    # --- observation

    def change_observation_info(
        self,
        observation_space: gym.spaces.Box,
        observation_type: EnvObservationType,
        rl_observation_type: RLObservationType,
        env: EnvBase,
    ) -> Tuple[gym.spaces.Box, EnvObservationType]:
        if rl_observation_type != RLObservationType.CONTINUOUS:
            return observation_space, observation_type

        return observation_space, observation_type

    def observation_encode(self, observation: np.ndarray, env: EnvBase) -> np.ndarray:
        return observation


if __name__ == "__main__":
    pass
