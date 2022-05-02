from abc import ABC
from typing import Any, List, Tuple

import gym
import gym.spaces
import numpy as np
from srl.base.define import EnvActionType, EnvObservationType, RLActionType, RLObservationType
from srl.base.env.base import EnvBase


class Processor(ABC):
    def change_action_info(
        self,
        action_space: gym.spaces.Space,
        action_type: EnvActionType,
        rl_action_type: RLActionType,
        env: EnvBase,
    ) -> Tuple[gym.spaces.Space, EnvActionType]:
        return action_space, action_type

    def action_decode(self, action: Any, env: EnvBase) -> Any:
        return action

    def invalid_actions_encode(self, invalid_actions: List[int], env: EnvBase) -> List:
        return invalid_actions

    def change_observation_info(
        self,
        observation_space: gym.spaces.Box,
        observation_type: EnvObservationType,
        rl_observation_type: RLObservationType,
        env: EnvBase,
    ) -> Tuple[gym.spaces.Box, EnvObservationType]:
        return observation_space, observation_type

    def observation_encode(self, observation: np.ndarray, env: EnvBase) -> np.ndarray:
        return observation


if __name__ == "__main__":
    pass
