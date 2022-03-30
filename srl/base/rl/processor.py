from abc import ABC
from typing import Any, List, Tuple

import gym
import gym.spaces
import numpy as np
from srl.base.define import EnvObservationType, RLActionType, RLObservationType


class Processor(ABC):
    def change_action_info(
        self,
        action_space: gym.spaces.Space,
        rl_action_type: RLActionType,
    ) -> gym.spaces.Space:
        return action_space

    def action_decode(self, action: Any) -> Any:
        return action

    def valid_actions_encode(self, valid_actions: List) -> List:
        return valid_actions

    def change_observation_info(
        self,
        observation_space: gym.spaces.Box,
        observation_type: EnvObservationType,
        rl_observation_type: RLObservationType,
    ) -> Tuple[gym.spaces.Box, EnvObservationType]:
        return observation_space, observation_type

    def observation_encode(self, observation: np.ndarray) -> np.ndarray:
        return observation


if __name__ == "__main__":
    pass
