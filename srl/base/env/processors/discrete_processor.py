import itertools
import logging
from dataclasses import dataclass
from typing import Any, List, Tuple

import gym
import gym.spaces
import numpy as np
from srl.base.define import EnvActionType, EnvObservationType, RLActionType, RLObservationType
from srl.base.env.base import EnvBase
from srl.base.env.processor import Processor

logger = logging.getLogger(__name__)


@dataclass
class DiscreteProcessor(Processor):

    action_division_num: int = 5
    observation_division_num: int = 50

    def __post_init__(self):
        # action
        self.action_tbl = {}
        self.change_type = ""

        # observation
        self._observation_discrete_diff = None
        self.low = None

    # --- action

    def change_action_info(
        self,
        action_space: gym.spaces.Space,
        action_type: EnvActionType,
        rl_action_type: RLActionType,
        env: EnvBase,
    ) -> Tuple[gym.spaces.Space, EnvActionType]:
        if rl_action_type != RLActionType.DISCRETE:
            return action_space, action_type

        if isinstance(action_space, gym.spaces.Discrete):
            return action_space, EnvActionType.DISCRETE

        if isinstance(action_space, gym.spaces.Tuple):
            # BOXの場合 TODO

            self.action_tbl = list(itertools.product(*[[n for n in range(s.n)] for s in action_space.spaces]))
            next_space = gym.spaces.Discrete(len(self.action_tbl))
            self.change_type = "Tuple->Discrete"
            return next_space, EnvActionType.DISCRETE

        if isinstance(action_space, gym.spaces.Box):
            new_space, self.action_tbl = self._box_to_discrete(action_space)
            self.change_type = "Box->Discrete"
            return new_space, EnvActionType.DISCRETE

        raise ValueError(f"Unimplemented: {action_space.__class__.__name__}")

    def _box_to_discrete(self, space):
        shape = space.shape
        low_flatten = space.low.flatten()
        high_flatten = space.high.flatten()

        act_list = []
        for i in range(len(low_flatten)):
            act = []
            for j in range(self.action_division_num):
                low = low_flatten[i]
                high = high_flatten[i]
                diff = (high - low) / (self.action_division_num - 1)

                a = low + diff * j
                act.append(a)
            act_list.append(act)

        act_list = list(itertools.product(*act_list))
        action_tbl = np.reshape(act_list, (-1,) + shape).tolist()
        new_space = gym.spaces.Discrete(len(action_tbl))
        return new_space, action_tbl

    def action_decode(self, action: Any, env: EnvBase) -> Any:
        if action is None:
            return 0
        if self.change_type == "":
            return int(action)
        if self.change_type == "Tuple->Discrete":
            return self.action_tbl[action]
        if self.change_type == "Box->Discrete":
            return self.action_tbl[action]

        raise ValueError()

    def invalid_actions_encode(self, invalid_actions: List, env: EnvBase) -> List:
        return invalid_actions

    # --- observation

    def change_observation_info(
        self,
        observation_space: gym.spaces.Box,
        observation_type: EnvObservationType,
        rl_observation_type: RLObservationType,
        env: EnvBase,
    ) -> Tuple[gym.spaces.Box, EnvObservationType]:
        if rl_observation_type != RLObservationType.DISCRETE:
            return observation_space, observation_type

        self._observation_discrete_diff = None

        if observation_type != EnvObservationType.CONTINUOUS:
            return observation_space, observation_type

        # 離散化
        division_num = self.observation_division_num
        self.low = observation_space.low
        high = observation_space.high
        self._observation_discrete_diff = (high - self.low) / division_num

        observation_type = EnvObservationType.DISCRETE
        return observation_space, observation_type

    def observation_encode(self, observation: np.ndarray, env: EnvBase) -> np.ndarray:
        if self._observation_discrete_diff is None:
            return observation

        next_state = (observation - self.low) / self._observation_discrete_diff
        return np.int64(next_state)


if __name__ == "__main__":
    pass
