import itertools
import logging
from dataclasses import dataclass
from typing import Any, List

import gym
import gym.spaces
import numpy as np
from srl.base.define import RLActionType
from srl.base.rl.processor import Processor

logger = logging.getLogger(__name__)


@dataclass
class ActionDiscreteProcessor(Processor):

    action_division_num: int = 5

    def __post_init__(self):
        self.action_tbl = {}
        self.change_type = ""

    def change_action_info(self, action_space: gym.spaces.Space, rl_action_type: RLActionType):
        if rl_action_type != RLActionType.DISCRETE:
            return action_space

        if isinstance(action_space, gym.spaces.Discrete):
            return action_space

        if isinstance(action_space, gym.spaces.Tuple):
            # BOXの場合 TODO

            self.action_tbl = list(itertools.product(*[[n for n in range(s.n)] for s in action_space.spaces]))
            next_space = gym.spaces.Discrete(len(self.action_tbl))
            self.change_type = "Tuple->Discrete"
            return next_space

        if isinstance(action_space, gym.spaces.Box):
            new_space, self.action_tbl = self._box_to_discrete(action_space)
            self.change_type = "Box->Discrete"
            return new_space

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

    def action_decode(self, action: Any):
        if self.change_type == "":
            return action
        if self.change_type == "Tuple->Discrete":
            return self.action_tbl[action]
        if self.change_type == "Box->Discrete":
            return self.action_tbl[action]

        raise ValueError()

    def valid_actions_encode(self, valid_actions: List) -> List:
        if self.change_type == "Tuple->Discrete":
            return [a for a in range(len(self.action_tbl))]
        if self.change_type == "Box->Discrete":
            return [a for a in range(len(self.action_tbl))]
        return valid_actions


if __name__ == "__main__":
    pass
