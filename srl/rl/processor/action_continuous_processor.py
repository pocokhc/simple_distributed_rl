import logging
from dataclasses import dataclass

import gym
import gym.spaces
import numpy as np
from srl.base.define import RLActionType
from srl.base.rl.processor import Processor
from srl.rl.processor.common import tuple_to_box

logger = logging.getLogger(__name__)


@dataclass
class ActionContinuousProcessor(Processor):
    def __post_init__(self):
        self.action_tbl = None
        self.change_type = ""

    def change_action_info(self, action_space: gym.spaces.Space, rl_action_type: RLActionType):
        if rl_action_type != RLActionType.CONTINUOUS:
            return action_space

        if isinstance(action_space, gym.spaces.Discrete):
            new_space = gym.spaces.Box(low=0, high=action_space.n - 1, shape=(1,))
            self.change_type = "Discrete->Box"
            return new_space

        if isinstance(action_space, gym.spaces.Tuple):
            new_space = tuple_to_box(action_space)
            self.change_type = "Tuple->Box"
            return new_space

        if isinstance(action_space, gym.spaces.Box):
            return action_space  # no change

        raise ValueError(f"Unimplemented: {action_space.__class__.__name__}")

    def action_decode(self, action):
        if self.change_type == "":
            return action
        if self.change_type == "Discrete->Box":
            return np.round(action[0])
        if self.change_type == "Tuple->Box":
            return [np.round(a) for a in action]
        raise ValueError()


if __name__ == "__main__":
    pass
