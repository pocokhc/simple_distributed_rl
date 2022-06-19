import json
import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Union, cast

import numpy as np
from srl.base.define import RLObservationType
from srl.base.env.base import EnvRun
from srl.base.rl.algorithms.discrete_action import DiscreteActionConfig, DiscreteActionWorker
from srl.base.rl.base import RLParameter, RLTrainer
from srl.base.rl.registration import register
from srl.base.rl.remote_memory import SequenceRemoteMemory
from srl.rl.functions.common import render_discrete_action, to_str_observation

logger = logging.getLogger(__name__)


"""
Other
    window length   : o
    invalid_actions : o
"""


# ------------------------------------------------------
# config
# ------------------------------------------------------
@dataclass
class Config(DiscreteActionConfig):

    epsilon: float = 0.1
    test_epsilon: float = 0
    gamma: float = 0.9
    lr: float = 0.1

    window_length: int = 1
    q_init: str = ""

    def __post_init__(self):
        super().__init__()

    @property
    def observation_type(self) -> RLObservationType:
        return RLObservationType.DISCRETE

    @staticmethod
    def getName() -> str:
        return "QL"

    def assert_params(self) -> None:
        super().assert_params()
        assert self.window_length > 0


register(
    Config,
    __name__ + ":RemoteMemory",
    __name__ + ":Parameter",
    __name__ + ":Trainer",
    __name__ + ":Worker",
)


# ------------------------------------------------------
# RemoteMemory
# ------------------------------------------------------
class RemoteMemory(SequenceRemoteMemory):
    pass


# ------------------------------------------------------
# Parameter
# ------------------------------------------------------
class Parameter(RLParameter):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)

        self.Q = {}

    def restore(self, data: Any) -> None:
        self.Q = json.loads(data)

    def backup(self):
        return json.dumps(self.Q)

    def get_action_values(
        self, state: Union[str, List[str]], invalid_actions: List[int], to_str: bool = True
    ) -> List[float]:
        if to_str:
            state = "_".join(state)
        if state not in self.Q:
            if self.config.q_init == "random":
                self.Q[state] = [
                    -np.inf if a in invalid_actions else np.random.normal() for a in range(self.config.action_num)
                ]
            else:
                self.Q[state] = [-np.inf if a in invalid_actions else 0.0 for a in range(self.config.action_num)]
        return self.Q[state]


# ------------------------------------------------------
# Trainer
# ------------------------------------------------------
class Trainer(RLTrainer):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)
        self.parameter = cast(Parameter, self.parameter)
        self.remote_memory = cast(RemoteMemory, self.remote_memory)

        self.train_count = 0

    def get_train_count(self):
        return self.train_count

    def train(self):

        batchs = self.remote_memory.sample()
        td_error_mean = 0
        for batch in batchs:

            state = "_".join(batch["states"][:-1])
            n_state = "_".join(batch["states"][1:])
            action = batch["action"]
            reward = batch["reward"]
            done = batch["done"]
            invalid_actions = batch["invalid_actions"]
            next_invalid_actions = batch["next_invalid_actions"]

            q = self.parameter.get_action_values(state, invalid_actions, to_str=False)
            n_q = self.parameter.get_action_values(n_state, next_invalid_actions, to_str=False)

            if done:
                target_q = reward
            else:
                target_q = reward + self.config.gamma * max(n_q)

            td_error = target_q - q[action]
            self.parameter.Q[state][action] += self.config.lr * td_error

            td_error_mean += td_error
            self.train_count += 1

        if len(batchs) > 0:
            td_error_mean /= len(batchs)

        return {
            "Q": len(self.parameter.Q),
            "td_error": td_error_mean,
        }


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Worker(DiscreteActionWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)
        self.parameter = cast(Parameter, self.parameter)
        self.remote_memory = cast(RemoteMemory, self.remote_memory)

    def call_on_reset(self, state: np.ndarray, invalid_actions: List[int]) -> None:
        self.recent_states = [""] * self.config.window_length
        self.invalid_actions = invalid_actions

        self.recent_states.append(to_str_observation(state))

    def call_policy(self, _state: np.ndarray, invalid_actions: List[int]) -> int:
        self.invalid_actions = invalid_actions

        if self.training:
            epsilon = self.config.epsilon
        else:
            epsilon = self.config.test_epsilon

        if random.random() < epsilon:
            # epsilonより低いならランダムに移動
            action = random.choice([a for a in range(self.config.action_num) if a not in invalid_actions])
        else:
            state = self.recent_states[1:]
            q = self.parameter.get_action_values(state, self.invalid_actions)
            q = np.asarray(q)

            # 最大値を選ぶ（複数あればランダム）
            action = random.choice(np.where(q == q.max())[0])

        self.action = int(action)
        return self.action

    def call_on_step(
        self,
        next_state: np.ndarray,
        reward: float,
        done: bool,
        next_invalid_actions: List[int],
    ) -> Dict:

        self.recent_states.pop(0)
        self.recent_states.append(to_str_observation(next_state))

        if not self.training:
            return {}

        batch = {
            "states": self.recent_states[:],
            "action": self.action,
            "reward": reward,
            "done": done,
            "invalid_actions": self.invalid_actions,
            "next_invalid_actions": next_invalid_actions,
        }
        self.remote_memory.add(batch)
        return {}

    def call_render(self, env: EnvRun) -> None:
        state = self.recent_states[1:]
        q = self.parameter.get_action_values(state, self.invalid_actions)
        maxa = np.argmax(q)

        def _render_sub(a: int) -> str:
            return f"{q[a]:7.5f}"

        render_discrete_action(self.invalid_actions, maxa, env, _render_sub)
