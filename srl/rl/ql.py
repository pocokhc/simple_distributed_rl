import json
import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List, cast

import numpy as np
from srl.base.define import RLObservationType
from srl.base.env.base import EnvBase
from srl.base.rl.algorithms.discrete_action import DiscreteActionConfig, DiscreteActionWorker
from srl.base.rl.base import RLParameter, RLTrainer
from srl.base.rl.registration import register
from srl.base.rl.remote_memory import SequenceRemoteMemory
from srl.rl.functions.common import render_discrete_action, to_str_observation

logger = logging.getLogger(__name__)


"""
Other
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

    def __post_init__(self):
        super().__init__()

    @property
    def observation_type(self) -> RLObservationType:
        return RLObservationType.DISCRETE

    @staticmethod
    def getName() -> str:
        return "QL"


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

    def get_action_values(self, state: str, invalid_actions: List[int]) -> List[float]:
        if state not in self.Q:
            self.Q[state] = [-np.inf if a in invalid_actions else 0.0 for a in range(self.config.nb_actions)]
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
        td_error = 0
        for batch in batchs:

            s = batch["state"]
            n_s = batch["next_state"]
            action = batch["action"]
            reward = batch["reward"]
            done = batch["done"]
            invalid_actions = batch["invalid_actions"]
            next_invalid_actions = batch["next_invalid_actions"]

            q = self.parameter.get_action_values(s, invalid_actions)
            n_q = self.parameter.get_action_values(n_s, next_invalid_actions)

            if done:
                target_q = reward
            else:
                target_q = reward + self.config.gamma * max(n_q)

            td_error = target_q - q[action]
            q[action] += self.config.lr * td_error

            td_error += td_error
            self.train_count += 1

        if len(batchs) > 0:
            td_error /= len(batchs)

        return {
            "Q": len(self.parameter.Q),
            "td_error": td_error,
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
        self.state = to_str_observation(state)
        self.invalid_actions = invalid_actions

        if self.training:
            self.epsilon = self.config.epsilon
        else:
            self.epsilon = self.config.test_epsilon

    def call_policy(self, state: np.ndarray, invalid_actions: List[int]) -> int:
        self.state = to_str_observation(state)
        self.invalid_actions = invalid_actions

        if random.random() < self.epsilon:
            # epsilonより低いならランダムに移動
            action = random.choice([a for a in range(self.config.nb_actions) if a not in invalid_actions])
        else:
            q = self.parameter.get_action_values(self.state, self.invalid_actions)
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
        if not self.training:
            return {}

        batch = {
            "state": self.state,
            "next_state": to_str_observation(next_state),
            "action": self.action,
            "reward": reward,
            "done": done,
            "invalid_actions": self.invalid_actions,
            "next_invalid_actions": next_invalid_actions,
        }
        self.remote_memory.add(batch)
        return {}

    def render(self, env: EnvBase, player_index: int) -> None:
        q = self.parameter.get_action_values(self.state, self.invalid_actions)
        maxa = np.argmax(q)

        def _render_sub(a: int) -> str:
            return f"{q[a]:7.5f}"

        render_discrete_action(self.invalid_actions, maxa, env, _render_sub)
