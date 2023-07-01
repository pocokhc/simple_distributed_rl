import json
import logging
import random
from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np

from srl.base.define import RLTypes
from srl.base.rl.algorithms.discrete_action import DiscreteActionWorker
from srl.base.rl.base import RLParameter, RLTrainer
from srl.base.rl.config import RLConfig
from srl.base.rl.registration import register
from srl.base.rl.remote_memory import SequenceRemoteMemory
from srl.rl.functions import common

logger = logging.getLogger(__name__)


"""
Other
    invalid_actions : o
"""


# ------------------------------------------------------
# config
# ------------------------------------------------------
@dataclass
class Config(RLConfig):
    epsilon: float = 0.1
    test_epsilon: float = 0
    discount: float = 0.9
    lr: float = 0.1


    @property
    def base_action_type(self) -> RLTypes:
        return RLTypes.DISCRETE

    @property
    def base_observation_type(self) -> RLTypes:
        return RLTypes.DISCRETE

    def getName(self) -> str:
        return "QL"

    @property
    def info_types(self) -> dict:
        return {
            "size": {"type": int, "data": "last"},
            "td_error": {},
            "epsilon": {},
        }


register(
    Config(),
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
        self.config: Config = self.config

        self.Q = {}

    def call_restore(self, data: Any, **kwargs) -> None:
        self.Q = json.loads(data)

    def call_backup(self, **kwargs):
        return json.dumps(self.Q)

    def get_action_values(self, state: str) -> List[float]:
        if state not in self.Q:
            self.Q[state] = [
                np.random.normal() if self.config.q_init == "random" else 0.0 for a in range(self.config.action_num)
            ]
        return self.Q[state]


# ------------------------------------------------------
# Trainer
# ------------------------------------------------------
class Trainer(RLTrainer):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config
        self.parameter: Parameter = self.parameter
        self.remote_memory: RemoteMemory = self.remote_memory

        self.train_count = 0

    def get_train_count(self) -> int:
        return self.train_count

    def train(self) -> dict:
        batchs = self.remote_memory.sample()
        td_error = 0
        for batch in batchs:
            state = batch[0]
            n_state = batch[1]
            action = batch[2]
            reward = batch[3]
            done = batch[4]

            target_q = reward
            if not done:
                n_q = self.parameter.get_action_values(n_state)
                target_q += self.config.discount * max(n_q)

            td_error = target_q - self.parameter.get_action_values(state)[action]
            self.parameter.Q[state][action] += self.config.lr * td_error

            self.train_count += 1

        return {
            "size": len(self.parameter.Q),
            "td_error": td_error,
        }


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Worker(DiscreteActionWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config
        self.parameter: Parameter = self.parameter
        self.remote_memory: RemoteMemory = self.remote_memory

    def call_on_reset(self, state: np.ndarray, invalid_actions: List[int]) -> dict:
        if self.training:
            self.epsilon = self.config.epsilon
        else:
            self.epsilon = self.config.test_epsilon
        return {}

    def call_policy(self, state: np.ndarray, invalid_actions: List[int]) -> Tuple[int, dict]:
        self.state = common.to_str_observation(state)

        if random.random() < self.epsilon:
            # epsilonより低いならランダムに移動
            self.action = random.choice([a for a in range(self.config.action_num) if a not in invalid_actions])
        else:
            # 最大値を選択
            q = self.parameter.get_action_values(self.state)
            self.action = common.get_random_max_index(q, invalid_actions)

        return self.action, {"epsilon": self.epsilon}

    def call_on_step(
        self,
        next_state: np.ndarray,
        reward: float,
        done: bool,
        next_invalid_actions: List[int],
    ) -> dict:
        if not self.training:
            return {}

        self.remote_memory.add(
            [
                self.state,
                common.to_str_observation(next_state),
                self.action,
                reward,
                done,
            ]
        )
        return {}

    def render_terminal(self, worker, **kwargs) -> None:
        q = self.parameter.get_action_values(self.state)
        maxa = np.argmax(q)

        def _render_sub(a: int) -> str:
            return f"{q[a]:7.5f}"

        common.render_discrete_action(maxa, worker.env, self.config, _render_sub)
