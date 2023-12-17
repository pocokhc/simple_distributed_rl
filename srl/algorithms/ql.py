import json
import logging
import random
from dataclasses import dataclass
from typing import Any, List, Tuple, cast

import numpy as np

from srl.base.define import RLBaseTypes
from srl.base.rl.algorithms.discrete_action import DiscreteActionWorker
from srl.base.rl.base import RLParameter, RLTrainer
from srl.base.rl.config import RLConfig
from srl.base.rl.registration import register
from srl.rl.functions import common
from srl.rl.memories.sequence_memory import SequenceMemory
from srl.rl.schedulers.scheduler import SchedulerConfig

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
    #: ε-greedy parameter for Test
    test_epsilon: float = 0
    #: <:ref:`scheduler`> ε-greedy parameter for Train
    epsilon: float = 0.1  # type: ignore , type OK
    #: <:ref:`scheduler`> Learning rate
    lr: float = 0.1  # type: ignore , type OK
    #: Discount rate
    discount: float = 0.9
    #: How to initialize Q table
    #:
    #: Parameters:
    #:   "": 0
    #:   "random": random.random()
    #:   "normal": np.random.normal()
    q_init: str = ""

    def __post_init__(self):
        super().__post_init__()

        self.epsilon: SchedulerConfig = SchedulerConfig(cast(float, self.epsilon))
        self.lr: SchedulerConfig = SchedulerConfig(cast(float, self.lr))

    @property
    def base_action_type(self) -> RLBaseTypes:
        return RLBaseTypes.DISCRETE

    @property
    def base_observation_type(self) -> RLBaseTypes:
        return RLBaseTypes.DISCRETE

    def get_use_framework(self) -> str:
        return ""

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
    __name__ + ":Memory",
    __name__ + ":Parameter",
    __name__ + ":Trainer",
    __name__ + ":Worker",
)


# ------------------------------------------------------
# Memory
# ------------------------------------------------------
class Memory(SequenceMemory):
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

    def get_action_values(self, state: str, update_invalid_actions: list = []) -> List[float]:
        if state not in self.Q:
            if self.config.q_init == "random":
                self.Q[state] = [random.random() for a in range(self.config.action_num)]
            elif self.config.q_init == "normal":
                self.Q[state] = [np.random.normal() for a in range(self.config.action_num)]
            else:
                self.Q[state] = [0.0 for a in range(self.config.action_num)]
        for a in update_invalid_actions:
            self.Q[state][a] = -np.inf
        return self.Q[state]


# ------------------------------------------------------
# Trainer
# ------------------------------------------------------
class Trainer(RLTrainer):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config
        self.parameter: Parameter = self.parameter
        self.lr_scheduler = self.config.lr.create_schedulers()

    def train(self) -> None:
        if self.memory.is_warmup_needed():
            return
        batchs = self.memory.sample(self.batch_size, self.train_count)

        td_error = 0
        lr = self.lr_scheduler.get_and_update_rate(self.train_count)
        for batch in batchs:
            state = batch[0]
            n_state = batch[1]
            action = batch[2]
            reward = batch[3]
            done = batch[4]
            next_invalid_actions = batch[5]

            target_q = reward
            if not done:
                n_q = self.parameter.get_action_values(n_state, next_invalid_actions)
                target_q += self.config.discount * max(n_q)

            td_error = target_q - self.parameter.get_action_values(state)[action]
            self.parameter.Q[state][action] += lr * td_error
            self.train_count += 1

        self.train_info = {
            "size": len(self.parameter.Q),
            "td_error": td_error,
            "lr": lr,
        }


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Worker(DiscreteActionWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config
        self.parameter: Parameter = self.parameter

        self.epsilon_scheduler = self.config.epsilon.create_schedulers()

    def call_on_reset(self, state: np.ndarray, invalid_actions: List[int]) -> dict:
        return {}

    def call_policy(self, state: np.ndarray, invalid_actions: List[int]) -> Tuple[int, dict]:
        self.state = common.to_str_observation(state)

        if self.training:
            epsilon = self.epsilon_scheduler.get_and_update_rate(self.total_step)
        else:
            epsilon = self.config.test_epsilon

        if random.random() < epsilon:
            # epsilonより低いならランダムに移動
            self.action = random.choice([a for a in range(self.config.action_num) if a not in invalid_actions])
        else:
            # 最大値を選択
            q = self.parameter.get_action_values(self.state)
            self.action = common.get_random_max_index(q, invalid_actions)

        return self.action, {"epsilon": epsilon}

    def call_on_step(
        self,
        next_state: np.ndarray,
        reward: float,
        done: bool,
        next_invalid_actions: List[int],
    ) -> dict:
        if not self.training:
            return {}

        self.memory.add(
            [
                self.state,
                common.to_str_observation(next_state),
                self.action,
                reward,
                done,
                next_invalid_actions,
            ]
        )
        return {}

    def render_terminal(self, worker, **kwargs) -> None:
        q = self.parameter.get_action_values(self.state)
        maxa = np.argmax(q)

        def _render_sub(a: int) -> str:
            return f"{q[a]:7.5f}"

        common.render_discrete_action(maxa, worker.env, self.config, _render_sub)
