import json
import logging
import random
from dataclasses import dataclass
from typing import Any, List, Tuple, Union

import numpy as np

from srl.base.define import InfoType, RLBaseTypes
from srl.base.rl.config import RLConfig
from srl.base.rl.parameter import RLParameter
from srl.base.rl.registration import register
from srl.base.rl.trainer import RLTrainer
from srl.base.rl.worker import RLWorker
from srl.base.spaces.array_discrete import ArrayDiscreteSpace
from srl.base.spaces.discrete import DiscreteSpace
from srl.rl.functions import helper
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
class Config(RLConfig[DiscreteSpace, ArrayDiscreteSpace]):
    #: ε-greedy parameter for Test
    test_epsilon: float = 0
    #: <:ref:`scheduler`> ε-greedy parameter for Train
    epsilon: Union[float, SchedulerConfig] = 0.1
    #: <:ref:`scheduler`> Learning rate
    lr: Union[float, SchedulerConfig] = 0.1
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

    def get_base_action_type(self) -> RLBaseTypes:
        return RLBaseTypes.DISCRETE

    def get_base_observation_type(self) -> RLBaseTypes:
        return RLBaseTypes.DISCRETE

    def get_framework(self) -> str:
        return ""

    def get_name(self) -> str:
        return "QL"

    def get_info_types(self) -> dict:
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
class Parameter(RLParameter[Config]):
    def __init__(self, *args):
        super().__init__(*args)

        self.Q = {}

    def call_restore(self, data: Any, **kwargs) -> None:
        self.Q = json.loads(data)

    def call_backup(self, **kwargs):
        return json.dumps(self.Q)

    def get_action_values(self, state: str, update_invalid_actions: list = []) -> List[float]:
        if state not in self.Q:
            if self.config.q_init == "random":
                self.Q[state] = [random.random() for a in range(self.config.action_space.n)]
            elif self.config.q_init == "normal":
                self.Q[state] = [np.random.normal() for a in range(self.config.action_space.n)]
            else:
                self.Q[state] = [0.0 for a in range(self.config.action_space.n)]
        for a in update_invalid_actions:
            self.Q[state][a] = -np.inf
        return self.Q[state]


# ------------------------------------------------------
# Trainer
# ------------------------------------------------------
class Trainer(RLTrainer[Config, Parameter]):
    def __init__(self, *args):
        super().__init__(*args)

        self.lr_sch = SchedulerConfig.create_scheduler(self.config.lr)

    def train(self) -> None:
        if self.memory.is_warmup_needed():
            return
        batchs = self.memory.sample()

        td_error = 0
        lr = self.lr_sch.get_and_update_rate(self.train_count)
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

        self.info = {
            "size": len(self.parameter.Q),
            "td_error": td_error,
            "lr": lr,
        }


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Worker(RLWorker[Config, Parameter]):
    def __init__(self, *args):
        super().__init__(*args)

        self.epsilon_sch = SchedulerConfig.create_scheduler(self.config.epsilon)

    def on_reset(self, worker) -> InfoType:
        return {}

    def policy(self, worker) -> Tuple[int, InfoType]:
        self.state = self.observation_space.to_str(worker.state)
        invalid_actions = worker.get_invalid_actions()

        if self.training:
            epsilon = self.epsilon_sch.get_and_update_rate(self.total_step)
        else:
            epsilon = self.config.test_epsilon

        if random.random() < epsilon:
            # epsilonより低いならランダムに移動
            self.action = random.choice([a for a in range(self.action_space.n) if a not in invalid_actions])
        else:
            # 最大値を選択
            q = self.parameter.get_action_values(self.state)
            self.action = helper.get_random_max_index(q, invalid_actions)

        return self.action, {"epsilon": epsilon}

    def on_step(self, worker) -> InfoType:
        if not self.training:
            return {}
        """
        [
            state,
            n_state,
            action,
            reward,
            done,
            next_invalid_actions,
        ]
        """
        batch = [
            self.state,
            self.observation_space.to_str(worker.state),
            self.action,
            worker.reward,
            worker.terminated,
            worker.get_invalid_actions(),
        ]
        self.memory.add(batch)
        return {}

    def render_terminal(self, worker, **kwargs) -> None:
        q = self.parameter.get_action_values(self.state)
        maxa = np.argmax(q)

        def _render_sub(a: int) -> str:
            return f"{q[a]:7.5f}"

        helper.render_discrete_action(int(maxa), self.action_space.n, worker.env, _render_sub)
