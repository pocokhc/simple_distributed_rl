import json
import logging
import random
from dataclasses import dataclass
from typing import Any, List, Union

import numpy as np

from srl.base.rl.algorithms.base_ql import RLConfig, RLWorker
from srl.base.rl.parameter import RLParameter
from srl.base.rl.registration import register
from srl.base.rl.trainer import RLTrainer
from srl.rl import functions as funcs
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

    def on_reset(self, worker):
        pass

    def policy(self, worker) -> int:
        self.state = self.config.observation_space.to_str(worker.state)
        invalid_actions = worker.get_invalid_actions()

        if self.training:
            epsilon = self.epsilon_sch.get_and_update_rate(self.total_step)
        else:
            epsilon = self.config.test_epsilon

        if random.random() < epsilon:
            # epsilonより低いならランダムに移動
            self.action = random.choice([a for a in range(self.config.action_space.n) if a not in invalid_actions])
        else:
            # 最大値を選択
            q = self.parameter.get_action_values(self.state)
            self.action = funcs.get_random_max_index(q, invalid_actions)

        self.info["epsilon"] = epsilon
        return self.action

    def on_step(self, worker):
        if not self.training:
            return
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
            self.config.observation_space.to_str(worker.state),
            self.action,
            worker.reward,
            worker.terminated,
            worker.get_invalid_actions(),
        ]
        self.memory.add(batch)

    def render_terminal(self, worker, **kwargs) -> None:
        q = self.parameter.get_action_values(self.state)
        maxa = np.argmax(q)

        def _render_sub(a: int) -> str:
            return f"{q[a]:7.5f}"

        funcs.render_discrete_action(int(maxa), self.config.action_space, worker.env, _render_sub)
