import json
import logging
import random
from dataclasses import dataclass, field
from typing import Any, List, Literal

import numpy as np

from srl.base.rl.algorithms.base_ql import RLConfig, RLWorker
from srl.base.rl.parameter import RLParameter
from srl.base.rl.registration import register
from srl.base.rl.trainer import RLTrainer
from srl.rl import functions as funcs
from srl.rl.memories.single_use_buffer import RLSingleUseBuffer
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

    #: ε-greedy parameter for Train
    epsilon: float = 0.1
    #: <:ref:`SchedulerConfig`>
    epsilon_scheduler: SchedulerConfig = field(default_factory=lambda: SchedulerConfig())
    #: Learning rate
    lr: float = 0.1
    #: <:ref:`SchedulerConfig`>
    lr_scheduler: SchedulerConfig = field(default_factory=lambda: SchedulerConfig())

    #: Discount rate
    discount: float = 0.9
    #: How to initialize Q table
    #:
    #: Parameters:
    #:   "": 0
    #:   "random": random.random()
    #:   "normal": np.random.normal()
    q_init: Literal["", "random", "normal"] = ""

    def get_name(self) -> str:
        return "QL"


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
class Memory(RLSingleUseBuffer):
    pass


# ------------------------------------------------------
# Parameter
# ------------------------------------------------------
class Parameter(RLParameter[Config]):
    def setup(self):
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
class Trainer(RLTrainer[Config, Parameter, Memory]):
    def on_setup(self) -> None:
        self.lr_sch = self.config.lr_scheduler.create(self.config.lr)

    def train(self) -> None:
        batches = self.memory.sample()
        if batches is None:
            return

        td_error = 0
        lr = self.lr_sch.update(self.train_count).to_float()
        for batch in batches:
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

        self.info["size"] = len(self.parameter.Q)
        self.info["td_error"] = td_error
        self.info["lr"] = lr


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Worker(RLWorker[Config, Parameter, Memory]):
    def on_setup(self, worker, context) -> None:
        self.epsilon_sch = self.config.epsilon_scheduler.create(self.config.epsilon)

    def on_teardown(self, worker) -> None:
        pass

    def on_reset(self, worker):
        pass

    def policy(self, worker) -> int:
        self.state = self.config.observation_space.to_str(worker.state)

        if self.training:
            epsilon = self.epsilon_sch.update(self.step_in_training).to_float()
        else:
            epsilon = self.config.test_epsilon

        if random.random() < epsilon:
            # epsilonより低いならランダムに移動
            action = random.choice([a for a in range(self.config.action_space.n) if a not in worker.invalid_actions])
        else:
            # 最大値を選択
            q = self.parameter.get_action_values(self.state)
            action = funcs.get_random_max_index(q, worker.invalid_actions)

        self.info["epsilon"] = epsilon
        return action

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
        self.memory.add(
            [
                self.state,
                self.config.observation_space.to_str(worker.next_state),
                worker.action,
                worker.reward,
                worker.terminated,
                worker.next_invalid_actions,
            ]
        )

    def render_terminal(self, worker, **kwargs) -> None:
        q = self.parameter.get_action_values(self.state)
        maxa = np.argmax(q)

        def _render_sub(a: int) -> str:
            return f"{q[a]:7.5f}"

        worker.print_discrete_action_info(int(maxa), _render_sub)
