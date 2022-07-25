import json
from dataclasses import dataclass
from typing import Any, Dict, List, cast

import numpy as np
from srl.base.define import RLObservationType
from srl.base.env.base import EnvRun
from srl.base.rl.algorithms.discrete_action import DiscreteActionConfig, DiscreteActionWorker
from srl.base.rl.base import RLParameter, RLTrainer
from srl.base.rl.registration import register
from srl.base.rl.remote_memory import SequenceRemoteMemory
from srl.rl.functions.common import render_discrete_action, to_str_observation


# ------------------------------------------------------
# config
# ------------------------------------------------------
@dataclass
class Config(DiscreteActionConfig):

    discount: float = 0.9
    lr: float = 0.1

    def __post_init__(self):
        super().__init__()

    @property
    def observation_type(self) -> RLObservationType:
        return RLObservationType.DISCRETE

    @staticmethod
    def getName() -> str:
        return "VanillaPolicyDiscrete"


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

        # パラメータ
        self.policy = {}

    def restore(self, data: Any) -> None:
        self.policy = json.loads(data)

    def backup(self) -> Any:
        return json.dumps(self.policy)

    # ---------------------------------

    def get_probs(self, state_str: str, invalid_actions):
        if state_str not in self.policy:
            self.policy[state_str] = [None if a in invalid_actions else 0.0 for a in range(self.config.action_num)]

        probs = []
        for val in self.policy[state_str]:
            if val is None:
                probs.append(0)
            else:
                probs.append(np.exp(val))
        probs /= np.sum(probs)
        return probs


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
        if len(batchs) == 0:
            return {}

        loss = []
        for batch in batchs:
            state = batch["state"]
            action = batch["action"]
            reward = batch["reward"]
            invalid_actions = batch["invalid_actions"]

            prob = self.parameter.get_probs(state, invalid_actions)[action]

            # ∇logπ
            diff_logpi = 1 - prob

            # ∇J(θ)
            diff_j = diff_logpi * reward

            # ポリシー更新
            self.parameter.policy[state][action] += self.config.lr * diff_j
            loss.append(abs(diff_j))

            self.train_count += 1

        return {"loss": np.mean(loss)}


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
        self.history = []

    def call_policy(self, state: np.ndarray, invalid_actions: List[int]) -> int:
        self.state = to_str_observation(state)
        self.invalid_actions = invalid_actions

        probs = self.parameter.get_probs(self.state, invalid_actions)
        action = np.random.choice([a for a in range(self.config.action_num)], p=probs)
        self.action = int(action)
        self.prob = probs[self.action]
        return action

    def call_on_step(
        self,
        next_state: np.ndarray,
        reward: float,
        done: bool,
        next_invalid_actions: List[int],
    ) -> Dict:
        if not self.training:
            return {}
        self.history.append([self.state, self.action, self.invalid_actions, reward])

        if done:
            reward = 0
            for h in reversed(self.history):
                reward = h[3] + self.config.discount * reward
                batch = {
                    "state": h[0],
                    "action": h[1],
                    "invalid_actions": h[2],
                    "reward": reward,
                }
                self.remote_memory.add(batch)

        return {}

    def call_render(self, env: EnvRun) -> None:
        probs = self.parameter.get_probs(self.state, self.invalid_actions)
        vals = [0 if v is None else v for v in self.parameter.policy[self.state]]
        maxa = np.argmax(vals)

        def _render_sub(action: int) -> str:
            return f"{probs[action]*100:5.1f}% ({vals[action]:.5f})"

        render_discrete_action(self.invalid_actions, maxa, env, _render_sub)
