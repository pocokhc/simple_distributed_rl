import json
from dataclasses import dataclass
from typing import Any, Dict, List, cast

import numpy as np
from srl.base.define import RLObservationType
from srl.base.env.base import EnvRun
from srl.base.rl.algorithms.continuous_action import ContinuousActionConfig, ContinuousActionWorker
from srl.base.rl.base import RLParameter, RLTrainer
from srl.base.rl.registration import register
from srl.base.rl.remote_memory import SequenceRemoteMemory
from srl.rl.functions.common import to_str_observation


# ------------------------------------------------------
# config
# ------------------------------------------------------
@dataclass
class Config(ContinuousActionConfig):

    discount: float = 0.9
    lr: float = 0.1

    def __post_init__(self):
        super().__init__()

    @property
    def observation_type(self) -> RLObservationType:
        return RLObservationType.DISCRETE

    @staticmethod
    def getName() -> str:
        return "VanillaPolicyContinuous"


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

    def get_param(self, state_str: str):
        if state_str not in self.policy:
            self.policy[state_str] = {
                "mean": 0.0,
                "stddev_logits": 0.5,
            }

        mean = self.policy[state_str]["mean"]
        stddev = np.exp(self.policy[state_str]["stddev_logits"])
        return mean, stddev


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

        loss_mean = []
        loss_stddev = []
        for batch in batchs:
            state = batch["state"]
            action = batch["action"]
            reward = batch["reward"]
            mean, stddev = self.parameter.get_param(state)

            # 平均
            mean_diff_logpi = (action - mean) / (stddev**2)
            mean_diff_j = mean_diff_logpi * reward
            new_mean = self.parameter.policy[state]["mean"] + self.config.lr * mean_diff_j

            # 分散
            stddev_diff_logpi = (((action - mean) ** 2) - (stddev**2)) / (stddev**3)
            stddev_diff_j = stddev_diff_logpi * reward
            new_stddev = self.parameter.policy[state]["stddev_logits"] + self.config.lr * stddev_diff_j

            # 更新幅が大きすぎる場合は更新しない
            if abs(mean_diff_j) < 1 and abs(stddev_diff_j) < 5:
                self.parameter.policy[state]["mean"] = new_mean
                self.parameter.policy[state]["stddev_logits"] = new_stddev

            loss_mean.append(mean_diff_j)
            loss_stddev.append(stddev_diff_j)

            self.train_count += 1

        return {
            "loss_mean": np.mean(loss_mean),
            "loss_stddev": np.mean(loss_stddev),
        }


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Worker(ContinuousActionWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)
        self.parameter = cast(Parameter, self.parameter)
        self.remote_memory = cast(RemoteMemory, self.remote_memory)

    def call_on_reset(self, state: np.ndarray) -> None:
        self.state = to_str_observation(state)
        self.history = []

    def call_policy(self, state: np.ndarray) -> List[float]:
        self.state = to_str_observation(state)

        # パラメータ
        mean, stddev = self.parameter.get_param(self.state)

        # ガウス分布に従った乱数を出す
        self.action = env_action = mean + np.random.normal() * stddev

        # -inf～infの範囲を取るので実際に環境に渡すアクションはlowとhighで切り取る
        # 本当はポリシーが変化しちゃうのでよくない（暫定対処）
        env_action = np.clip(env_action, self.config.action_low[0], self.config.action_high[0])

        return env_action

    def call_on_step(
        self,
        next_state: np.ndarray,
        reward: float,
        done: bool,
    ) -> Dict:
        if not self.training:
            return {}
        self.history.append([self.state, self.action, reward])

        if done:
            reward = 0
            for h in reversed(self.history):
                reward = h[2] + self.config.discount * reward
                batch = {
                    "state": h[0],
                    "action": h[1],
                    "reward": reward,
                }
                self.remote_memory.add(batch)

        return {}

    def render_terminal(self, env, worker, **kwargs) -> None:
        mean, stddev = self.parameter.get_param(self.state)
        print(f"mean {mean:.5f}, stddev {stddev:.5f}")
