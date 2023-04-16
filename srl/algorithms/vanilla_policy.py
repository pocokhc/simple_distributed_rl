import json
from dataclasses import dataclass
from typing import Any, List, Tuple, cast

import numpy as np

from srl.base.define import RLAction, RLActionType, RLObservationType
from srl.base.rl.algorithms.any_action import AnyActionConfig, AnyActionWorker
from srl.base.rl.base import RLParameter, RLTrainer
from srl.base.rl.registration import register
from srl.base.rl.remote_memory import SequenceRemoteMemory
from srl.rl.functions.common import render_discrete_action, to_str_observation


# ------------------------------------------------------
# config
# ------------------------------------------------------
@dataclass
class Config(AnyActionConfig):
    discount: float = 0.9
    lr: float = 0.1

    @property
    def observation_type(self) -> RLObservationType:
        return RLObservationType.DISCRETE

    def getName(self) -> str:
        return "VanillaPolicy"


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
        self.config = cast(Config, self.config)

        # パラメータ
        self.policy = {}

    def call_restore(self, data: Any, **kwargs) -> None:
        self.policy = json.loads(data)

    def call_backup(self, **kwargs) -> Any:
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

    def get_normal(self, state_str: str):
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

        if self.config.env_action_type == RLActionType.DISCRETE:
            return self._train_discrete(batchs)
        else:
            return self._train_continuous(batchs)

    def _train_discrete(self, batchs):
        loss = []
        for batch in batchs:
            state = batch["state"]
            action = batch["action"]
            reward = batch["reward"]
            invalid_actions = batch["invalid_actions"]

            prob = self.parameter.get_probs(state, invalid_actions)[action]

            # ∇log π
            diff_logpi = 1 - prob

            # ∇J(θ)
            diff_j = diff_logpi * reward

            # ポリシー更新
            self.parameter.policy[state][action] += self.config.lr * diff_j
            loss.append(abs(diff_j))

            self.train_count += 1

        return {"loss": np.mean(loss)}

    def _train_continuous(self, batchs):
        loss_mean = []
        loss_stddev = []
        for batch in batchs:
            state = batch["state"]
            action = batch["action"]
            reward = batch["reward"]
            mean, stddev = self.parameter.get_normal(state)

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
class Worker(AnyActionWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)
        self.parameter = cast(Parameter, self.parameter)
        self.remote_memory = cast(RemoteMemory, self.remote_memory)

    def call_on_reset(self, state: np.ndarray, invalid_actions: List[RLAction]) -> dict:
        self.state = to_str_observation(state)
        self.invalid_actions = invalid_actions
        self.history = []

        return {}

    def call_policy(self, state: np.ndarray, invalid_actions: List[RLAction]) -> Tuple[RLAction, dict]:
        self.state = to_str_observation(state)
        self.invalid_actions = invalid_actions

        if self.config.env_action_type == RLActionType.DISCRETE:
            # --- 離散
            probs = self.parameter.get_probs(self.state, invalid_actions)
            action = np.random.choice([a for a in range(self.config.action_num)], p=probs)
            self.action = int(action)
            env_action = self.action
            self.prob = probs[self.action]

        else:
            # --- 連続
            # パラメータ
            mean, stddev = self.parameter.get_normal(self.state)

            # ガウス分布に従った乱数を出す
            self.action = env_action = mean + np.random.normal() * stddev

            # -inf～infの範囲を取るので実際に環境に渡すアクションはlowとhighで切り取る
            # 本当はポリシーが変化しちゃうのでよくない（暫定対処）
            env_action = np.clip(env_action, self.config.action_low[0], self.config.action_high[0])

        return env_action, {}

    def call_on_step(
        self,
        next_state: np.ndarray,
        reward: float,
        done: bool,
        next_invalid_actions: List[RLAction],
    ) -> dict:
        if not self.training:
            return {}
        self.history.append(
            [
                self.state,
                self.action,
                self.invalid_actions,
                reward,
            ]
        )

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

    def render_terminal(self, env, worker, **kwargs) -> None:
        if self.config.env_action_type == RLActionType.DISCRETE:
            probs = self.parameter.get_probs(self.state, self.invalid_actions)
            vals = [0 if v is None else v for v in self.parameter.policy[self.state]]
            maxa = np.argmax(vals)

            def _render_sub(action: int) -> str:
                return f"{probs[action]*100:5.1f}% ({vals[action]:.5f})"

            render_discrete_action(self.invalid_actions, maxa, env, _render_sub)

        else:
            mean, stddev = self.parameter.get_normal(self.state)
            print(f"mean {mean:.5f}, stddev {stddev:.5f}")
