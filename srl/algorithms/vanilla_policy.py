import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from srl.base.define import SpaceTypes
from srl.base.rl.algorithms.base_vanilla_policy import RLConfig, RLWorker
from srl.base.rl.parameter import RLParameter
from srl.base.rl.registration import register
from srl.base.rl.trainer import RLTrainer
from srl.base.spaces.array_continuous import ArrayContinuousSpace
from srl.base.spaces.discrete import DiscreteSpace
from srl.rl import functions as funcs
from srl.rl.memories.single_use_buffer import RLSingleUseBuffer
from srl.rl.schedulers.scheduler import SchedulerConfig


@dataclass
class Config(RLConfig):
    discount: float = 0.9
    lr: float = 0.1
    #: <:ref:`SchedulerConfig`>
    lr_scheduler: SchedulerConfig = field(default_factory=lambda: SchedulerConfig())

    update_max_mean: float = 1
    update_max_stddev: float = 5
    update_stddev_rate: float = 0.1

    def get_name(self) -> str:
        return "VanillaPolicy"


register(
    Config(),
    __name__ + ":Memory",
    __name__ + ":Parameter",
    __name__ + ":Trainer",
    __name__ + ":Worker",
)


class Memory(RLSingleUseBuffer):
    pass


class Parameter(RLParameter[Config]):
    def setup(self):
        self.policy = {}

    def call_restore(self, data: Any, **kwargs) -> None:
        self.policy = json.loads(data)

    def call_backup(self, **kwargs) -> Any:
        return json.dumps(self.policy)

    # ---------------------------------

    def get_probs(self, state_str: str, invalid_actions):
        if state_str not in self.policy:
            self.policy[state_str] = [None if a in invalid_actions else 0.0 for a in range(self.config.action_space.n)]

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


class Trainer(RLTrainer[Config, Parameter, Memory]):
    def on_setup(self) -> None:
        self.lr_sch = self.config.lr_scheduler.create(self.config.lr)

    def train(self) -> None:
        batches = self.memory.sample()
        if batches is None:
            return

        if self.config.action_space.stype == SpaceTypes.DISCRETE:
            self._train_discrete(batches)
        else:
            self._train_continuous(batches)
        self.train_count += len(batches)

    def _train_discrete(self, batches):
        loss = []
        lr = 0
        for batch in batches:
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
            lr = self.lr_sch.update(self.train_count).to_float()
            self.parameter.policy[state][action] += lr * diff_j
            loss.append(abs(diff_j))

        self.info["size"] = len(self.parameter.policy)
        self.info["loss"] = np.mean(loss)
        self.info["lr"] = lr

    def _train_continuous(self, batches):
        loss_mean = []
        loss_stddev = []
        for batch in batches:
            state = batch["state"]
            action = batch["action"]
            reward = batch["reward"]
            mean, stddev = self.parameter.get_normal(state)

            lr = self.lr_sch.update(self.train_count).to_float()

            # 平均
            mean_diff_logpi = (action - mean) / (stddev**2)
            mean_diff_j = mean_diff_logpi * reward
            new_mean = self.parameter.policy[state]["mean"] + lr * mean_diff_j

            # 分散
            stddev_diff_logpi = (((action - mean) ** 2) - (stddev**2)) / (stddev**3)
            stddev_diff_j = stddev_diff_logpi * reward
            new_stddev = self.parameter.policy[state]["stddev_logits"] + lr * stddev_diff_j * self.config.update_stddev_rate

            # 更新幅が大きすぎる場合は更新しない
            if abs(mean_diff_j) < self.config.update_max_mean and abs(stddev_diff_j) < self.config.update_max_stddev:
                self.parameter.policy[state]["mean"] = new_mean
                self.parameter.policy[state]["stddev_logits"] = new_stddev

            loss_mean.append(mean_diff_j)
            loss_stddev.append(stddev_diff_j)

        self.info["size"] = len(self.parameter.policy)
        self.info["loss_mean"] = np.mean(loss_mean)
        self.info["loss_stddev"] = np.mean(loss_stddev)


class Worker(RLWorker[Config, Parameter, Memory]):
    def on_reset(self, worker):
        self.state = self.config.observation_space.to_str(worker.state)
        self.history = []

    def policy(self, worker) -> Any:
        self.state = self.config.observation_space.to_str(worker.state)
        invalid_actions = worker.invalid_actions

        if isinstance(self.config.action_space, DiscreteSpace):
            act_space = self.config.action_space
            # --- 離散
            probs = self.parameter.get_probs(self.state, invalid_actions)
            if self.training:
                action = np.random.choice([a for a in range(act_space.n)], p=probs)
                self.action = env_action = int(action)
                self.prob = probs[self.action]
            else:
                env_action = funcs.get_random_max_index(probs, invalid_actions)

        elif isinstance(self.config.action_space, ArrayContinuousSpace):
            act_space = self.config.action_space
            # --- 連続
            # パラメータ
            mean, stddev = self.parameter.get_normal(self.state)

            if self.training:
                # ガウス分布に従った乱数を出す
                self.action = env_action = mean + np.random.normal() * stddev
            else:
                env_action = mean

            # -inf～infの範囲を取るので実際に環境に渡すアクションはlowとhighで切り取る
            # 本当はポリシーが変化しちゃうのでよくない（暫定対処）
            env_action = np.clip(env_action, act_space.low[0], act_space.high[0])
            env_action = [env_action]  # list float

        return env_action

    def on_step(self, worker):
        if not self.training:
            return
        self.history.append(
            [
                self.state,
                self.action,
                worker.invalid_actions,
                worker.reward,
            ]
        )

        if worker.done:
            reward = 0
            for h in reversed(self.history):
                reward = h[3] + self.config.discount * reward
                batch = {
                    "state": h[0],
                    "action": h[1],
                    "invalid_actions": h[2],
                    "reward": reward,
                }
                self.memory.add(batch)

    def render_terminal(self, worker, **kwargs) -> None:
        if isinstance(self.config.action_space, DiscreteSpace):
            probs = self.parameter.get_probs(self.state, worker.invalid_actions)
            vals = [0 if v is None else v for v in self.parameter.policy[self.state]]
            maxa = np.argmax(vals)

            def _render_sub(action: int) -> str:
                return f"{probs[action] * 100:5.1f}% ({vals[action]:.5f})"

            funcs.render_discrete_action(int(maxa), self.config.action_space, worker.env, _render_sub)

        elif isinstance(self.config.action_space, ArrayContinuousSpace):
            mean, stddev = self.parameter.get_normal(self.state)
            print(f"mean {mean:.5f}, stddev {stddev:.5f}")
