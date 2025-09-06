import logging
import math
import random

import numpy as np

from srl.base.rl.algorithms.base_dqn import RLWorker
from srl.base.rl.worker_run import WorkerRun

from .config import Config
from .memory import Memory
from .parameter import Parameter

logger = logging.getLogger(__name__)


def symlog_scalar(x: float, shift: float = 1) -> float:
    if -shift <= x <= shift:
        return x
    return math.copysign(math.log1p(abs(x) - shift) + shift, x)


def plot_symlog_scalar():
    import matplotlib.pyplot as plt

    def symlog(x):
        return np.sign(x) * np.log(1 + np.abs(x))

    x_vals = np.linspace(-5, 5, 10000)
    y_scalar = np.array([symlog_scalar(x) for x in x_vals])
    y_symlog = np.array([symlog(x) for x in x_vals])

    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y_scalar, label="symlog_scalar")
    plt.plot(x_vals, y_symlog, label="symlog")
    plt.axvline(-1, color="gray", linestyle=":")
    plt.axvline(1, color="gray", linestyle=":")
    plt.title("Scalar Symlog Function")
    plt.xlabel("x")
    plt.ylabel("output")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


class Worker(RLWorker[Config, Parameter, Memory]):
    def on_setup(self, worker, context) -> None:
        self.screen = None
        self.np_dtype = self.config.get_dtype("np")
        self.abort_count = 0
        worker.set_tracking_max_size(self.config.max_discount_steps)

    def on_reset(self, worker):
        self.oe = None
        self.q = None

        # --- arvhie(restoreの可能性あり)
        if self.config.enable_archive:
            self.parameter.archive.on_reset(worker, self)
            self.search_step = 0

        # --- policy
        if self.training:
            if self.train_count == 0:
                self.policy_mode = "go"
                self.go_action = self.sample_action()
            elif self.config.enable_int_q:
                self.policy_mode = "int"
            else:
                self.policy_mode = "q"
        else:
            self.policy_mode = self.config.test_policy

    def set_q_oe(self, state: np.ndarray, set_q: bool = True, set_oe: bool = True, is_mean: bool = False):
        if set_oe:
            self.oe = self.parameter.net.pred_oe(state[np.newaxis, ...])
        if set_q:
            if self.oe is None:
                self.oe = self.parameter.net.pred_oe(state[np.newaxis, ...])
            self.q = self.parameter.net.pred_q(self.oe, is_mean)
            self.q = self.q[0]

    def policy(self, worker) -> int:
        if self.policy_mode == "go":
            if random.random() < 0.1:
                self.go_action = self.sample_action()
            return self.go_action
        elif self.policy_mode == "int":
            epsilon = self.config.epsilon if self.training else self.config.test_epsilon
            if random.random() < epsilon:
                return self.sample_action()

            rate = random.random() * 1

            self.set_q_oe(worker.state, is_mean=not self.training)
            assert self.oe is not None
            q_int = self.parameter.net.pred_q_int(self.oe)[0].copy()
            q = self.q + q_int * rate
            q[worker.invalid_actions] = -np.inf
            return int(np.argmax(q))
        else:  # "q"
            epsilon = self.config.epsilon if self.training else self.config.test_epsilon
            if random.random() < epsilon:
                return self.sample_action()
            self.set_q_oe(worker.state, is_mean=not self.training)
            assert self.q is not None
            q = self.q.copy()
            q[worker.invalid_actions] = -np.inf
            return int(np.argmax(q))

    def on_step(self, worker):
        prev_q = self.q
        self.oe = None
        self.q = None

        if self.rendering and self.config.enable_int_q:
            self.int_reward = self.parameter.net.pred_single_int_reward(
                worker.state,
                worker.action,
                worker.next_state,
            )

        if not self.training:
            return

        # --- abort
        # restore機構がないとstepが進まないのでarchiveが有効の場合のみ実行
        if (not worker.done) and (self.config.enable_archive):
            self.search_step += 1
            if (self.config.search_max_step > 0) and (self.search_step > self.config.search_max_step):
                worker.abort_episode()
                self.abort_count += 1
                self.info["abort"] = self.abort_count

        # --- archive
        if self.config.enable_archive:
            if self.parameter.archive.on_step(worker, self):
                # 新しいcellを見つけたらリセット
                self.search_step = 0

        # --- add memory
        reward = worker.reward
        if self.config.enable_reward_symlog_scalar:
            reward = float(symlog_scalar(reward))
        worker.add_tracking(
            {
                "state": worker.state,
                "n_state": worker.next_state,
                "action": worker.action,
                "reward": reward,
                "not_done": int(not worker.terminated),
                "priority": self._calc_priority(worker, reward, prev_q),
            }
        )
        if not worker.done:
            if worker.get_tracking_length() == self.config.max_discount_steps:
                total_reward = 0
                for b in reversed(worker.get_trackings()):
                    total_reward = b[3] + self.config.discount * total_reward
                self.memory.add_q(b[:-1] + [total_reward], b[-1])
        else:
            total_reward = 0
            for b in reversed(worker.get_trackings()):
                total_reward = b[3] + self.config.discount * total_reward
                self.memory.add_q(b[:-1] + [total_reward], b[-1])

    def _calc_priority(self, worker: WorkerRun, reward: float, prev_q):
        if not self.config.memory.requires_priority():
            return None

        # next q(簡易版としてqがあればそのまま使う(targetではない))
        # 分散の場合は計算
        if self.distributed:
            if prev_q is None:
                state = self.config.observation_space.rescale_from(worker.state, -1, 1)
                prev_oe = self.parameter.net.pred_oe(state[np.newaxis, ...])
                prev_q = self.parameter.net.pred_q(prev_oe, is_mean=False)
                prev_q = prev_q[0]

        if prev_q is None:
            return None

        select_q = prev_q[worker.action]
        target_q = reward
        if not worker.terminated:
            if self.q is None:
                return None
            n_max_q = np.max(self.q)
            target_q += self.config.discount * n_max_q
        priority = abs(target_q - select_q)
        return priority

    def render_terminal(self, worker, **kwargs):
        self.parameter.net.render_terminal(worker)
        if self.config.enable_int_q:
            if worker.step_in_episode > 0:
                print(f"int_reward: {float(self.int_reward):.5f}")
            else:
                print()

        if self.config.enable_archive:
            self.parameter.archive.render_terminal(worker)
