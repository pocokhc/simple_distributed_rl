import logging
import math
import random

import numpy as np

from srl.base.rl.algorithms.base_dqn import RLWorker

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
        self.abort_count = 0
        self.np_dtype = self.config.get_dtype("np")

    def on_reset(self, worker):
        self.oe = None

        # --- arvhie(restoreの可能性あり)
        if self.training and self.config.enable_archive:
            self.parameter.archive.on_reset(worker, self)
            self.search_step = 0

        # --- encode, q
        if self.oe is None:
            self.oe, self.q = self.parameter.net.pred_q(worker.state[np.newaxis, ...])
            self.q = self.q[0]

    def policy(self, worker) -> int:
        # e-greedy
        epsilon = self.config.epsilon if self.training else self.config.test_epsilon
        if random.random() < epsilon:
            return self.sample_action()

        q = self.q
        q[worker.invalid_actions] = -np.inf
        action = int(np.argmax(q))

        return action

    def on_step(self, worker):
        # --- encode, q
        prev_q = self.q
        self.oe, self.q = self.parameter.net.pred_q(worker.next_state[np.newaxis, ...])
        self.q = self.q[0]
        self.oz = None

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
            if self.oz is None:
                self.oz = self.parameter.net.encode_latent(self.oe)
            if self.parameter.archive.on_step(self.oz, worker):
                # 新しいcellを見つけたらリセット
                self.search_step = 0

        # --- add memory
        reward = worker.reward
        if self.config.enable_reward_symlog_scalar:
            reward = symlog_scalar(reward)
        if self.config.memory.requires_priority():
            select_q = prev_q[worker.action]

            # next q(簡易版としてqをそのまま使う)
            target_q = reward
            if not worker.terminated:
                n_max_q = np.max(self.q)
                target_q += self.config.discount * n_max_q
            priority = abs(target_q - select_q)
        else:
            priority = None
        self.memory.add_q(
            [
                worker.state,
                worker.action,
                worker.next_state,
                float(reward),
                1 - int(worker.terminated),
            ],
            priority=priority,
        )

    def render_terminal(self, worker, **kwargs):
        if self.config.enable_archive:
            if self.oe is not None:
                oz = self.parameter.net.encode_latent(self.oe)
                self.parameter.archive.render_terminal(oz)
        self.parameter.net.render_terminal(worker)

    def render_rgb_array(self, worker, **kwargs):
        if not self.config.enable_archive:
            return None

        from srl.utils.pygame_wrapper import PygameScreen

        WIDTH = 600
        HEIGHT = 400

        if self.screen is None:
            self.screen = PygameScreen(WIDTH, HEIGHT)
        self.screen.draw_fill((0, 0, 0))

        if self.config.enable_archive:
            if self.oe is not None:
                oz = self.parameter.net.encode_latent(self.oe)
                self.parameter.archive.render_rgb_array(self.screen, oz, 0, 0)

        return self.screen.get_rgb_array()
