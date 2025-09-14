import logging
import random

import numpy as np

from srl.base.rl.algorithms.base_dqn import RLWorker

from .config import Config, Memory
from .parameter import Parameter

logger = logging.getLogger(__name__)


class Worker(RLWorker[Config, Parameter, Memory]):
    def on_setup(self, worker, context) -> None:
        self.epsilon_sch = self.config.epsilon_scheduler.create(self.config.epsilon)
        worker.set_tracking_max_size(self.config.max_discount_steps)

    def policy(self, worker) -> int:
        epsilon = self.epsilon_sch.update(self.step_in_training).to_float() if self.training else self.config.test_epsilon
        if random.random() < epsilon:
            return self.sample_action()
        q, _ = self.parameter.pred_q(worker.state[np.newaxis, ...])
        return int(np.argmax(q[0]))

    def on_step(self, worker):
        if not self.training:
            return

        worker.add_tracking(
            {
                "state": worker.state,
                "n_state": worker.next_state,
                "action": worker.action,
                "reward": worker.reward,
                "not_done": int(not worker.terminated),
            }
        )
        if not worker.done:
            if worker.get_tracking_length() == self.config.max_discount_steps:
                total_reward = 0
                for b in reversed(worker.get_trackings()):
                    total_reward = b[3] + self.config.discount * total_reward
                self.memory.add(b[:] + [total_reward])
        else:
            total_reward = 0
            for b in reversed(worker.get_trackings()):
                total_reward = b[3] + self.config.discount * total_reward
                self.memory.add(b[:] + [total_reward])

    def render_terminal(self, worker, **kwargs):
        q, v = self.parameter.pred_q(worker.state[np.newaxis, ...])
        q = q[0]
        v = v[0][0]
        print(f"V={v:.7f}")

        def _render_sub(a: int) -> str:
            return f"{q[a]:8.5f} (adv: {q[a] - v:8.5f})"

        worker.print_discrete_action_info(int(np.argmax(q)), _render_sub)
