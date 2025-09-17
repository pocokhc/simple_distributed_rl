import logging
import random

import numpy as np

from srl.base.rl.worker import RLWorkerGeneric
from srl.base.spaces.discrete import DiscreteSpace
from srl.base.spaces.multi import MultiSpace

from .config import Config
from .memory import Memory
from .parameter import Parameter

logger = logging.getLogger(__name__)


class Worker(RLWorkerGeneric[Config, Parameter, Memory, DiscreteSpace, int, MultiSpace, list]):
    def on_setup(self, worker, context) -> None:
        self.screen = None
        self.np_dtype = self.config.get_dtype("np")
        self.net = self.parameter.net
        self.abort_count = 0

    def on_reset(self, worker):
        worker.add_tracking(
            {
                "state": worker.state,
                "action": 0,
                "reward": 0,
                "not_terminated": 1,
                "not_start": 0,
                "not_done": True,
                "step": -1,  # debug
            }
        )

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

        # --- q
        if self.policy_mode != "go":
            self.hc = self.net.encoder.get_initial_state()
            self.oe, self.hc = self.net.pred_oe(worker.state, 0, self.hc)
            self.q, self.v = self.net.pred_q(self.oe)
            self.q = self.q[0]
            self.v = self.v[0][0]

        # --- arvhie(restoreの可能性あり)
        if self.config.enable_archive:
            self.parameter.archive.on_reset(worker, self)
            self.search_step = 0

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
            q_int = self.parameter.net.pred_q_int(self.oe)[0]
            q = self.q + q_int * rate
            q[worker.invalid_actions] = -np.inf
            return int(np.argmax(q))
        else:  # "q"
            epsilon = self.config.epsilon if self.training else self.config.test_epsilon
            if random.random() < epsilon:
                return self.sample_action()
            self.q[worker.invalid_actions] = -np.inf
            return int(np.argmax(self.q))

    def on_step(self, worker):
        if self.policy_mode != "go":
            pred_oe = self.oe
            self.oe, self.hc = self.net.pred_oe(worker.next_state, worker.action, self.hc)
            self.q, self.v = self.net.pred_q(self.oe)
            self.q = self.q[0]
            self.v = self.v[0][0]

        if self.rendering and self.config.enable_int_q:
            self.int_reward = self.parameter.net.pred_single_int_reward(
                pred_oe,
                self.oe,
                worker.action,
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
        worker.add_tracking(
            {
                "state": worker.next_state,
                "action": worker.action,
                "reward": worker.reward,
                "not_terminated": int(not worker.terminated),
                "not_start": 1,
                "not_done": not worker.done,
                "step": worker.step_in_episode,  # debug
            }
        )
        if worker.done:
            steps = worker.get_trackings()
            total_reward = 0
            for i in reversed(range(len(steps))):
                total_reward = steps[i][2] + self.config.discount * total_reward
                steps[i].append(total_reward)
            self.memory.add_steps(steps)

    def render_terminal(self, worker, **kwargs):
        # --- q
        print("--- q")
        print(f"     V: {self.v:.7f}")
        if self.config.enable_q_distribution:
            q_dist, v_dist, adv_dist = self.net.q_online.get_distribution(self.oe)
            v_mean = v_dist.mean().detach().cpu().numpy()[0][0]
            v_stdev = v_dist.stddev().detach().cpu().numpy()[0][0]
            adv_mean = adv_dist.mean().detach().cpu().numpy()[0]
            adv_stdev = adv_dist.stddev().detach().cpu().numpy()[0]
            q_mean = q_dist.mean().detach().cpu().numpy()[0]
            q_stdev = q_dist.stddev().detach().cpu().numpy()[0]
            print(f" V: {v_mean:.7f}(sigma: {v_stdev:.5f})")

        def _render_sub(a: int) -> str:
            s = f"{self.q[a]:6.3f}"
            if self.config.enable_q_distribution:
                s += f" q({q_mean[a]:6.3f}, {q_stdev[a]:6.3f})"
                s += f" adv({adv_mean[a]:6.3f}, {adv_stdev[a]:6.3f})"
            return s

        worker.print_discrete_action_info(int(np.argmax(self.q)), _render_sub)

        # --- q int
        if self.config.enable_int_q:
            print("--- q int")
            q_int = self.net.pred_q_int(self.oe)[0]

            def _render_sub2(a: int) -> str:
                s = f"{q_int[a]:6.3f}"
                return s

            worker.print_discrete_action_info(int(np.argmax(q_int)), _render_sub2)
            if worker.step_in_episode > 0:
                print(f"int_reward: {float(self.int_reward):.5f}")
            else:
                print()

        # --- archive
        if self.config.enable_archive:
            self.parameter.archive.render_terminal(worker)
