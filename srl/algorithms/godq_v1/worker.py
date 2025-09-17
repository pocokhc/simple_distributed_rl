import logging
import random

import numpy as np

from srl.base.rl.worker import RLWorkerGeneric
from srl.base.rl.worker_run import WorkerRun
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

    def set_q_oe(self, state: list, set_q: bool = True, set_oe: bool = True, is_mean: bool = False):
        if set_oe:
            self.oe = self.parameter.net.pred_oe(state)
        if set_q:
            if self.oe is None:
                self.oe = self.parameter.net.pred_oe(state)
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
            q = self.q.copy()  # type: ignore
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

        # --- add diff memory
        if self.config.enable_diffusion and self.config.train_diffusion:
            self.memory.add_diff(
                [
                    worker.render_image_state,
                    worker.next_render_image_state,
                    worker.action,
                ]
            )

        # --- add memory
        if self.config.train_q:
            worker.add_tracking(
                {
                    "state": worker.state,
                    "n_state": worker.next_state,
                    "action": worker.action,
                    "reward": worker.reward,
                    "not_done": int(not worker.terminated),
                    "priority": self._calc_priority(worker, worker.reward, prev_q),
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
                prev_oe = self.parameter.net.pred_oe(worker.state)
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
        # --- q
        print("--- q")
        oe = self.parameter.net.pred_oe(worker.state)
        q, v = self.parameter.net.pred_q(oe, is_mean=True)
        q = q[0]
        v = v[0][0]
        print(f"     V: {v:.7f}")
        if self.config.enable_q_distribution:
            q_dist, v_dist, adv_dist = self.parameter.net.q_online.get_distribution(oe)
            v_mean = v_dist.mean().detach().cpu().numpy()[0][0]
            v_stdev = v_dist.stddev().detach().cpu().numpy()[0][0]
            adv_mean = adv_dist.mean().detach().cpu().numpy()[0]
            adv_stdev = adv_dist.stddev().detach().cpu().numpy()[0]
            q_mean = q_dist.mean().detach().cpu().numpy()[0]
            q_stdev = q_dist.stddev().detach().cpu().numpy()[0]
            print(f"dist V: {v_mean:.7f}(sigma: {v_stdev:.5f})")

        def _render_sub(a: int) -> str:
            s = f"{q[a]:6.3f} adv:{q[a] - v:6.3f}"
            if self.config.enable_q_distribution:
                s += f"|q({q_mean[a]:6.3f},{q_stdev[a]:5.3f})"
                s += f" adv({adv_mean[a]:6.3f},{adv_stdev[a]:5.3f})"
            return s

        worker.print_discrete_action_info(int(np.argmax(q)), _render_sub)

        # --- q int
        if self.config.enable_int_q:
            print("--- q int")
            q_int = self.parameter.net.pred_q_int(oe)[0]

            def _render_sub2(a: int) -> str:
                s = f"{q_int[a]:6.3f}"
                return s

            worker.print_discrete_action_info(int(np.argmax(q_int)), _render_sub2)

        # --- int q
        if self.config.enable_int_q:
            if worker.step_in_episode > 0:
                print(f"int_reward: {float(self.int_reward):.5f}")
            else:
                print()

        # --- archive
        if self.config.enable_archive:
            self.parameter.archive.render_terminal(worker)

    def render_rgb_array(self, worker, **kwargs):
        if not self.config.enable_diffusion:
            return None

        from srl.utils.pygame_wrapper import PygameScreen

        WIDTH = 700
        HEIGHT = 600

        if self.screen is None:
            self.screen = PygameScreen(WIDTH, HEIGHT)
        self.screen.draw_fill((0, 0, 0))

        if self.config.enable_diffusion:
            self.parameter.sampler.render_rgb_array(self.screen, worker)

        return self.screen.get_rgb_array()
