import logging
import random
from collections import deque
from itertools import islice

import numpy as np

from srl.base.rl.worker import RLWorkerGeneric
from srl.base.rl.worker_run import WorkerRun
from srl.base.spaces.discrete import DiscreteSpace
from srl.base.spaces.multi import MultiSpace
from srl.rl.functions import inverse_linear_symlog, softmax

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
        self.int_reward = 0  # render

        if self.config.enable_int_q and self.config.enable_int_episodic:
            self.episodic_memory = deque(maxlen=self.config.episodic_memory_capacity)

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
            self.q, _ = self.parameter.net.pred_q(self.oe, is_mean)
            self.q = self.q[0]

    def policy(self, worker) -> int:
        if self.policy_mode == "go":
            if random.random() < 0.1:
                self.go_action = self.sample_action()
            return self.go_action

        # --- add episodic
        if self.config.enable_int_q and self.config.enable_int_episodic:
            self.set_q_oe(worker.state, set_q=False, is_mean=not self.training)
            epi_reward = self.calc_episodic_reward(self.oe, add=True, calc=True)

        if self.policy_mode == "int":
            epsilon = self.config.epsilon if self.training else self.config.test_epsilon

            if self.config.enable_int_episodic:
                # 同じ場所を訪れる毎にランダムを上げる
                epsilon = np.clip(1 - np.sqrt(epi_reward), 0, 0.2)

            if random.random() < epsilon:
                return self.sample_action()

            # --- q int
            self.set_q_oe(worker.state, is_mean=not self.training)
            assert self.oe is not None
            assert self.q is not None
            q_int, _ = self.parameter.net.pred_q_int(self.oe)
            q_int = softmax(q_int[0])
            q_ext = softmax(self.q - np.mean(self.q))
            q = q_ext + self.config.int_rate * q_int
            q[worker.invalid_actions] = -np.inf
            return int(np.argmax(q))
        else:  # "q"
            epsilon = self.config.epsilon if self.training else self.config.test_epsilon
            if random.random() < epsilon:
                return self.sample_action()

            # --- q
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
                    self.memory.add_q(b[:-1] + [total_reward], b[-3])

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

    def calc_episodic_reward(self, oe, add: bool = False, calc: bool = True):
        # 制御可能状態を取得
        cont = self.parameter.net.emb_net.predict(oe)
        cont = cont.detach().cpu().numpy()[0]

        if add:
            self.episodic_memory.append(cont)
        if not calc:
            return 1

        if len(self.episodic_memory) == 0:
            return 1

        k = self.config.episodic_count_max
        epsilon = self.config.episodic_epsilon
        cluster_distance = self.config.episodic_cluster_distance

        # エピソードメモリ内の全要素とユークリッド距離を求める
        itr_memory = islice(self.episodic_memory, len(self.episodic_memory) - 1)
        euclidean_list = [np.linalg.norm(m - cont, ord=2) for m in itr_memory]

        # 近いk個を対象
        euclidean_list = np.sort(euclidean_list)[:k]

        # 上位k個の移動平均を出す
        mode_ave = np.mean(euclidean_list)
        if mode_ave == 0.0:
            # ユークリッド距離は正なので平均0は全要素0のみ
            dn = euclidean_list
        else:
            dn = euclidean_list / mode_ave  # 正規化

        # 一定距離以下を同じ状態とする
        dn = np.maximum(dn - cluster_distance, 0)

        # 訪問回数を計算(Dirac delta function の近似)
        dn = epsilon / (dn + epsilon)
        N = np.sum(dn)

        reward = 1 / np.sqrt(N + 1)
        return reward

    def render_terminal(self, worker, **kwargs):
        # --- q
        print("--- q")
        oe = self.parameter.net.pred_oe(worker.state)
        q, v = self.parameter.net.pred_q(oe, is_mean=True)
        q = q[0]
        v = v[0][0]
        if self.config.enable_q_rescale:
            q_ext = inverse_linear_symlog(q)
            v_ext = inverse_linear_symlog(v)
        print(f" V: {v_ext:.7f}")
        if self.config.enable_q_distribution:
            adv_dist = self.parameter.net.q_online.get_distribution(oe)
            adv_mean = adv_dist.mean().detach().cpu().numpy()[0]
            adv_stdev = adv_dist.stddev().detach().cpu().numpy()[0]

        def _render_sub(a: int) -> str:
            s = f"{q_ext[a]:6.3f} adv:{q_ext[a] - v_ext:6.3f}"
            if self.config.enable_q_distribution:
                s += f"|adv({adv_mean[a]:6.3f},{adv_stdev[a]:5.3f})"
            return s

        worker.print_discrete_action_info(int(np.argmax(q)), _render_sub)

        # --- q int
        if self.config.enable_int_q:
            print("--- q int")
            q_int, v_int = self.parameter.net.pred_q_int(oe, is_mean=True)
            q_int = q_int[0]
            v_int = v_int[0][0]

            print(f" int_reward     : {float(self.int_reward):.5f}")

            if self.config.enable_int_episodic:
                epi_reward = self.calc_episodic_reward(oe)
                e = 1 - np.sqrt(epi_reward)
                print(f" episodic_reward: {float(epi_reward):.5f} ({e:.5f})")

            print(f" V: {v_int:.7f}")
            if self.config.enable_q_distribution:
                adv_dist = self.parameter.net.q_int_online.get_distribution(oe)
                adv_mean = adv_dist.mean().detach().cpu().numpy()[0]
                adv_stdev = adv_dist.stddev().detach().cpu().numpy()[0]

            def _render_sub2(a: int) -> str:
                s = f"{q_int[a]:6.3f} adv:{q_int[a] - v_int:6.3f}"
                if self.config.enable_q_distribution:
                    s += f"|adv({adv_mean[a]:6.3f},{adv_stdev[a]:5.3f})"
                return s

            worker.print_discrete_action_info(int(np.argmax(q_int)), _render_sub2)

            # --- rate
            q_ext_rate = softmax(q - np.mean(q))
            q_int_rate = softmax(q_int)
            q_rate = q_ext_rate + q_int_rate
            print("          rate | ext + int")

            def _render_sub3(a: int) -> str:
                s = f"{q_rate[a] * 100:5.1f} | {q_ext_rate[a] * 100:5.1f} + {q_int_rate[a] * 100:5.1f}"
                return s

            worker.print_discrete_action_info(int(np.argmax(q_rate)), _render_sub3)

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
