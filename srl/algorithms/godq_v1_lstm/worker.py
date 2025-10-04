import logging
import random
from collections import deque
from itertools import islice

import numpy as np

from srl.base.rl.worker import RLWorkerGeneric
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
        self.net = self.parameter.net
        self.abort_count = 0

    def on_reset(self, worker):
        self.int_reward = 0  # render
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

        if self.config.enable_int_q and self.config.enable_int_episodic:
            self.episodic_memory = deque(maxlen=self.config.episodic_memory_capacity)

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
        # TODO
        if False and self.config.enable_archive:
            self.parameter.archive.on_reset(worker, self)
            self.search_step = 0

    def policy(self, worker) -> int:
        if self.policy_mode == "go":
            if random.random() < 0.1:
                self.go_action = self.sample_action()
            return self.go_action

        # --- add episodic
        if self.config.enable_int_q and self.config.enable_int_episodic:
            epi_reward = self.calc_episodic_reward(self.oe, add=True, calc=True)

        if self.policy_mode == "int":
            epsilon = self.config.epsilon if self.training else self.config.test_epsilon

            if self.config.enable_int_episodic:
                # 同じ場所を訪れる毎にランダムを上げる
                epsilon = np.clip(1 - np.sqrt(epi_reward), 0, 0.2)

            if random.random() < epsilon:
                return self.sample_action()

            # --- q int
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
            q = self.q.copy()
            q[worker.invalid_actions] = -np.inf
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
        # TODO
        if False and (not worker.done) and (self.config.enable_archive):
            self.search_step += 1
            if (self.config.search_max_step > 0) and (self.search_step > self.config.search_max_step):
                worker.abort_episode()
                self.abort_count += 1
                self.info["abort"] = self.abort_count

        # --- archive
        # TODO
        if False and self.config.enable_archive:
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
        if self.config.enable_q_rescale:
            q_ext = inverse_linear_symlog(self.q)
            v_ext = inverse_linear_symlog(self.v)
        print("--- q")
        print(f"     V: {v_ext:.7f}")
        if self.config.enable_q_distribution:
            adv_dist = self.net.q_online.get_distribution(self.oe)
            adv_mean = adv_dist.mean().detach().cpu().numpy()[0]
            adv_stdev = adv_dist.stddev().detach().cpu().numpy()[0]

        def _render_sub(a: int) -> str:
            s = f"{q_ext[a]:6.3f} adv:{q_ext[a] - v_ext:6.3f}"
            if self.config.enable_q_distribution:
                s += f"|adv({adv_mean[a]:6.3f}, {adv_stdev[a]:5.3f})"
            return s

        worker.print_discrete_action_info(int(np.argmax(self.q)), _render_sub)

        # --- q int
        if self.config.enable_int_q:
            print("--- q int")
            q_int, v_int = self.net.pred_q_int(self.oe, is_mean=True)
            q_int = q_int[0]
            v_int = v_int[0][0]

            print(f" int_reward     : {float(self.int_reward):.5f}")

            if self.config.enable_int_episodic:
                epi_reward = self.calc_episodic_reward(self.oe)
                e = 1 - np.sqrt(epi_reward)
                print(f" episodic_reward: {float(epi_reward):.5f} ({e:.5f})")

            print(f" V: {v_int:.7f}")
            if self.config.enable_q_distribution:
                adv_dist = self.parameter.net.q_int_online.get_distribution(self.oe)
                adv_mean = adv_dist.mean().detach().cpu().numpy()[0]
                adv_stdev = adv_dist.stddev().detach().cpu().numpy()[0]

            def _render_sub2(a: int) -> str:
                s = f"{q_int[a]:6.3f} adv:{q_int[a] - v_int:6.3f}"
                if self.config.enable_q_distribution:
                    s += f"|adv({adv_mean[a]:6.3f},{adv_stdev[a]:5.3f})"
                return s

            worker.print_discrete_action_info(int(np.argmax(q_int)), _render_sub2)

            # --- rate
            q_ext_rate = softmax(self.q - np.mean(self.q))
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
