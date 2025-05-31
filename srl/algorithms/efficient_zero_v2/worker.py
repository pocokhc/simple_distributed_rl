import logging
import random
from typing import Any

import numpy as np

from srl.base.spaces.discrete import DiscreteSpace
from srl.base.spaces.np_array import NpArraySpace
from srl.rl import functions as funcs
from srl.rl.functions import inverse_rescaling, rescaling

from .config import Config, RLWorker
from .mcts import MCTS
from .memory import Memory
from .parameter import Parameter

logger = logging.getLogger(__name__)


def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax関数（数値的に安定な計算）"""
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))  # Why not: オーバーフロー防止のため最大値を引く
    return e / np.sum(e, axis=-1, keepdims=True)


class Worker(RLWorker[Config, Parameter, Memory]):
    def on_setup(self, worker, context) -> None:
        self.env_player_num = worker.env.player_num
        self.np_dtype = self.config.get_dtype("np")
        self.policy_tau_sch = self.config.policy_tau_scheduler.create(self.config.policy_tau)
        self.mcts = MCTS(self.config, self.parameter)
        self.root = None

    def on_reset(self, worker):
        self.history = []

    def policy(self, worker):
        # --- シミュレーションしてpolicyを作成
        root = self.mcts.simulation(worker.state, worker.invalid_actions, self.training)
        self.step_v = root.v
        if self.rendering:
            self.root = root

        # --- 選択回数に応じてアクションを選択
        if self.training:
            policy_tau = self.policy_tau_sch.update(self.step_in_training).to_float()
        else:
            policy_tau = self.config.test_policy_tau
        self.info["policy_tau"] = policy_tau

        node_cnt = np.array([n.visit_count for n in root.children])

        if policy_tau == 0:
            node_idx = np.random.choice(np.where(node_cnt == node_cnt.max())[0])
        else:
            probs = node_cnt ** (1 / policy_tau)
            probs /= node_cnt.sum()
            node_idx = funcs.random_choice_by_probs(probs)
        self.action: Any = root.children[node_idx].action

        # --- 学習用ポリシーを計算
        if isinstance(self.config.action_space, DiscreteSpace):
            if self.config.enable_gumbel_search:
                completed_q = []
                v = 0
                max_visit = 0
                for a in range(self.config.action_space.n):
                    node = root.get_children_for_action(a)
                    if node is None:
                        completed_q.append(None)
                    elif node.visit_count == 0:
                        completed_q.append(None)
                    else:
                        completed_q.append(node.q)
                        v += node.prior * node.q
                        max_visit = max(max_visit, node.visit_count)
                for a in range(self.config.action_space.n):
                    if completed_q[a] is None:
                        completed_q[a] = v
                completed_q = np.array(completed_q)
                completed_q = (self.config.c_visit + max_visit) * self.config.c_scale * completed_q
                self.step_policy = softmax(completed_q)
            else:
                # 学習用のpolicyはtau=1
                self.step_policy = node_cnt / node_cnt

            return int(self.action)
        elif isinstance(self.config.action_space, NpArraySpace):
            self.step_policy = None
            env_action = self.config.action_space.rescale_from(self.action, -1, 1)
            return env_action

    def on_step(self, worker):
        if not self.training:
            return

        # 正規化用Qを保存できるように送信(memory -> trainer -> parameter)
        self.memory.add_q(self.parameter.q_min, self.parameter.q_max)
        self.info["q_min"] = self.parameter.q_min
        self.info["q_max"] = self.parameter.q_max

        reward = worker.reward
        if self.config.enable_rescale:
            reward = rescaling(reward)

        self.history.append(
            {
                "state": worker.state,
                "action": self.action,
                "policy": self.step_policy,
                "reward": reward,
                "v": self.step_v,
            }
        )

        if worker.done:
            for _ in range(self.config.unroll_steps + 1):
                if isinstance(self.config.action_space, DiscreteSpace):
                    dummy_action = random.randint(0, self.config.action_space.n - 1)
                    dummy_policy = [1 / self.config.action_space.n for _ in range(self.config.action_space.n)]
                elif isinstance(self.config.action_space, NpArraySpace):
                    dummy_action = np.tanh(np.random.normal(size=(self.config.action_space.size,)))
                    dummy_policy = None
                self.history.append(
                    {
                        "state": worker.state,
                        "action": dummy_action,
                        "policy": dummy_policy,
                        "reward": 0,
                        "v": 0,
                    }
                )

            # --- calc discount reward
            reward = 0
            for h in reversed(self.history):
                reward = h["reward"] + self.config.discount * reward
                h["discount_reward"] = reward

                # twohot value
                h["twohot_z"] = funcs.twohot_encode(
                    h["discount_reward"],
                    self.config.value_range_num,
                    self.config.value_range[0],
                    self.config.value_range[1],
                    self.np_dtype,
                )

                # twohot reward
                h["twohot_value_prefix"] = funcs.twohot_encode(
                    h["discount_reward"],
                    self.config.reward_range_num,
                    self.config.reward_range[0],
                    self.config.reward_range[1],
                    self.np_dtype,
                )

            # --- add batch
            for idx in range(len(self.history) - self.config.unroll_steps - 1):
                batch = []
                priority = 0
                for i in range(self.config.unroll_steps + 1):
                    h = self.history[idx + i]
                    if not self.config.use_max_priority:
                        priority += abs(h["v"] - h["discount_reward"])
                    batch.append(
                        [
                            h["state"],
                            h["action"],
                            h["policy"],
                            h["twohot_z"],
                            h["twohot_value_prefix"],
                        ]
                    )
                if self.config.use_max_priority:
                    priority = None
                else:
                    priority /= self.config.unroll_steps + 1
                self.memory.add(batch, priority)

    def render_terminal(self, worker, **kwargs) -> None:
        if self.root is None:
            return

        v = float(self.root.v)
        if self.config.enable_rescale:
            v = inverse_rescaling(v)
        print(f"V: {v:.5f}")

        if isinstance(self.config.action_space, DiscreteSpace):
            s_state = self.parameter.representation_net(worker.state[np.newaxis, ...])
            policy = self.step_policy

            def _render_sub(a: int) -> str:
                hc = self.parameter.dynamics_net.get_initial_state()
                n_s_state, value_prefix, _ = self.parameter.dynamics_net.pred(s_state, np.array([a], self.np_dtype), hc)
                value_prefix = value_prefix[0]

                if self.config.enable_rescale:
                    value_prefix = inverse_rescaling(value_prefix)

                _, n_v = self.parameter.prediction_net.pred(n_s_state)
                n_v = n_v[0]

                s = f"{policy[a] * 100:5.1f}%"
                s += f" {value_prefix:6.3f}(value_prefix)"
                s += f" {n_v:6.3f}(V)"
                assert self.root is not None
                node = self.root.get_children_for_action(a)
                if node is not None:
                    s += f"({node.visit_count:3d})(N)"
                    s += f" {float(node.q):5.3f}(Q)"
                    s += f" {node.prior:6.3f}(P)"
                    s += f" {node.score:6.3f}(UCT)"
                return s

            worker.print_discrete_action_info(worker.action, _render_sub)

        elif isinstance(self.config.action_space, NpArraySpace):
            for node in self.root.children:
                s = f" {node.visit_count:3d}(N)"
                s += f" {node.q:5.3f}(Q)"
                s += f" {node.value_prefix:5.3f}(value_prefix)"
                s += f" {node.v:5.3f}(V)"
                s += f" {float(node.prior[0]):6.3f}(P)"
                s += f" {float(node.score):6.3f}(UCT)"
                s += f" {node.action}(action)"
                print(s)
