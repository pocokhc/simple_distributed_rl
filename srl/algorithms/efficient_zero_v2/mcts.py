import logging
import random
from typing import List

import numpy as np

from .config import Config
from .parameter import Parameter

logger = logging.getLogger(__name__)


class Node:
    def __init__(self, action, prior, is_root=False):
        self.action = action
        self.prior = prior
        self.is_root = is_root
        self.children: List[Node] = []
        self.visit_count: int = 0
        self.value_prefix: float = 0.0
        self.q_sum: float = 0.0
        self.score: float = 0.0
        self.v = 0.0
        self.s_state = None
        self.hc = None

    @property
    def q(self) -> float:
        return self.q_sum / self.visit_count if self.visit_count > 0 else 0.0

    def expand(self, p_dist, cfg: Config, invalid_actions: list = []) -> None:
        if cfg.action_space.name == "Discrete":
            probs = p_dist.probs()[0].numpy().tolist()

            if cfg.enable_gumbel_search:
                # アクション数がtopk以下ならそのまま
                if len(probs) == cfg.num_top_actions:
                    selected_acts = [i for i in range(len(probs))]
                elif self.is_root:
                    # 離散の場合の記載がなかったので、半分をgumbel、半分を一様分布で取得
                    num_acts_half = cfg.num_top_actions // 2
                    d_acts = p_dist.sample_topk(num_acts_half)[0].numpy().tolist()
                    _choice_acts = [a for a in range(len(probs)) if a not in d_acts]
                    r_acts = random.sample(_choice_acts, k=num_acts_half)
                    selected_acts = d_acts + r_acts
                else:
                    selected_acts = p_dist.sample_topk(cfg.num_top_actions)[0].numpy().tolist()
            else:
                selected_acts = [i for i in range(len(probs))]
            selected_probs = [0 if a in invalid_actions else probs[a] for a in selected_acts]
        else:
            selected_acts = []
            selected_probs = []
            if self.is_root:
                num_acts_half = cfg.num_top_actions // 2
                for _ in range(num_acts_half):
                    act, logprob = self._calc_rsample_logprob(p_dist)
                    selected_acts.append(act[0])
                    selected_probs.append(np.exp(logprob[0]))
                p_dist.increase_variance(0.1)
                for _ in range(num_acts_half):
                    act, logprob = self._calc_rsample_logprob(p_dist)
                    selected_acts.append(act[0])
                    selected_probs.append(np.exp(logprob[0]))
            else:
                for _ in range(cfg.num_top_actions):
                    act, logprob = self._calc_rsample_logprob(p_dist)
                    selected_acts.append(act[0])
                    selected_probs.append(np.exp(logprob[0]))

        self.children = [Node(act, prior) for act, prior in zip(selected_acts, selected_probs)]

    def _calc_rsample_logprob(self, p_dist):
        action = p_dist.rsample()
        logpi = p_dist.log_prob_sgp(action)
        return action, logpi

    def get_children_for_action(self, action: int):
        for n in self.children:
            if n.action == action:
                return n
        return None


class MCTS:
    def __init__(self, config: Config, parameter: Parameter) -> None:
        self.cfg = config
        self.parameter = parameter
        self.np_dtype = config.get_dtype("np")

    def simulation(self, root_state: np.ndarray, invalid_actions: list, training: bool):
        # --- root情報
        root_s_state = self.parameter.representation_net(root_state[np.newaxis, ...])
        p_dist, v = self.parameter.prediction_net.pred(root_s_state)
        root = Node(action=0, prior=0.0, is_root=True)
        root.s_state = root_s_state
        root.hc = self.parameter.dynamics_net.get_initial_state()
        root.v = v[0]
        root.expand(p_dist, self.cfg, invalid_actions)

        for _ in range(self.cfg.num_simulations):
            # --- 子ノードまで降りる
            node = root
            search_path = [node]
            while node.children:
                node_idx = self._select_node(node, training)
                node = node.children[node_idx]
                search_path.append(node)

            # --- expand
            parent = search_path[-2]
            action_tf = np.array([node.action], dtype=self.np_dtype)
            s_state, value_prefix, hc = self.parameter.dynamics_net.pred(parent.s_state, action_tf, parent.hc)
            p_dist, v = self.parameter.prediction_net.pred(s_state)
            node.s_state = s_state
            node.hc = hc
            node.value_prefix = value_prefix[0]
            node.v = v[0]
            node.expand(p_dist, self.cfg)

            # --- backup
            # v = node.v  # MC法なのでvを使っていない
            v = 0
            for node in reversed(search_path):
                node.visit_count += 1
                node.q_sum += node.value_prefix + v

                # 正規化用
                self.parameter.q_min = min(self.parameter.q_min, node.q)
                self.parameter.q_max = max(self.parameter.q_max, node.q)

        return root

    def _select_node(self, node: Node, training: bool):
        if node.is_root and training:
            noises = np.random.dirichlet([self.cfg.root_dirichlet_alpha] * len(node.children))
            e = self.cfg.root_exploration_fraction

        # mean Q value mechanism
        q_min = self.parameter.q_min
        q_max = self.parameter.q_max
        q_vals = [c.q for c in node.children if c.visit_count > 0] + [node.q]
        mean_q = sum(q_vals) / len(q_vals)

        N = node.visit_count
        c = np.log((1 + N + self.cfg.c_base) / self.cfg.c_base) + self.cfg.c_init
        scores = np.zeros(len(node.children))
        for idx, child in enumerate(node.children):
            n = child.visit_count
            p = child.prior
            q = mean_q if n == 0 else child.q

            # rootはディリクレノイズを追加
            if node.is_root and training:
                p = (1 - e) * p + e * noises[idx]

            # 過去観測したQ値で正規化(soft-MinMax)
            if q_min < q_max:
                q = (q - q_min) / max(q_max - q_min, self.cfg.soft_minmax_q_e)

            node.score = q + c * p * (np.sqrt(N) / (1 + n))
            scores[idx] = node.score

        node_idx = int(np.random.choice(np.where(scores == np.max(scores))[0]))
        return node_idx
