import copy
import logging
import math
import pickle
import random
import time
from dataclasses import dataclass
from textwrap import shorten
from typing import Any, Dict, List

import numpy as np

from srl.base.exception import UndefinedError
from srl.base.rl.algorithms.base_ql import RLConfig, RLWorker
from srl.base.rl.parameter import RLParameter
from srl.base.rl.registration import register
from srl.base.rl.trainer import RLTrainer
from srl.rl import functions as funcs
from srl.rl.memories.single_use_buffer import RLSingleUseBuffer

logger = logging.getLogger(__name__)


@dataclass
class Config(RLConfig):
    search_rate: float = 0.5
    test_search_rate: float = 0.01

    #: 近似モデルの学習時のEMAの割合
    int_reward_ema_rate: float = 0.5

    q_iter_interval_num: int = 1_000
    q_iter_interval_max: int = 100_000
    q_iter_threshold: float = 0.001
    q_iter_timeout: float = 10

    #: 学習時にQ値policyを実行する確率
    train_q_policy_rate: float = 0.01

    #: アクション選択におけるUCBのペナルティ項の反映率
    action_ucb_penalty_rate: float = 0.5
    #: セル選択におけるUCBのQ値の反映率
    cell_select_ucb_penalty_rate: float = 1.0

    #: 外部報酬の割引率
    q_ext_discount: float = 0.9
    #: 内部報酬の割引率
    q_int_discount: float = 0.9
    #: 外部報酬の学習率
    q_ext_lr: float = 0.1
    #: 内部報酬の学習率
    q_int_lr: float = 0.1
    #: 外部報酬の目標方策で、最大価値を選ぶ確率
    q_ext_target_policy: float = 1.0
    #: 内部報酬の目標方策で、最大価値を選ぶ確率
    q_int_target_policy: float = 0.9

    explore_max_step: int = 50

    #: lifelong rewardの減少率
    lifelong_decrement_rate: float = 0.999

    def get_name(self) -> str:
        return "GoDynaQ"


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
        # [state][action][next_state] = [trans(count), reward, done, int_reward]
        self.mdp: Dict[str, List[Dict[str, List]]] = {}
        self.mdp_size = 0  # for info

        # [state]
        self.invalid_actions = {}

        # [state][action]
        self.q_ext: Dict[str, List[float]] = {}
        self.q_int: Dict[str, List[float]] = {}
        self.action_count = {}

        self.q_ext_min = math.inf
        self.q_ext_max = -math.inf
        self.q_int_max = 0

        # [state]
        self.lifelong = {}

        # --- archive
        # [start_state][state_action] = 1
        self.archive_reached_area: Dict[str, dict] = {}
        # [state] = {}
        self.archive_state: Dict[str, dict] = {}
        # [state_action] = {}
        self.archive_action: Dict[str, dict] = {}

    def update_archive(self, archive_reached_area, archive_state, archive_action):
        for start_state, key_dict in archive_reached_area.items():
            if start_state not in self.archive_reached_area:
                self.archive_reached_area[start_state] = {}
            for k, v in key_dict.items():
                self.archive_reached_area[start_state][k] = v
        for state, info in archive_state.items():
            # とりあえず最新ので更新
            self.archive_state[state] = copy.deepcopy(info)
        for key, info in archive_action.items():
            # とりあえず最新ので更新
            self.archive_action[key] = copy.deepcopy(info)

    def call_restore(self, data: Any, **kwargs) -> None:
        d = pickle.loads(data)
        self.mdp = d[0]
        self.mdp_size = d[1]
        self.invalid_actions = d[2]
        self.q_ext = d[3]
        self.q_int = d[4]
        self.action_count = d[5]
        self.q_ext_min = d[6]
        self.q_ext_max = d[7]
        self.q_int_max = d[8]
        self.lifelong = d[9]
        self.archive_reached_area = d[10]
        self.archive_state = d[11]
        self.archive_action = d[12]

    def call_backup(self, **kwargs):
        return pickle.dumps(
            [
                self.mdp,
                self.mdp_size,
                self.invalid_actions,
                self.q_ext,
                self.q_int,
                self.action_count,
                self.q_ext_min,
                self.q_ext_max,
                self.q_int_max,
                self.lifelong,
                self.archive_reached_area,
                self.archive_state,
                self.archive_action,
            ]
        )

    def init_model(self, state: str, action: int, n_state: str, invalid_actions, next_invalid_actions):
        if state not in self.mdp:
            self.mdp[state] = [{} for a in range(self.config.action_space.n)]
            self.invalid_actions[state] = invalid_actions
        if (n_state is not None) and (n_state not in self.mdp[state][action]):
            self.mdp[state][action][n_state] = [0, 0.0, 0.0, 1.0]
            self.mdp_size += 1
            self.invalid_actions[n_state] = next_invalid_actions

    def init_q(self, state: str):
        if state not in self.q_ext:
            self.q_ext[state] = [0.0 for a in range(self.config.action_space.n)]
            self.q_int[state] = [0.0 for a in range(self.config.action_space.n)]
            self.action_count[state] = [0 for a in range(self.config.action_space.n)]
            self.lifelong[state] = 1.0

    def iteration_q(
        self,
        mode: str,
        threshold: float,
        timeout: float,
    ):
        if mode == "ext":
            discount = self.config.q_ext_discount
            q_tbl = self.q_ext
            reward_idx = 1
            target_policy = self.config.q_ext_target_policy
            self.q_ext_min = math.inf
            self.q_ext_max = -math.inf
        elif mode == "int":
            discount = self.config.q_int_discount
            q_tbl = self.q_int
            reward_idx = 3
            target_policy = self.config.q_int_target_policy
            self.q_int_max = 0
        else:
            raise UndefinedError(mode)

        all_states = []
        for state in self.mdp.keys():
            if state not in q_tbl:
                q_tbl[state] = [-math.inf if a in self.invalid_actions[state] else 0.0 for a in range(self.config.action_space.n)]
            for act in range(self.config.action_space.n):
                if act in self.invalid_actions[state]:
                    continue
                N = sum([c[0] for c in self.mdp[state][act].values()])
                if N == 0:
                    continue

                _next_states = []
                for next_state, c in self.mdp[state][act].items():
                    trans = c[0]
                    if trans == 0:
                        continue
                    _next_states.append(
                        [
                            next_state,
                            trans / N,
                            c[reward_idx],
                            c[2],  # done
                        ]
                    )
                if len(_next_states) == 0:
                    continue
                all_states.append([state, act, _next_states])

        delta = 0
        count = 0
        t0 = time.time()
        while time.time() - t0 < timeout:  # for safety
            delta = 0
            for state, act, next_states in all_states:
                q = 0
                for next_state, trans_prob, reward, done in next_states:
                    n_q = self.calc_q(q_tbl, next_state, target_policy)
                    q += trans_prob * (reward + (1 - done) * discount * n_q)
                delta = max(delta, abs(q_tbl[state][act] - q))
                q_tbl[state][act] = q
                count += 1
            if delta < threshold:
                break
        else:
            logger.info(f"iteration timeout(delta={delta}, threshold={threshold}, count={count})")

        # update range
        for state, act, next_states in all_states:
            if mode == "ext":
                # ifよりmin,maxを使う方が少し早い
                self.q_ext_min = min(self.q_ext_min, self.q_ext[state][act])
                self.q_ext_max = max(self.q_ext_max, self.q_ext[state][act])
            elif mode == "int":
                self.q_int_max = max(self.q_int_max, self.q_int[state][act])

    def calc_q(self, q_tbl: Dict[str, List[float]], state: str, prob: float) -> float:
        if state not in q_tbl:
            return 0

        invalid_actions = self.invalid_actions[state]
        if self.config.action_space.n == len(invalid_actions):
            return 0

        q_max = max(q_tbl[state])
        if prob == 1:
            return q_max

        q_max_idx = [a for a, q in enumerate(q_tbl[state]) if q == q_max]
        valid_act_num = self.config.action_space.n - len(invalid_actions)
        if valid_act_num == len(q_max_idx):
            return q_max

        n_q = 0
        for a in range(self.config.action_space.n):
            if a in invalid_actions:
                continue
            if a in q_max_idx:
                p = prob / len(q_max_idx)
            else:
                p = (1 - prob) / (valid_act_num - len(q_max_idx))
            n_q += p * q_tbl[state][a]
        return n_q


class Trainer(RLTrainer[Config, Parameter, Memory]):
    def on_setup(self) -> None:
        self.interval = self.config.q_iter_interval_num
        self.iteration_num = 0

    def train(self) -> None:
        batches = self.memory.sample()
        if batches is None:
            return

        for batch in batches:
            (
                state,
                n_state,
                action,
                reward,
                episodic_reward,
                done,
                invalid_actions,
                next_invalid_actions,
            ) = batch

            self.parameter.init_model(state, action, n_state, invalid_actions, next_invalid_actions)
            self.parameter.init_q(state)
            self.parameter.init_q(n_state)

            # --- mdp update
            m = self.parameter.mdp[state][action][n_state]
            # trans count
            m[0] += 1
            # reward (online mean)
            m[1] += (reward - m[1]) / m[0]
            # done (online mean)
            m[2] += (done - m[2]) / m[0]

            # --- int reward (discount online mean)
            # update lifelong
            self.parameter.lifelong[state] *= self.config.lifelong_decrement_rate
            int_reward = self.parameter.lifelong[n_state] * episodic_reward
            # int reward (ema)
            m[3] = self.config.int_reward_ema_rate * m[3] + (1 - self.config.int_reward_ema_rate) * int_reward

            # --- q ext (greedy)
            n_q = self.parameter.calc_q(self.parameter.q_ext, n_state, self.config.q_ext_target_policy)
            target_q = reward + (1 - done) * self.config.q_ext_discount * n_q
            td_error = target_q - self.parameter.q_ext[state][action]
            self.parameter.q_ext[state][action] += self.config.q_ext_lr * td_error

            self.parameter.q_ext_min = min(self.parameter.q_ext_min, self.parameter.q_ext[state][action])
            self.parameter.q_ext_max = max(self.parameter.q_ext_max, self.parameter.q_ext[state][action])

            # --- int(sarsa)
            n_q = self.parameter.calc_q(self.parameter.q_int, n_state, self.config.q_int_target_policy)
            target_q = int_reward + (1 - done) * self.config.q_int_discount * n_q
            td_error = target_q - self.parameter.q_int[state][action]
            self.parameter.q_int[state][action] += self.config.q_int_lr * td_error

            self.parameter.q_int_max = max(self.parameter.q_int_max, self.parameter.q_int[state][action])

            if self.train_count % self.interval == 0:
                self.parameter.iteration_q("ext", self.config.q_iter_threshold, self.config.q_iter_timeout)
                self.parameter.iteration_q("int", self.config.q_iter_threshold, self.config.q_iter_timeout)
                self.interval *= 2
                if self.interval > self.config.q_iter_interval_max:
                    self.interval = self.config.q_iter_interval_max
                self.iteration_num += 1

            # --- action count update
            if self.distributed:
                self.parameter.action_count[state][action] += 1

            self.train_count += 1
        self.info["q_iter"] = self.iteration_num
        self.info["mdp_size"] = self.parameter.mdp_size


class Worker(RLWorker[Config, Parameter, Memory]):
    def on_setup(self, worker, context):
        # 1step毎に更新したいのでparameterの同期は最初と最後
        self.archive_reached_area: Dict[str, dict] = copy.deepcopy(self.parameter.archive_reached_area)
        self.archive_state: Dict[str, dict] = copy.deepcopy(self.parameter.archive_state)
        self.archive_action: Dict[str, dict] = copy.deepcopy(self.parameter.archive_action)

        self.parameter.iteration_q("ext", self.config.q_iter_threshold / 10, self.config.q_iter_timeout)
        self.parameter.iteration_q("int", self.config.q_iter_threshold / 10, self.config.q_iter_timeout)

    def on_teardown(self, worker):
        if self.training:
            # 学習最後にQテーブルを更新
            self.parameter.iteration_q("ext", self.config.q_iter_threshold / 10, self.config.q_iter_timeout * 10)
            self.parameter.iteration_q("int", self.config.q_iter_threshold / 10, self.config.q_iter_timeout * 10)

            self.parameter.update_archive(self.archive_reached_area, self.archive_state, self.archive_action)

    def on_reset(self, worker):
        self.policy_type = "q"
        self.start_state = self.config.observation_space.to_str(worker.state)
        self.move_actions = []

        if self.rendering:
            self.render_prev_state = self.config.observation_space.get_default()
            self.render_state = worker.state
            self.render_prev_action = 0
            self.render_action = 0
            self.render_prev_invalid_actions = []
            self.render_invalid_actions = []

        if not self.training:
            return

        self.episode_reward = 0
        self.explore_step = 0
        self.action_history = []
        self.action_history_deter = True

        self.episodic = {}

        state = self.config.observation_space.to_str(worker.state)
        self._archive_init(self.start_state, state, worker.invalid_actions)
        self._archive_state_update(state, 0, [], True)

        if random.random() < self.config.train_q_policy_rate:
            # アーカイブ更新用にQ値のルートも低確率で試す
            self.policy_type = "q"
        else:
            # アーカイブの中から目的地を探す
            cell = self._archive_select(self.start_state)
            if cell is not None:
                cell["select"] += 1
                self.policy_type = "ucb"  # アーカイブ後はUCBで探索
                cell_state = self.archive_state[cell["state"]]
                if cell_state["deter"]:
                    self.move_actions = cell_state["actions"][:]
                    self.move_actions.append(cell["action"])
                else:
                    # Q tbl を作成し、ターゲットの状態まで移動する
                    # けどQ tblの作成が重いので内部報酬で探索する
                    pass

    def policy(self, worker) -> int:
        state = self.config.observation_space.to_str(worker.state)
        invalid_actions = worker.invalid_actions
        self.parameter.init_q(state)

        # --- move
        if len(self.move_actions) > 0:
            act = self.move_actions.pop(0)

            # 無効なアクションの場合
            if act in invalid_actions:
                self.move_actions = []
            else:
                return act

        if self.policy_type == "ucb":
            # UCBベースでアクションを決定
            ucb_list = self._calc_policy_ucb(state, invalid_actions)
            action = funcs.get_random_max_index(ucb_list, invalid_actions)
        elif self.policy_type == "q":
            # Qテーブルベースでアクションを決定
            q_ext = np.array(self.parameter.q_ext[state])
            q_int = np.array(self.parameter.q_int[state])
            if self.parameter.q_ext_min < self.parameter.q_ext_max:
                q_ext = (q_ext - self.parameter.q_ext_min) / (self.parameter.q_ext_max - self.parameter.q_ext_min)
            if self.parameter.q_int_max > 0:
                q_int = q_int / self.parameter.q_int_max

            q = (1 - self.config.test_search_rate) * q_ext + self.config.test_search_rate * q_int
            action = funcs.get_random_max_index(q, invalid_actions)
        else:
            raise ValueError(self.policy_type)

        return action

    def _calc_policy_ucb(self, state, invalid_actions):
        N = sum(self.parameter.action_count[state])
        if N == 0:
            return [0.0 for _ in range(self.config.action_space.n)]

        N = 2 * math.log(N)
        ucb_list: List[float] = []
        for a in range(self.config.action_space.n):
            if a in invalid_actions:
                ucb_list.append(0)
                continue

            # ucb用に0～1に正規化
            q_int = self.parameter.q_int[state][a]
            if self.parameter.q_int_max > 0:
                q_int = q_int / self.parameter.q_int_max

            # 外部報酬が疎な場合は探索を優先
            if self.parameter.q_ext_min >= self.parameter.q_ext_max:
                q = q_int
            else:
                q_ext = self.parameter.q_ext[state][a]
                q_ext = (q_ext - self.parameter.q_ext_min) / (self.parameter.q_ext_max - self.parameter.q_ext_min)
                q = (1 - self.config.search_rate) * q_ext + self.config.search_rate * q_int

            # 1は保証して初めて実行する場合はqを信じる
            n = self.parameter.action_count[state][a] + 1
            ucb = q + self.config.action_ucb_penalty_rate * math.sqrt(N / n)
            ucb_list.append(ucb)

        return ucb_list

    def on_step(self, worker):
        if self.rendering:
            self.render_prev_state = self.render_state
            self.render_state = worker.state
            self.render_prev_action = self.render_action
            self.render_action = worker.action
            self.render_prev_invalid_actions = self.render_invalid_actions
            self.render_invalid_actions = worker.invalid_actions

        if not self.training:
            return
        state = self.config.observation_space.to_str(worker.state)
        n_state = self.config.observation_space.to_str(worker.next_state)

        # --- batch
        batch = [
            state,
            n_state,
            worker.action,
            worker.reward,
            self._calc_episodic_reward(n_state),
            1 if self.worker.terminated else 0,
            worker.invalid_actions,
            worker.next_invalid_actions,
        ]
        self.memory.add(batch)

        # --- update action_count
        self.parameter.action_count[state][worker.action] += 1

        # --- action history update
        self.action_history.append(worker.action)
        # deterチェック
        if self.action_history_deter:
            if state in self.parameter.mdp:
                m = self.parameter.mdp[state][worker.action]
                # 次の遷移先が1つならdeter
                if len(m.keys()) > 1:
                    self.action_history_deter = False

        # --- update archive
        self.episode_reward += worker.reward
        self._archive_init(self.start_state, state, worker.invalid_actions)
        self._archive_init(self.start_state, n_state, worker.next_invalid_actions)
        archive_state_is_update = self._archive_state_update(
            n_state,
            total_reward=self.episode_reward,
            action_history=self.action_history,
            action_history_deter=self.action_history_deter,
        )
        self._archive_action_update(state, worker.action, archive_state_is_update)

        if not worker.done:
            if len(self.move_actions) == 0:
                self.explore_step += 1
                if self.explore_step > self.config.explore_max_step:
                    worker.env.abort_episode()

    # ----------------------------------------------------------
    def _archive_init(self, start_state, state, invalid_actions):
        if start_state not in self.archive_reached_area:
            self.archive_reached_area[start_state] = {}

        if state not in self.archive_state:
            self.archive_state[state] = {
                "actions": [],
                "deter": True,
                "total_reward": -math.inf,
            }

        key = state + "_" + str(0)
        if key not in self.archive_action:
            for a in range(self.config.action_space.n):
                if a in invalid_actions:
                    continue
                akey = state + "_" + str(a)
                self.archive_reached_area[start_state][akey] = 1
                self.archive_action[akey] = {
                    "state": state,
                    "action": a,
                    "select": 0,
                    "visit": 0,
                }

    def _archive_action_update(self, state, action, is_update):
        action_cell = self.archive_action[state + "_" + str(action)]
        action_cell["visit"] += 1
        if is_update:
            action_cell["select"] = 0

    def _archive_state_update(
        self,
        n_state,
        total_reward,
        action_history,
        action_history_deter,
    ):
        n_state_cell = self.archive_state[n_state]

        # --- deter check
        if n_state_cell["deter"]:
            n_state_cell["deter"] = action_history_deter

        is_update = False
        if n_state_cell["total_reward"] < total_reward:
            is_update = True
        elif (n_state_cell["total_reward"] == total_reward) and (len(n_state_cell["actions"]) > len(action_history)):
            is_update = True
        if is_update:
            n_state_cell["total_reward"] = total_reward
            n_state_cell["actions"] = action_history[:]
        else:
            # updateしない場合はアーカイブのアクション履歴を採用
            self.action_history = self.archive_state[n_state]["actions"][:]
            if self.action_history_deter:
                self.action_history_deter = self.archive_state[n_state]["deter"]

        return is_update

    def _archive_select(self, start_state):
        if start_state not in self.archive_reached_area:
            return None
        if len(self.archive_reached_area[start_state]) == 0:
            return None

        N = 0
        arr = []
        for key in self.archive_reached_area[start_state].keys():
            if key not in self.archive_action:
                continue
            cell = self.archive_action[key]
            # 0回の場合はQ値を信じるために+1
            n = cell["visit"] + cell["select"] + 1
            arr.append((n, cell))
            N += n

        if N == 0:
            return None

        N = 2 * math.log(N)
        max_ucb = -math.inf
        max_cells = []
        for n, cell in arr:
            if cell["state"] in self.parameter.q_int:
                # --- calc policy q
                q_int = self.parameter.q_int[cell["state"]][cell["action"]]
                if self.parameter.q_int_max > 0:
                    q_int = q_int / self.parameter.q_int_max
                if self.parameter.q_ext_min >= self.parameter.q_ext_max:
                    q = q_int
                else:
                    q_ext = self.parameter.q_ext[cell["state"]][cell["action"]]
                    q_ext = (q_ext - self.parameter.q_ext_min) / (self.parameter.q_ext_max - self.parameter.q_ext_min)
                    q = (1 - self.config.search_rate) * q_ext + self.config.search_rate * q_int
            else:
                q = 0
            ucb = q + self.config.cell_select_ucb_penalty_rate * math.sqrt(N / n)
            if max_ucb < ucb:
                max_ucb = ucb
                max_cells = [cell]
            elif max_ucb == ucb:
                max_cells.append(cell)
        max_cell = random.choice(max_cells)
        return max_cell

    # ----------------------------------------------------------

    def _calc_episodic_reward(self, state, update: bool = True):
        if state not in self.episodic:
            self.episodic[state] = 1

        # 0回だと無限大になるので1回は保証する
        reward = 1 / math.sqrt(self.episodic[state] + 1)

        # 数える
        if update:
            self.episodic[state] += 1
        return reward

    # ----------------------------------------------------------

    def render_terminal(self, worker, **kwargs) -> None:
        # policy -> render -> env.step -> on_step

        prev_state = self.config.observation_space.to_str(self.render_prev_state)
        prev_act = self.render_prev_action
        state = self.config.observation_space.to_str(worker.state)
        self.parameter.init_model(prev_state, prev_act, state, self.render_prev_invalid_actions, worker.invalid_actions)
        self.parameter.init_q(prev_state)
        self.parameter.init_q(state)

        print(f"policy: {self.policy_type}")

        # --- MDP
        if prev_state in self.parameter.mdp:
            m = self.parameter.mdp[prev_state][prev_act]
            if state in m:
                N = sum([n[0] for n in m.values()])
                trans, reward, done, episodic_reward = m[state]
                p = 0 if N == 0 else trans / N
                print(f"trans {100 * p:3.1f}%({trans}/{N})")
                s = f"reward {reward:8.5f}"
                s += f", done {done:.1%}"
                s += f", episodic {episodic_reward:6.3f}"
                print(s)

        # --- q
        q_ext_nor = q_ext = np.array(self.parameter.q_ext[state])
        q_int_nor = q_int = np.array(self.parameter.q_int[state])
        if self.parameter.q_ext_min < self.parameter.q_ext_max:
            q_ext_nor = (q_ext_nor - self.parameter.q_ext_min) / (self.parameter.q_ext_max - self.parameter.q_ext_min)
        if self.parameter.q_int_max > 0:
            q_int_nor = q_int_nor / self.parameter.q_int_max
        q = (1 - self.config.test_search_rate) * q_ext + self.config.test_search_rate * q_int
        maxa = np.argmax(q)
        print(f"q_ext range[{self.parameter.q_ext_min:.3f}, {self.parameter.q_ext_max:.3f}]")
        print(f"q_int range[ 0.000, {self.parameter.q_int_max:.3f}]")
        ucb_list = self._calc_policy_ucb(state, worker.invalid_actions)

        def _render_sub(a: int) -> str:
            s = f"{q[a]:6.3f}: {q_ext[a]:6.3f}->{q_ext_nor[a]:6.3f}(ext)"
            s += f", {q_int[a]:6.3f}->{q_int_nor[a]:6.3f}(int)"
            s += f", {self.parameter.action_count[state][a]:4d}n"
            s += f", ucb {ucb_list[a]:.3f}"
            return s

        funcs.render_discrete_action(int(maxa), self.config.action_space, worker.env, _render_sub)

        # --- archive action
        key = prev_state + "_" + str(worker.action)
        print(f"archive action size={len(self.archive_action)}, action={worker.action}, key={shorten(key, width=50)}")
        if key in self.archive_action:
            cell = self.archive_action[key]
            print(f" select      : {cell['select']}")
            print(f" visit       : {cell['visit']}")

        # --- archive state
        print(f"archive state size={len(self.archive_state)}, state={shorten(str(state), width=50)}")
        if state in self.archive_state:
            cell = self.archive_state[state]
            print(f" total reward: {cell['total_reward']}")
            print(f" actions     : {cell['actions'][:20]}...")
            print(f" deter       : {cell['deter']}")
