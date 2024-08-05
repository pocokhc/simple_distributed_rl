import logging
import pickle
import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from srl.base.rl.algorithms.base_ql import RLConfig, RLWorker
from srl.base.rl.parameter import RLParameter
from srl.base.rl.registration import register
from srl.base.rl.trainer import RLTrainer
from srl.rl import functions as funcs
from srl.rl.memories.sequence_memory import SequenceMemory

logger = logging.getLogger(__name__)


@dataclass
class Config(RLConfig):

    q_epsilon: float = 0.001
    test_epsilon: float = 0.0001

    ucb_scale: float = 0.1
    q_policy_prob: float = 1.0
    q_discount: float = 0.9
    q_lr: float = 0.1

    explore_rate: float = 0.5
    move_discount: float = 0.9
    move_policy_prob: float = 0.9
    explore_action_change_rate: float = 0.1
    explore_max_step: int = 10

    #: 方策反復法の学習完了の閾値
    iteration_threshold: float = 0.0001
    #: 方策反復法におけるタイムアウト
    iteration_timeout: float = 10
    #: 方策反復法を実行する間隔(学習回数)
    iteration_interval: int = 10_000

    def get_framework(self) -> str:
        return ""

    def get_name(self) -> str:
        return "GoDynaQ"

    def assert_params(self) -> None:
        super().assert_params()


register(
    Config(),
    __name__ + ":Memory",
    __name__ + ":Parameter",
    __name__ + ":Trainer",
    __name__ + ":Worker",
)


class Memory(SequenceMemory):
    pass


class Parameter(RLParameter[Config]):
    def __init__(self, *args):
        super().__init__(*args)

        # [state][action][next_state]
        self.trans = {}
        self.reward = {}
        self.done = {}
        # [state]
        self.invalid_actions = {}
        self.mdp_size = 0  # for info

        # [state][action]
        self.q_tbl = {}

    def call_restore(self, data: Any, **kwargs) -> None:
        d = pickle.loads(data)
        self.trans = d[0]
        self.reward = d[1]
        self.done = d[2]
        self.invalid_actions = d[3]
        self.mdp_size = d[4]
        self.q_tbl = d[5]

    def call_backup(self, **kwargs):
        return pickle.dumps(
            [
                self.trans,
                self.reward,
                self.done,
                self.invalid_actions,
                self.mdp_size,
                self.q_tbl,
            ]
        )

    def init_model(self, state, action, n_state, invalid_actions, next_invalid_actions):
        if state not in self.trans:
            n = self.config.action_space.n
            self.trans[state] = [{} for _ in range(n)]
            self.reward[state] = [{} for _ in range(n)]
            self.done[state] = [{} for _ in range(n)]
            self.invalid_actions[state] = invalid_actions
        if n_state is not None and n_state not in self.trans[state][action]:
            self.trans[state][action][n_state] = 0
            self.reward[state][action][n_state] = 0.0
            self.done[state][action][n_state] = 0.0
            self.invalid_actions[n_state] = next_invalid_actions
            self.mdp_size += 1

    def init_q(self, state):
        if state not in self.q_tbl:
            self.q_tbl[state] = [0 for _ in range(self.config.action_space.n)]

    def iteration_q(
        self,
        mode: str,
        target_state,
        discount: float,
        policy_prob: float,
        threshold: float = 0.0001,
        timeout: float = 1,
    ):
        if mode == "go_q":
            q_tbl = {}
        else:
            q_tbl = self.q_tbl

        all_states = []
        for state in self.trans.keys():
            if state not in q_tbl:
                q_tbl[state] = [0.0 for _ in range(self.config.action_space.n)]
            for action in range(self.config.action_space.n):
                if action in self.invalid_actions[state]:
                    continue
                N = sum(self.trans[state][action].values())
                if N == 0:
                    continue
                _next_states = []
                for next_state in self.trans[state][action].keys():
                    if self.trans[state][action][next_state] == 0:
                        continue
                    if next_state not in q_tbl:
                        q_tbl[next_state] = [0.0 for _ in range(self.config.action_space.n)]

                    done = self.done[state][action][next_state]
                    if mode == "move":
                        if target_state == next_state:
                            reward = 1
                        else:
                            reward = -done
                        _next_states.append(
                            [
                                next_state,
                                self.trans[state][action][next_state] / N,
                                reward,
                                done,
                            ]
                        )
                    else:
                        _next_states.append(
                            [
                                next_state,
                                self.trans[state][action][next_state] / N,
                                self.reward[state][action][next_state],
                                done,
                            ]
                        )
                if len(_next_states) == 0:
                    continue
                all_states.append([state, action, _next_states])

        delta = 0
        count = 0
        t0 = time.time()
        while time.time() - t0 < timeout:  # for safety
            delta = 0
            for state, action, next_states in all_states:
                q = 0
                for next_state, trans_prob, reward, done in next_states:
                    n_q = self.calc_next_q(q_tbl[next_state], policy_prob, self.invalid_actions[next_state])
                    gain = reward + (1 - done) * discount * n_q
                    q += trans_prob * gain
                delta = max(delta, abs(q_tbl[state][action] - q))
                q_tbl[state][action] = q
                count += 1
            if delta < threshold:
                break
        else:
            logger.info(f"iteration timeout(delta={delta}, threshold={threshold}, count={count})")
        return q_tbl

    def calc_next_q(self, q_tbl, prob: float, invalid_actions):
        if self.config.action_space.n == len(invalid_actions):
            # 有効アクションがない場合
            return 0

        q_max = max(q_tbl)
        if prob == 1:
            return q_max

        q_max_idx = [a for a, q in enumerate(q_tbl) if q == q_max and (a not in invalid_actions)]
        valid_actions = self.config.action_space.n - len(invalid_actions)
        if valid_actions == len(q_max_idx):
            prob = 1.0

        n_q = 0
        for a in range(self.config.action_space.n):
            if a in invalid_actions:
                continue
            elif a in q_max_idx:
                p = prob / len(q_max_idx)
            else:
                p = (1 - prob) / (valid_actions - len(q_max_idx))
            n_q += p * q_tbl[a]
        return n_q

    def sample_next_state(self, state: str, action: int):
        if state not in self.trans:
            return None
        n_s_list = list(self.trans[state][action].keys())
        if len(n_s_list) == 0:
            return None
        weights = list(self.trans[state][action].values())
        r_idx = funcs.random_choice_by_probs(weights)
        return n_s_list[r_idx]


class Trainer(RLTrainer[Config, Parameter, Memory]):
    def __init__(self, *args):
        super().__init__(*args)

        self.iteration_num = 0

    def train(self) -> None:
        if self.memory.length() == 0:
            return
        for batch in self.memory.sample():
            state, n_state, action, reward, done, invalid_actions, next_invalid_actions = batch

            self.parameter.init_model(state, action, n_state, invalid_actions, next_invalid_actions)
            self.parameter.trans[state][action][n_state] += 1
            c = self.parameter.trans[state][action][n_state]
            # online mean
            self.parameter.done[state][action][n_state] += (done - self.parameter.done[state][action][n_state]) / c
            # online mean
            self.parameter.reward[state][action][n_state] += (
                reward - self.parameter.reward[state][action][n_state]
            ) / c

            # --- q
            self.parameter.init_q(state)
            self.parameter.init_q(n_state)
            n_q = self.parameter.calc_next_q(
                self.parameter.q_tbl[n_state],
                self.config.q_policy_prob,
                next_invalid_actions,
            )
            target_q = reward + (1 - done) * self.config.q_discount * n_q
            td_error = target_q - self.parameter.q_tbl[state][action]
            self.parameter.q_tbl[state][action] += self.config.q_lr * td_error

            if self.train_count % self.config.iteration_interval == 0:
                self.parameter.iteration_q(
                    "q",
                    None,
                    self.config.q_discount,
                    self.config.q_policy_prob,
                    self.config.iteration_threshold,
                    self.config.iteration_timeout,
                )
                self.iteration_num += 1
            self.train_count += 1
        self.info["iteration"] = self.iteration_num
        self.info["mdp_size"] = self.parameter.mdp_size


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Worker(RLWorker[Config, Parameter]):
    def __init__(self, *args):
        super().__init__(*args)

        self.archive = {}
        self.archive_total_visit = {}

    def on_start(self, worker, context):
        assert not self.distributed
        self.parameter.iteration_q(
            "q",
            None,
            discount=self.config.q_discount,
            policy_prob=self.config.q_policy_prob,
            threshold=self.config.iteration_threshold,
            timeout=self.config.iteration_timeout,
        )
        self.iteration_num = 0

    def on_reset(self, worker):
        self.mode = ""
        self.start_state = self.config.observation_space.to_str(worker.state)
        if not self.training:
            return

        self.episode_step = 0
        self.episode_reward = 0
        self.explore_step = 0
        self.recent_actions = []

        if random.random() < self.config.explore_rate:
            self.cell = self.archive_select(self.start_state)
            if self.cell is not None:
                if self.cell["deter"]:
                    self.mode = "go_action"
                else:
                    # Q tbl を作成し、ターゲットの状態まで移動する
                    self.mode = "go_q"
                    self.target_q_tbl = self.parameter.iteration_q(
                        "go_q",
                        self.cell["state"],
                        discount=self.config.move_discount,
                        policy_prob=self.config.move_policy_prob,
                        threshold=0.1,
                    )
                    self.iteration_num += 1
                    self.info["go_iteration"] = self.iteration_num
            else:
                self._set_explore()
        else:
            self.mode = "q"

    def _set_explore(self):
        # 50%でランダム移動かexplore移動か
        self.explore_action = self.sample_action()
        if random.random() < 0.5:
            self.mode = "explore"
        else:
            self.mode = "random"

    def policy(self, worker) -> int:
        invalid_actions = worker.invalid_actions
        state = self.config.observation_space.to_str(worker.state)

        if self.mode == "go_action":
            # 決定的な遷移で移動
            assert self.cell is not None
            if self.cell["state"] == state:
                self._set_explore()
                return self.cell["action"]
            elif len(self.cell["actions"]) <= self.episode_step:
                self.cell["deter"] = False
                self._set_explore()
                return self.explore_action
            else:
                return self.cell["actions"][self.episode_step]
        elif self.mode == "go_q":
            # Q_tblに従って移動
            assert self.cell is not None
            if self.cell["state"] == state:
                self._set_explore()
                return self.cell["action"]
            elif state in self.target_q_tbl:
                q = self.target_q_tbl[state]
                return funcs.get_random_max_index(q, invalid_actions)
            else:
                self._set_explore()
                return self.explore_action
        elif self.mode == "explore":
            if random.random() < self.config.explore_action_change_rate:
                self.explore_action = self.sample_action()
            return self.explore_action
        elif self.mode == "random":
            return self.sample_action()
        elif self.mode == "q":
            epsilon = self.config.q_epsilon
        else:
            epsilon = self.config.test_epsilon

        if random.random() < epsilon:
            action = self.sample_action()
        else:
            self.parameter.init_q(state)
            q = self.parameter.q_tbl[state]
            action = funcs.get_random_max_index(q, invalid_actions)
        return action

    def on_step(self, worker):
        if not self.training:
            return
        state = self.config.observation_space.to_str(worker.prev_state)
        n_state = self.config.observation_space.to_str(worker.state)

        batch = [
            state,
            n_state,
            worker.action,
            worker.reward,
            1 if self.worker.terminated else 0,
            worker.prev_invalid_actions,
            worker.invalid_actions,
        ]
        self.memory.add(batch)

        # --- update archive
        self.archive_update(
            self.start_state,
            state,
            worker.action,
            self.episode_step,
            self.episode_reward,
            self.recent_actions[:],
        )
        if not worker.done:
            self.episode_step += 1
            self.episode_reward += worker.reward
            self.recent_actions.append(worker.action)
            self.archive_update(
                self.start_state,
                n_state,
                None,
                self.episode_step,
                self.episode_reward,
                self.recent_actions[:],
            )
            if self.mode in ["explore", "random"]:
                self.explore_step += 1
                if self.explore_step > self.config.explore_max_step:
                    worker.env.abort_episode()

    # ----------------------------------------------------------

    def archive_update(self, start_state, state, action, step, total_reward, actions):
        if start_state not in self.archive:
            self.archive[start_state] = {}
            self.archive_total_visit[start_state] = 0

        key = state + "_" + str(0)
        if key not in self.archive[start_state]:
            for a in range(self.config.action_space.n):
                akey = state + "_" + str(a)
                self.archive[start_state][akey] = {
                    "state": state,
                    "action": a,
                    "select": 0,
                    "visit": 0,
                    "step": np.inf,
                    "total_reward": -np.inf,
                    "deter": True,
                    "actions": actions,
                }

        if action is not None:
            key = state + "_" + str(action)
            cell = self.archive[start_state][key]
            cell["visit"] += 1
            self.archive_total_visit[start_state] += 1
            _update = False
            if cell["total_reward"] < total_reward:
                _update = True
            elif (cell["total_reward"] == total_reward) and (cell["step"] > step):
                _update = True
            if _update:
                cell["select"] = 0
                cell["step"] = step
                cell["total_reward"] = total_reward
                cell["actions"] = actions

    def archive_select(self, start_state):
        if start_state not in self.archive:
            return None
        if self.archive_total_visit[start_state] == 0:
            return None

        max_ucb = -np.inf
        max_cells = []
        N = self.archive_total_visit[start_state]
        for cell in self.archive[start_state].values():
            n = cell["visit"] + cell["select"]
            if n == 0:
                ucb = np.inf
            else:
                # --- 状態のQ値で正規化
                self.parameter.init_q(cell["state"])
                qmin = min(self.parameter.q_tbl[cell["state"]])
                qmax = max(self.parameter.q_tbl[cell["state"]])
                q = self.parameter.q_tbl[cell["state"]][cell["action"]]
                if qmin < qmax:
                    q = (q - qmin) / (qmax - qmin)
                ucb = self.config.ucb_scale * q + np.sqrt(2 * np.log(N) / n)
            if max_ucb < ucb:
                max_ucb = ucb
                max_cells = [cell]
            elif max_ucb == ucb:
                max_cells.append(cell)
        max_cell = random.choice(max_cells)
        max_cell["select"] += 1
        return max_cell

    # ----------------------------------------------------------

    def render_terminal(self, worker, **kwargs) -> None:
        prev_state = self.config.observation_space.to_str(worker.prev_state)
        act = worker.prev_action
        state = self.config.observation_space.to_str(worker.state)
        self.parameter.init_model(prev_state, act, state, worker.prev_invalid_actions, worker.invalid_actions)

        # --- archive
        if self.start_state in self.archive:
            print(f"archive total: {self.archive_total_visit[self.start_state]}")
            key = state + "_" + str(act)
            if key in self.archive[self.start_state]:
                cell = self.archive[self.start_state][key]
                print(f"archive {key}")
                print(f"  select:{cell['select']}")
                print(f"  visit :{cell['visit']}")
                print(f"  step  :{cell['step']}")
                print(f"  deter :{cell['deter']}")

        # --- MDP
        N = sum(self.parameter.trans[prev_state][act].values())
        if N > 0:
            n = self.parameter.trans[prev_state][act][state]
            print(f"trans {100 * n / N:3.1f}%({n}/{N})")
        r = self.parameter.reward[prev_state][act][state]
        done = self.parameter.done[prev_state][act][state]
        s = f"reward {r:8.5f}"
        s += f", done {done:.1%}"
        print(s)

        # --- q
        self.parameter.init_q(state)
        q = self.parameter.q_tbl[state]
        maxa = np.argmax(q)

        def _render_sub(a: int) -> str:
            s = f"q {q[a]:6.3f}"
            return s

        funcs.render_discrete_action(int(maxa), self.config.action_space, worker.env, _render_sub)
