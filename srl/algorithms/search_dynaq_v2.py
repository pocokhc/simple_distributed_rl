import json
import logging
import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from srl.base.exception import UndefinedError
from srl.base.rl.algorithms.base_ql import RLConfig, RLWorker
from srl.base.rl.memory import RLMemory
from srl.base.rl.parameter import RLParameter
from srl.base.rl.registration import register
from srl.base.rl.trainer import RLTrainer
from srl.rl import functions as funcs

logger = logging.getLogger(__name__)


@dataclass
class Config(RLConfig):

    q_epsilon: float = 0.001
    search_epsilon: float = 0.5
    test_epsilon: float = 0.0001

    search_rate: float = 0.5
    ucb_scale: float = 0.1
    q_target_policy_prob: float = 1.0
    q_discount: float = 0.9
    q_lr: float = 0.1
    search_max_step: int = 10

    #: 方策反復法の学習完了の閾値
    iteration_threshold: float = 0.0001
    #: 方策反復法におけるタイムアウト
    iteration_timeout: float = 10
    #: 方策反復法を実行する間隔(学習回数)
    iteration_interval: int = 10_000

    def get_framework(self) -> str:
        return ""

    def get_name(self) -> str:
        return "SearchDynaQ_v2"

    def assert_params(self) -> None:
        super().assert_params()


register(
    Config(),
    __name__ + ":Memory",
    __name__ + ":Parameter",
    __name__ + ":Trainer",
    __name__ + ":Worker",
)


class Memory(RLMemory[Config]):
    def __init__(self, *args):
        super().__init__(*args)

        self.buffer_mdp = []
        self.buffer_archive = []

    def length(self) -> int:
        return len(self.buffer_mdp) + len(self.buffer_archive)

    def add(self, mode, batch) -> None:
        if mode == "mdp":
            self.buffer_mdp.append(batch)
        elif mode == "archive":
            self.buffer_archive.append(batch)
        else:
            raise UndefinedError(mode)

    def sample_mdp(self):
        b = self.buffer_mdp
        self.buffer_mdp = []
        return b

    def sample_archive(self):
        b = self.buffer_archive
        self.buffer_archive = []
        return b


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

        # [dist_state + action]
        self.archive = {}
        self.archive_total_visit = 0

        # [state][action]
        self.q_tbl = {}
        self.q_min = np.inf
        self.q_max = -np.inf

    def call_restore(self, data: Any, **kwargs) -> None:
        d = json.loads(data)
        self.trans = d[0]
        self.reward = d[1]
        self.done = d[2]
        self.invalid_actions = d[3]
        self.mdp_size = d[4]
        self.archive = d[5]
        self.archive_total_visit = d[6]
        self.q_tbl = d[7]
        self.q_min = d[8]
        self.q_max = d[9]

    def call_backup(self, **kwargs):
        return json.dumps(
            [
                self.trans,
                self.reward,
                self.done,
                self.invalid_actions,
                self.mdp_size,
                self.archive,
                self.archive_total_visit,
                self.q_tbl,
                self.q_min,
                self.q_max,
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
        discount: float,
        policy_prob: float,
        threshold: float = 0.0001,
        timeout: float = 1,
    ):
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
                    _next_states.append(
                        [
                            next_state,
                            self.trans[state][action][next_state] / N,
                            self.reward[state][action][next_state],
                            self.done[state][action][next_state],
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

    # -------------------------------------------

    def archive_update(self, batch):
        state, action, step, backup = batch

        key = state + "_" + str(0)
        if key not in self.archive:
            for a in range(self.config.action_space.n):
                akey = state + "_" + str(a)
                self.archive[akey] = {
                    "state": state,
                    "action": a,
                    "select": 0,
                    "visit": 0,
                    "step": step,
                    "backup": backup,
                }

        if action is not None:
            key = state + "_" + str(action)
            cell = self.archive[key]
            cell["visit"] += 1
            self.archive_total_visit += 1
            if cell["step"] > step:
                cell["select"] = 0
                cell["step"] = step
                cell["backup"] = backup

    def archive_select(self):
        if len(self.archive):
            return None
        if self.archive_total_visit == 0:
            return None

        max_ucb = -np.inf
        max_cells = []
        for cell in self.archive.values():
            n = cell["visit"]
            if n == 0:
                ucb = np.inf
            else:
                q = self.q_tbl[cell["state"]][cell["action"]]
                if self.q_min < self.q_max:
                    q = (q - self.q_min) / (self.q_max - self.q_min)
                ucb = self.config.ucb_scale * q + np.sqrt(2 * np.log(self.archive_total_visit) / n)
            if max_ucb < ucb:
                max_ucb = ucb
                max_cells = [cell]
            elif max_ucb == ucb:
                max_cells.append(cell)
        max_cell = random.choice(max_cells)
        max_cell["select"] += 1
        return max_cell


class Trainer(RLTrainer[Config, Parameter, Memory]):
    def __init__(self, *args):
        super().__init__(*args)

        self.iteration_num = 0

    def train(self) -> None:
        self._train_archive()
        self._train_mdp()

    def _train_archive(self):
        if len(self.memory.buffer_archive) == 0:
            return
        for batch in self.memory.sample_archive():
            self.parameter.archive_update(batch)
        self.info["archive_size"] = len(self.parameter.archive)

    def _train_mdp(self):
        if len(self.memory.buffer_mdp) == 0:
            return
        for batch in self.memory.sample_mdp():
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
                self.config.q_target_policy_prob,
                next_invalid_actions,
            )
            target_q = reward + (1 - done) * self.config.q_discount * n_q
            td_error = target_q - self.parameter.q_tbl[state][action]
            self.parameter.q_tbl[state][action] += self.config.q_lr * td_error

            self.parameter.q_min = min(self.parameter.q_min, self.parameter.q_tbl[state][action])
            self.parameter.q_max = max(self.parameter.q_max, self.parameter.q_tbl[state][action])

            if self.train_count % self.config.iteration_interval == 0:
                self.parameter.iteration_q(
                    self.config.q_discount,
                    self.config.q_target_policy_prob,
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
    def on_start(self, worker, context):
        self.parameter.iteration_q(
            discount=self.config.q_discount,
            policy_prob=self.config.q_target_policy_prob,
            threshold=self.config.iteration_threshold,
            timeout=self.config.iteration_timeout,
        )

    def on_reset(self, worker):
        self.cell_action = None
        self.mode = ""

        if not self.training:
            return

        self.episode_step = 0
        self.cell_step = 0

        if random.random() < self.config.search_rate:
            self.mode = "search"
            cell = self.parameter.archive_select()
            if cell is not None:
                self.mode = "move"
                self.episode_step = cell["step"]
                self.cell_action = cell["action"]
                self.backup = cell["backup"]
                worker.restore(self.backup)
            else:
                self.backup = worker.backup()
        else:
            self.mode = "q"
            self.backup = worker.backup()

    def policy(self, worker) -> int:
        invalid_actions = worker.invalid_actions

        if self.cell_action is not None:
            action = self.cell_action
            self.cell_action = None
        else:
            if self.training:
                if self.mode == "q":
                    epsilon = self.config.q_epsilon
                else:
                    epsilon = self.config.search_epsilon
            else:
                epsilon = self.config.test_epsilon

            if random.random() < epsilon:
                action = self.sample_action()
            else:
                state = self.config.observation_space.to_str(worker.state)
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
        self.memory.add("mdp", batch)

        # --- update archive
        if not worker.done:
            batch = [state, worker.action, self.episode_step, self.backup]
            self.memory.add("archive", batch)
            self.episode_step += 1
            self.backup = worker.backup()
            batch = [n_state, None, self.episode_step, self.backup]
            self.memory.add("archive", batch)

        self.cell_step += 1
        if self.cell_step > self.config.search_max_step:
            worker.env.abort_episode()

    def render_terminal(self, worker, **kwargs) -> None:
        prev_state = self.config.observation_space.to_str(worker.prev_state)
        act = worker.prev_action
        state = self.config.observation_space.to_str(worker.state)
        self.parameter.init_model(prev_state, act, state, worker.prev_invalid_actions, worker.invalid_actions)

        # --- archive
        print(f"mode: {self.mode}, archive total: {self.parameter.archive_total_visit}")
        key = state + "_" + str(act)
        if key in self.parameter.archive:
            cell = self.parameter.archive[key]
            print(f"archive {key}")
            print(f"  select:{cell['select']}")
            print(f"  visit :{cell['visit']}")
            print(f"  step  :{cell['step']}")

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
