import json
import logging
import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from srl.base.define import DoneTypes
from srl.base.exception import UndefinedError
from srl.base.rl.algorithms.base_ql import RLConfig, RLWorker
from srl.base.rl.parameter import RLParameter
from srl.base.rl.registration import register
from srl.base.rl.trainer import RLTrainer
from srl.rl import functions as funcs
from srl.rl.memories.sequence_memory import SequenceMemory

logger = logging.getLogger(__name__)


# ------------------------------------------------------
# config
# ------------------------------------------------------
@dataclass
class Config(RLConfig):
    #: 方策反復法におけるタイムアウト
    iteration_timeout: float = 10
    #: 方策反復法の学習完了の閾値
    iteration_threshold: float = 0.0001

    #: 割引率
    discount: float = 0.9

    ucb_reward_scale: float = 0.1
    cell_addition_step: int = 10
    cell_policy_prob: float = 0.99
    cell_iteration_threshold: float = 0.1

    search_max_step: int = 20
    backup_rate: float = 0.9

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


class Memory(SequenceMemory):
    pass


# ------------------------------------------------------
# Parameter
# ------------------------------------------------------
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

        # [reset_state][dist_state + action]
        self.archive = {}
        # [reset_state]
        self.archive_count = {}
        self.archive_size = 0  # for info

        self.rmin = np.inf
        self.rmax = -np.inf

        # --- cache
        self.q_tbl_ext = {}
        self.q_tbl_target = {}

    def call_restore(self, data: Any, **kwargs) -> None:
        d = json.loads(data)
        self.trans = d[0]
        self.reward = d[1]
        self.done = d[3]
        self.invalid_actions = d[4]
        self.mdp_size = d[5]
        self.archive = d[6]
        self.archive_count = d[7]
        self.archive_size = d[8]
        self.rmin = d[9]
        self.rmax = d[10]

    def call_backup(self, **kwargs):
        return json.dumps(
            [
                self.trans,
                self.reward,
                self.done,
                self.invalid_actions,
                self.mdp_size,
                self.archive,
                self.archive_count,
                self.archive_size,
                self.rmin,
                self.rmax,
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

    def iteration_q(
        self,
        mode: str,
        discount: float,
        policy_prob: float,
        target_state: str = "",
        threshold: float = 0.0001,
        timeout: float = 1,
    ):
        if mode == "ext":
            q_tbl = self.q_tbl_ext
        elif mode == "target":
            q_tbl = self.q_tbl_target
        else:
            raise UndefinedError(mode)

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
                    if mode == "ext":
                        _next_states.append(
                            [
                                next_state,
                                self.trans[state][action][next_state] / N,
                                self.reward[state][action][next_state],
                                self.done[state][action][next_state],
                            ]
                        )
                    elif mode == "target":
                        # targetなら1、終了なら-1
                        done = self.done[state][action][next_state]
                        if state == target_state:
                            reward = 1
                            done = 1
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
                        raise UndefinedError(mode)

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
            logger.info(f"[{mode}] iteration timeout(delta={delta}, threshold={threshold}, count={count})")
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

    def calc_q_normalize(self, q: np.ndarray, q_min: float, q_max: float):
        if q_min >= q_max:
            return q
        return (q - q_min) / (q_max - q_min)

    def sample_next_state(self, state: str, action: int):
        if state not in self.trans:
            return None
        n_s_list = list(self.trans[state][action].keys())
        if len(n_s_list) == 0:
            return None
        weights = list(self.trans[state][action].values())
        r_idx = funcs.random_choice_by_probs(weights)
        return n_s_list[r_idx]

    def archive_update(self, reset_state, state, action, n_state, step, backup):
        if reset_state not in self.archive:
            self.archive[reset_state] = {}
            self.archive_count[reset_state] = 0
        self.archive_count[reset_state] += 1

        cell_key = self._archive_init(reset_state, state, action)
        self._archive_init(reset_state, n_state, action)

        cell = self.archive[reset_state][cell_key]
        cell["visit"] += 1
        if (cell["step"] > step) and (backup is not None):
            cell["step"] = step
            cell["backup"] = backup
            cell["select"] = 0

    def is_archive_update(self, reset_state, state, action, step):
        if reset_state not in self.archive:
            return True
        key = state + "_" + str(action)
        if key not in self.archive[reset_state]:
            return True
        cell = self.archive[reset_state][key]
        if cell["step"] > step:
            return True
        return False

    def _archive_init(self, reset_state, state, action):
        key = state + "_" + str(action)
        if key not in self.archive[reset_state]:
            for a in range(self.config.action_space.n):
                self.archive[reset_state][state + "_" + str(a)] = {
                    "state": state,
                    "action": a,
                    "step": np.inf,
                    "visit": 0,
                    "select": 0,
                }
                self.archive_size += 1
        return key

    def archive_select(self, reset_state):
        if reset_state not in self.archive:
            return None

        # --- UCB
        max_ucb = -np.inf
        max_cells = []
        N = self.archive_count[reset_state]
        for cell in self.archive[reset_state].values():
            n = cell["visit"] + cell["select"]
            if n == 0:
                ucb = np.inf
            else:
                # MDPで[state,action]の平均報酬
                reward = self.get_reward(cell["state"], cell["action"], normalized=True)
                ucb = self.config.ucb_reward_scale * reward + np.sqrt(2 * np.log(N) / n)
            if max_ucb < ucb:
                max_ucb = ucb
                max_cells = [cell]
            elif max_ucb == ucb:
                max_cells.append(cell)
        max_cell = random.choice(max_cells)
        return max_cell

    def get_reward(self, state, action, normalized: bool):
        if state not in self.trans:
            return 0
        N = sum(self.trans[state][action].values())
        if N == 0:
            return 0
        reward = 0
        for n_state, n in self.trans[state][action].items():
            if n == 0:
                continue
            prob = n / N
            r = self.reward[state][action][n_state]
            if normalized:
                if self.rmax > self.rmin:
                    r = (r - self.rmin) / (self.rmax - self.rmin)
            reward += r * prob
        return reward


# ------------------------------------------------------
# Trainer
# ------------------------------------------------------
class Trainer(RLTrainer[Config, Parameter, Memory]):
    def __init__(self, *args):
        super().__init__(*args)

    def train(self) -> None:
        if self.memory.is_warmup_needed():
            return
        batchs = self.memory.sample()

        for batch in batchs:
            reset_state = batch["reset_state"]
            state = batch["state"]
            n_state = batch["next_state"]
            action = batch["action"]
            reward = batch["reward"]
            done = batch["done"]
            invalid_actions = batch["invalid_actions"]
            next_invalid_actions = batch["next_invalid_actions"]
            step = batch["step"]
            backup = batch["backup"]

            # --- update reward
            self.parameter.rmin = min(self.parameter.rmin, reward)
            self.parameter.rmax = max(self.parameter.rmax, reward)

            # --- update model
            self.parameter.init_model(state, action, n_state, invalid_actions, next_invalid_actions)
            self.parameter.trans[state][action][n_state] += 1
            c = self.parameter.trans[state][action][n_state]
            # online mean
            self.parameter.done[state][action][n_state] += (done - self.parameter.done[state][action][n_state]) / c
            # online mean
            self.parameter.reward[state][action][n_state] += (
                reward - self.parameter.reward[state][action][n_state]
            ) / c

            # --- update archive
            self.parameter.archive_update(reset_state, state, action, n_state, step, backup)

            self.train_count += 1

        self.info["mdp_size"] = self.parameter.mdp_size
        self.info["archive_size"] = self.parameter.archive_size


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Worker(RLWorker[Config, Parameter]):
    def on_start(self, worker, context):
        self.q_tbl = self.parameter.iteration_q(
            "ext",
            discount=self.config.discount,
            policy_prob=1.0,
            threshold=self.config.iteration_threshold,
            timeout=self.config.iteration_timeout,
        )

    def on_reset(self, worker):
        self.reset_state = self.config.observation_space.to_str(worker.state)

        if self.training:
            self.cell_step = 0
            self.search_step = 0

            cell = self.parameter.archive_select(self.reset_state)
            if cell is None:
                self.mode = "search"
            elif ("backup" in cell) and (random.random() < self.config.backup_rate):
                self.mode = "search"
                cell["select"] += 1
                worker.restore(cell["backup"])
            else:
                self.mode = "move"
                cell["select"] += 1
                self.cell_target = cell["state"]
                self.cell_action = cell["action"]
                self.cell_max_step = cell["step"] + self.config.cell_addition_step
                self.cell_q_tbl = self.parameter.iteration_q(
                    "target",
                    target_state=self.cell_target,
                    discount=0.9,
                    policy_prob=self.config.cell_policy_prob,
                    threshold=self.config.cell_iteration_threshold,
                )
                if self.distributed:
                    assert False, "TODO"

        self.state = self.config.observation_space.to_str(worker.state)

    def policy(self, worker) -> int:
        invalid_actions = worker.invalid_actions

        if self.training:
            if self.mode == "move":
                if self.cell_target == self.state:
                    action = self.cell_action
                    self.mode = "search"
                elif self.state in self.cell_q_tbl:
                    q = self.cell_q_tbl[self.state]
                    action = funcs.get_random_max_index(q, invalid_actions)
                else:
                    action = self.sample_action()
                    self.mode = "search"
            else:
                action = self.sample_action()
        else:
            if self.state in self.q_tbl:
                action = int(np.argmax(self.q_tbl[self.state]))
            else:
                action = self.sample_action()

        return action

    def on_step(self, worker):
        prev_state = self.state
        self.state = self.config.observation_space.to_str(worker.state)
        if not self.training:
            return
        self.cell_step += 1

        if self.parameter.is_archive_update(self.reset_state, prev_state, worker.action, self.cell_step):
            backup = worker.backup()
        else:
            backup = None

        batch = {
            "reset_state": self.reset_state,
            "state": prev_state,
            "next_state": self.state,
            "action": worker.action,
            "reward": worker.reward,
            "done": 1 if self.worker.done_type == DoneTypes.TERMINATED else 0,
            "invalid_actions": worker.prev_invalid_actions,
            "next_invalid_actions": worker.invalid_actions,
            "step": self.cell_step,
            "backup": backup,
        }
        self.memory.add(batch)

        if self.mode == "move":
            if self.cell_step > self.cell_max_step:
                self.mode = "search"
        else:
            self.search_step += 1
            if self.search_step > self.config.search_max_step:
                worker.env.end_episode()

    def render_terminal(self, worker, **kwargs) -> None:
        prev_state = self.config.observation_space.to_str(worker.prev_state)
        act = worker.prev_action
        state = self.config.observation_space.to_str(worker.state)
        self.parameter.init_model(prev_state, act, state, worker.prev_invalid_actions, worker.invalid_actions)

        # --- MDP
        N = sum(self.parameter.trans[prev_state][act].values())
        if N > 0:
            n = self.parameter.trans[prev_state][act][state]
            print(f"trans {100 * n / N:.1}%({n}/{N})")
        r = self.parameter.reward[prev_state][act][state]
        done = self.parameter.done[prev_state][act][state]
        s = f"reward {r:8.5f}"
        s += f", done {done:.1%}"
        print(s)

        print(f"reward range: {self.parameter.rmin} - {self.parameter.rmax}")

        q = np.array(self.q_tbl.get(state, [0] * self.config.action_space.n))
        maxa = np.argmax(q)

        def _render_sub(a: int) -> str:
            s = f"q {q[a]:6.3f}"
            return s

        funcs.render_discrete_action(int(maxa), self.config.action_space, worker.env, _render_sub)
