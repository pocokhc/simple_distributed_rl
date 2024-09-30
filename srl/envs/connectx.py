import logging
import random
import time
from typing import Any, List, Optional, Tuple, cast

import numpy as np

from srl.base.define import EnvActionType, SpaceTypes
from srl.base.env import registration
from srl.base.env.base import EnvBase
from srl.base.env.env_run import EnvRun
from srl.base.rl.algorithms.env_worker import EnvWorker
from srl.base.spaces.array_discrete import ArrayDiscreteSpace
from srl.base.spaces.box import BoxSpace
from srl.base.spaces.discrete import DiscreteSpace

logger = logging.getLogger(__name__)


# ref: https://github.com/Kaggle/kaggle-environments/tree/master/kaggle_environments/envs/connectx

registration.register(
    id="connectx",
    entry_point=__name__ + ":ConnectX",
    check_duplicate=False,
)
registration.register(
    id="connectx-layer",
    entry_point=__name__ + ":ConnectX",
    kwargs={"obs_type": "layer"},
    check_duplicate=False,
)


def board_reverse(board):
    b = board[:]
    for i in range(6):
        b[i * 7 + 6], b[i * 7 + 0] = b[i * 7 + 0], b[i * 7 + 6]
        b[i * 7 + 5], b[i * 7 + 1] = b[i * 7 + 1], b[i * 7 + 5]
        b[i * 7 + 4], b[i * 7 + 2] = b[i * 7 + 2], b[i * 7 + 4]
    return b


class ConnectX(EnvBase):
    def __init__(
        self,
        obs_type: str = "",  # "" or "layer"
    ):
        super().__init__()
        self.obs_type = obs_type

        self.columns = 7
        self.rows = 6
        self._next_player_index = 0

    @property
    def action_space(self) -> DiscreteSpace:
        return DiscreteSpace(self.columns)

    @property
    def observation_space(self):
        if self.obs_type == "layer":
            return BoxSpace(
                shape=(self.columns, self.rows, 2),
                low=0,
                high=1,
                stype=SpaceTypes.IMAGE,
            )
        else:
            return ArrayDiscreteSpace(self.columns * self.rows, low=0, high=2)

    @property
    def player_num(self) -> int:
        return 2

    @property
    def max_episode_steps(self) -> int:
        return self.columns * self.rows + 2

    def reset(self, *, seed: Optional[int] = None, **kwargs) -> Any:
        self.board = [0] * self.columns * self.rows
        self.next_player = 0
        return self._create_state()

    def backup(self) -> Any:
        return [
            self.board[:],
            self.next_player,
        ]

    def restore(self, data: Any) -> None:
        self.board = data[0][:]
        self.next_player = data[1]

    def _create_state(self):
        if self.obs_type == "layer":
            # Layer0: my player field (0 or 1)
            # Layer1: enemy player field (0 or 1)
            _field = np.zeros((self.columns, self.rows, 2))
            if self.next_player == 0:
                my_player = 1
                enemy_player = 2
            else:
                my_player = 2
                enemy_player = 1
            for y in range(self.columns):
                for x in range(self.rows):
                    idx = x + y * self.rows
                    if self.board[idx] == my_player:
                        _field[y][x][0] = 1
                    elif self.board[idx] == enemy_player:
                        _field[y][x][1] = 1
            return _field
        else:
            return self.board

    def step(self, action: int) -> Tuple[Any, List[float], bool, bool]:
        column = action

        # Mark the position.
        row = max([r for r in range(self.rows) if self.board[column + (r * self.columns)] == 0])
        self.board[column + (row * self.columns)] = self.next_player + 1

        # Check for a win.
        if self._is_win(column, row):
            if self.next_player == 0:
                reward1 = 1.0
                reward2 = -1.0
            else:
                reward1 = -1.0
                reward2 = 1.0

            return self._create_state(), [reward1, reward2], True, False

        # Check for a tie.
        if all(mark != 0 for mark in self.board):
            return self._create_state(), [0.0, 0.0], True, False

        # change player
        if self.next_player == 0:
            self.next_player = 1
        else:
            self.next_player = 0

        return self._create_state(), [0.0, 0.0], False, False

    def _is_win(self, column, row):
        inarow = 4 - 1
        mark = self.next_player + 1

        def count(offset_row, offset_column):
            for i in range(1, inarow + 1):
                r = row + offset_row * i
                c = column + offset_column * i
                if r < 0 or r >= self.rows or c < 0 or c >= self.columns or self.board[c + (r * self.columns)] != mark:
                    return i - 1
            return inarow

        return (
            count(1, 0) >= inarow  # vertical.
            or (count(0, 1) + count(0, -1)) >= inarow  # horizontal.
            or (count(-1, -1) + count(1, 1)) >= inarow  # top left diagonal.
            or (count(-1, 1) + count(1, -1)) >= inarow  # top right diagonal.
        )

    def get_invalid_actions(self, player_index: int = 0) -> List[int]:
        invalid_actions = [a for a in range(self.action_space.n) if self.board[a] != 0]
        return invalid_actions

    def render_terminal(self, **kwargs) -> None:
        def print_row(values, delim="|"):
            return f"{delim} " + f" {delim} ".join(str(v) for v in values) + f" {delim}\n"

        row_bar = "+" + "+".join(["---"] * self.columns) + "+\n"
        out = row_bar
        for r in range(self.rows):
            out += print_row(self.board[r * self.columns : r * self.columns + self.columns]) + row_bar

        print(out)

    @property
    def render_interval(self) -> float:
        return 1000 / 1

    def make_worker(self, name: str, **kwargs) -> Optional[EnvWorker]:
        for n in range(2, 12):
            if name == "alphabeta" + str(n):
                return AlphaBeta(max_depth=n, **kwargs)
        return None

    def direct_step(self, observation, configuration) -> Tuple[bool, Any]:
        """kaggle_environment を想定
        observation = {
            "remainingOverageTime": 60,
            "step": 0,
            "board": [0, 0, 1, 2, ...] (6*7)
            "mark": 1,
        }
        configuration = {
            "episodeSteps": 1000,
            "actTimeout": 2,
            "runTimeout": 1200,
            "columns": 7,
            "rows": 6,
            "inarow": 4,
            "agentTimeout": 60,
            "timeout": 2,
        }
        """
        self.next_player = observation.mark - 1
        self.board = observation.board[:]

        # 先行なら step==0、後攻なら step==1 がエピソードの最初
        step = observation.step
        is_start_episode = step == 0 or step == 1

        return is_start_episode, self._create_state()

    def decode_action(self, action):
        return action

    @property
    def can_simulate_from_direct_step(self) -> bool:
        return True


class AlphaBeta(EnvWorker):
    def __init__(
        self,
        max_depth: int = 4,
        timeout: float = 6,  # s
        equal_cut: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_depth = max_depth
        self.timeout = timeout
        self.equal_cut = equal_cut

    def call_policy(self, env: EnvRun) -> Tuple[EnvActionType, dict]:
        self._count = 0
        self.t0 = time.time()
        scores, action = self._alphabeta(cast(ConnectX, cast(ConnectX, env.unwrapped).copy()))

        scores = np.array(scores)
        if self.player_index == 1:
            scores = -scores

        self._action = action
        self._scores = scores
        self._count = self._count
        self._time = time.time() - self.t0

        # action = int(np.random.choice(np.where(scores == scores.max())[0]))
        return action, {}

    def _alphabeta(self, env: ConnectX, alpha=-np.inf, beta=np.inf, depth: int = 0):
        if depth == self.max_depth:
            return [0] * env.action_space.n, 0
        if self.t0 - time.time() > self.timeout:
            return [0] * env.action_space.n

        self._count += 1
        env_dat = env.backup()
        invalid_actions = env.get_invalid_actions()

        actions = [a for a in range(env.action_space.n)]
        random.shuffle(actions)
        select_action = 0

        if env.next_player == 0:
            # 自分の番
            scores = [-9.0 for _ in range(env.action_space.n)]
            for a in actions:
                if a in invalid_actions:
                    continue
                env.restore(env_dat)

                _, (r1, r2), done, _ = env.step(a)
                # print(np.array(env.board).reshape((6, 7)))
                if done:
                    scores[a] = r1
                else:
                    n_scores, _ = self._alphabeta(env, alpha, beta, depth + 1)
                    scores[a] = np.min(n_scores)

                # maximum
                if alpha < scores[a]:
                    select_action = a
                    alpha = scores[a]

                # beta cut
                if self.equal_cut:
                    if scores[a] >= beta:
                        break

        else:
            # 相手の番
            scores = [9.0 for _ in range(env.action_space.n)]
            for a in actions:
                if a in invalid_actions:
                    continue
                env.restore(env_dat)

                _, (r1, r2), done, _ = env.step(a)
                # print(np.array(env.board).reshape((6, 7)))
                if done:
                    scores[a] = r1
                else:
                    n_scores, _ = self._alphabeta(env, alpha, beta, depth + 1)
                    scores[a] = np.max(n_scores)

                # minimum
                if beta > scores[a]:
                    select_action = a
                    beta = scores[a]

                # alpha cut
                if self.equal_cut:
                    if scores[a] <= alpha:
                        break

        return scores, select_action

    def render_terminal(self, env: EnvRun, worker, **kwargs) -> None:
        print(f"- alphabeta act: {self._action}, count: {self._count}, {self._time:.3f}s) -")
        print("+---+---+---+---+---+---+---+")
        s = "|"
        for a in range(cast(DiscreteSpace, env.action_space).n):
            s += "{:2d} |".format(int(self._scores[a]))
        print(s)
        print("+---+---+---+---+---+---+---+")
