import logging
import random
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, cast

import numpy as np
from srl.base.define import EnvAction, EnvObservationType, RLObservationType
from srl.base.env import registration
from srl.base.env.base import EnvRun, SpaceBase
from srl.base.env.genre import TurnBase2Player
from srl.base.env.spaces import BoxSpace, DiscreteSpace
from srl.base.rl.base import RuleBaseWorker, WorkerRun
from srl.base.rl.processor import Processor

logger = logging.getLogger(__name__)


# ref: https://github.com/Kaggle/kaggle-environments/tree/master/kaggle_environments/envs/connectx

registration.register(
    id="ConnectX",
    entry_point=__name__ + ":ConnectX",
)


class ConnectX(TurnBase2Player):
    def __init__(self):
        super().__init__()

        self.columns = 7
        self.rows = 6
        self._player_index = 0

    @property
    def action_space(self) -> DiscreteSpace:
        return DiscreteSpace(self.columns)

    @property
    def observation_space(self) -> SpaceBase:
        return BoxSpace(low=0, high=2, shape=(self.columns * self.rows,))

    @property
    def observation_type(self) -> EnvObservationType:
        return EnvObservationType.DISCRETE

    @property
    def max_episode_steps(self) -> int:
        return self.columns * self.rows + 2

    @property
    def player_index(self) -> int:
        return self._player_index

    def call_reset(self) -> np.ndarray:
        self.board = [0] * self.columns * self.rows
        self._player_index = 0
        return np.array(self.board)

    def call_step(self, action: int) -> Tuple[np.ndarray, float, float, bool, dict]:
        column = action

        # Mark the position.
        row = max([r for r in range(self.rows) if self.board[column + (r * self.columns)] == 0])
        self.board[column + (row * self.columns)] = self.player_index + 1

        # Check for a win.
        if self._is_win(column, row):
            if self.player_index == 0:
                reward1 = 1
                reward2 = -1
            else:
                reward1 = -1
                reward2 = 1

            return np.array(self.board), reward1, reward2, True, {}

        # Check for a tie.
        if all(mark != 0 for mark in self.board):
            return np.array(self.board), 0, 0, True, {}

        # change player
        if self._player_index == 0:
            self._player_index = 1
        else:
            self._player_index = 0

        return np.array(self.board), 0, 0, False, {}

    def _is_win(self, column, row):
        inarow = 4 - 1
        mark = self.player_index + 1

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

    def render_gui(self, **kwargs) -> None:
        raise NotImplementedError()

    def render_rgb_array(self, **kwargs) -> np.ndarray:
        raise NotImplementedError()

    def backup(self) -> Any:
        return [
            self.board[:],
            self._player_index,
        ]

    def restore(self, data: Any) -> None:
        self.board = data[0][:]
        self._player_index = data[1]

    def make_worker(self, name: str) -> Optional[RuleBaseWorker]:
        if name == "alphabeta6":
            return AlphaBeta(max_depth=6)
        elif name == "alphabeta7":
            return AlphaBeta(max_depth=7)
        elif name == "alphabeta8":
            return AlphaBeta(max_depth=8)
        elif name == "alphabeta9":
            return AlphaBeta(max_depth=9)
        elif name == "alphabeta10":
            return AlphaBeta(max_depth=10)

        return None

    """
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

    def call_direct_reset(self, observation, configuration) -> np.ndarray:
        self._player_index = observation.mark - 1
        self.board = observation.board[:]
        return np.array(self.board)

    def call_direct_step(self, observation, configuration) -> Tuple[np.ndarray, float, float, bool, dict]:
        self._player_index = observation.mark - 1
        self.board = observation.board[:]
        return np.array(self.board), 0, 0, False, {}


@dataclass
class AlphaBeta(RuleBaseWorker):

    max_depth: int = 4
    timeout: int = 6  # s
    equal_cut: bool = True

    def call_on_reset(self, env: EnvRun, worker: WorkerRun) -> None:
        pass  #

    def call_policy(self, env: EnvRun, worker: WorkerRun) -> EnvAction:
        self._count = 0
        self.t0 = time.time()
        scores, action = self._alphabeta(env.get_original_env().copy())

        scores = np.array(scores)
        if worker.player_index == 1:
            scores = -scores

        self._action = action
        self._scores = scores
        self._count = self._count
        self._time = time.time() - self.t0

        # action = int(random.choice(np.where(scores == scores.max())[0]))
        return action

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

        if env.player_index == 0:

            # 自分の番
            scores = [-9.0 for _ in range(env.action_space.n)]
            for a in actions:
                if a in invalid_actions:
                    continue
                env.restore(env_dat)

                _, r1, r2, done, _ = env.call_step(a)
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

                _, r1, r2, done, _ = env.call_step(a)
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

    def call_render(self, env: EnvRun, worker_run: WorkerRun) -> None:
        print(f"- alphabeta act: {self._action}, count: {self._count}, {self._time:.3f}s) -")
        print("+---+---+---+---+---+---+---+")
        s = "|"
        for a in range(env.action_space.n):
            s += "{:2d} |".format(int(self._scores[a]))
        print(s)
        print("+---+---+---+---+---+---+---+")


class LayerProcessor(Processor):
    def change_observation_info(
        self,
        env_observation_space: SpaceBase,
        env_observation_type: EnvObservationType,
        rl_observation_type: RLObservationType,
        _env: EnvRun,
    ) -> Tuple[SpaceBase, EnvObservationType]:
        env = cast(ConnectX, _env.get_original_env())

        observation_space = BoxSpace(
            low=0,
            high=1,
            shape=(3, env.columns, env.rows),
        )
        return observation_space, EnvObservationType.SHAPE3

    def process_observation(self, observation: np.ndarray, _env: EnvRun) -> np.ndarray:
        env = cast(ConnectX, _env.get_original_env())

        board = observation
        # Layer0: player1 field (0 or 1)
        # Layer1: player2 field (0 or 1)
        # Layer2: player_index (all0 or all1)
        _field = np.zeros((3, env.columns, env.rows))
        for y in range(env.columns):
            for x in range(env.rows):
                idx = x + y * env.rows
                if board[idx] == 1:
                    _field[0][y][x] = 1
                elif board[idx] == 2:
                    _field[1][y][x] = 1
        _field[2] = env.player_index
        return _field
