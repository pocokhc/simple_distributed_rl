import logging
from typing import Any, List, Optional, Tuple, Type

import numpy as np
from srl.base.define import EnvObservationType, RLObservationType
from srl.base.env import registration
from srl.base.env.base import SpaceBase
from srl.base.env.genre import TurnBase2Player
from srl.base.env.spaces import BoxSpace, DiscreteSpace
from srl.base.rl.algorithms.rulebase import RuleBaseWorker
from srl.base.rl.base import RLWorker
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

    def get_invalid_actions(self, player_index: int) -> List[int]:
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

    def make_worker(self, name: str) -> Optional[Type[RLWorker]]:
        if name == "negamax":
            return NegaMax
        return None

    # ---------------------
    def set_kaggle_agent_step(self, observation, configuration):
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
        self._player_index = observation.mark - 1
        self.board = observation.board[:]


class NegaMax(RuleBaseWorker):
    def call_on_reset(self, env) -> None:
        pass  #

    def call_policy(self, env: ConnectX) -> int:
        observation = env.env.state[0]["observation"]
        configuration = env.env.configuration
        action = negamax_agent(observation, configuration)
        return action

    def call_render(self, env: ConnectX) -> None:
        pass  #


class LayerProcessor(Processor):
    def change_observation_info(
        self,
        env_observation_space: SpaceBase,
        env_observation_type: EnvObservationType,
        rl_observation_type: RLObservationType,
        env: ConnectX,
    ) -> Tuple[SpaceBase, EnvObservationType]:
        observation_space = BoxSpace(
            low=0,
            high=1,
            shape=(3, env.columns, env.rows),
        )
        return observation_space, EnvObservationType.SHAPE3

    def process_observation(self, observation: np.ndarray, env: ConnectX) -> np.ndarray:
        # Layer0: player1 field (0 or 1)
        # Layer1: player2 field (0 or 1)
        # Layer2: player_index (all0 or all1)
        _field = np.zeros((3, env.columns, env.rows))
        for y in range(env.columns):
            for x in range(env.rows):
                idx = x + y * env.rows
                if env.board[idx] == 1:
                    _field[0][y][x] = 1
                elif env.board[idx] == 2:
                    _field[1][y][x] = 1
        _field[2] = env.player_index
        return _field
