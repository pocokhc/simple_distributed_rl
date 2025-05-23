import logging
from typing import List, Optional, Tuple, cast

import numpy as np
from kaggle_environments.envs.connectx.connectx import negamax_agent

from srl.base.define import SpaceTypes
from srl.base.env import registration
from srl.base.env.env_run import EnvRun, SpaceBase
from srl.base.env.kaggle_wrapper import KaggleWorker, KaggleWrapper
from srl.base.rl.config import RLConfig
from srl.base.rl.processor import RLProcessor
from srl.base.spaces import ArrayDiscreteSpace, BoxSpace, DiscreteSpace

logger = logging.getLogger(__name__)


registration.register(
    id="kaggle_connectx",
    entry_point=__name__ + ":ConnectX",
    check_duplicate=False,
)


class ConnectX(KaggleWrapper):
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

    def __init__(self):
        super().__init__("connectx")

        assert isinstance(self.env.configuration, dict)
        self._action_num = self.env.configuration["columns"]
        self.columns = self.env.configuration["columns"]
        self.rows = self.env.configuration["rows"]

    @property
    def action_space(self) -> DiscreteSpace:
        return DiscreteSpace(self._action_num)

    @property
    def observation_space(self) -> ArrayDiscreteSpace:
        return ArrayDiscreteSpace(self.columns * self.rows, low=0, high=2)

    @property
    def max_episode_steps(self) -> int:
        return self.columns * self.rows + 2

    @property
    def player_num(self) -> int:
        return 2

    def encode_obs(self, observation, configuration) -> Tuple[bool, List[int], int, dict]:
        step = observation.step
        player_index = observation.mark - 1
        self.board = observation.board

        # 先行なら step==0、後攻なら step==1 がエピソードの最初
        is_start_episode = step == 0 or step == 1

        return is_start_episode, self.board, player_index, {}

    def decode_action(self, action):
        return action

    def get_invalid_actions(self, player_index: int) -> List[int]:
        invalid_actions = [a for a in range(self.action_space.n) if self.board[a] != 0]
        return invalid_actions

    def make_worker(self, name: str, **kwargs) -> Optional[KaggleWorker]:
        if name == "negamax":
            return NegaMax(**kwargs)
        return None


class NegaMax(KaggleWorker):
    def kaggle_policy(self, observation, configuration):
        return negamax_agent(observation, configuration)


class LayerProcessor(RLProcessor):
    def remap_observation_space(self, env_observation_space: SpaceBase, env: EnvRun, rl_config: RLConfig) -> SpaceBase:
        _env = cast(ConnectX, env.env)
        observation_space = BoxSpace(
            low=0,
            high=1,
            shape=(2, _env.columns, _env.rows),
            stype=SpaceTypes.IMAGE,
        )
        return observation_space

    def remap_observation(self, observation: np.ndarray, worker, env: EnvRun) -> np.ndarray:
        _env = cast(ConnectX, env.env)

        # Layer0: my player field (0 or 1)
        # Layer1: enemy player field (0 or 1)
        _field = np.zeros((_env.columns, _env.rows, 2))
        if env.next_player == 0:
            my_player = 1
            enemy_player = 2
        else:
            my_player = 2
            enemy_player = 1
        for y in range(_env.columns):
            for x in range(_env.rows):
                idx = x + y * _env.rows
                if _env.board[idx] == my_player:
                    _field[y][x][0] = 1
                elif _env.board[idx] == enemy_player:
                    _field[y][x][1] = 1
        return _field


if __name__ == "__main__":
    from pprint import pprint

    from kaggle_environments import make

    env = make("connectx", debug=True)
    pprint(env.configuration)

    obs = env.reset(2)
    pprint(obs[0]["observation"])
