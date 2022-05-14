import logging
import pickle
from typing import Any, List, Optional, Tuple

import numpy as np
from srl.base.define import EnvObservationType, RLObservationType
from srl.base.env import registration
from srl.base.env.base import SpaceBase
from srl.base.env.genre import TurnBase2Player
from srl.base.env.processor import Processor
from srl.base.env.spaces import BoxSpace, DiscreteSpace
from srl.base.rl.algorithms.rulebase import RuleBaseWorker
from srl.base.rl.base import RLWorker

logger = logging.getLogger(__name__)

try:
    import kaggle_environments
    from kaggle_environments.envs.connectx.connectx import negamax_agent

    registration.register(
        id="ConnectX",
        entry_point=__name__ + ":ConnectX",
    )
except ImportError:
    logger.debug("kaggle env didn't read.")


class ConnectX(TurnBase2Player):
    def __init__(self):
        super().__init__()

        self.env = kaggle_environments.make("connectx", debug=True)
        self._action_num = self.env.configuration["columns"]
        self.W = self.env.configuration["columns"]
        self.H = self.env.configuration["rows"]

        self._player_index = 0

    @property
    def action_space(self) -> DiscreteSpace:
        return DiscreteSpace(self._action_num)

    @property
    def observation_space(self) -> SpaceBase:
        return BoxSpace(low=0, high=2, shape=(self.H * self.W,))

    @property
    def observation_type(self) -> EnvObservationType:
        return EnvObservationType.DISCRETE

    @property
    def max_episode_steps(self) -> int:
        return self.W * self.H + 2

    @property
    def player_index(self) -> int:
        return self._player_index

    def reset_turn(self) -> np.ndarray:
        states = self.env.reset()
        self.board = states[0].observation["board"]
        self._player_index = 0
        return np.array(self.board)

    def step_turn(self, action: int) -> Tuple[np.ndarray, float, float, bool, dict]:
        states = self.env.step([action, action])
        self.board = states[0].observation["board"]
        reward1 = states[0].reward
        reward2 = states[1].reward
        done = self.env.done

        if self._player_index == 0:
            self._player_index = 1
        else:
            self._player_index = 0

        return np.array(self.board), reward1, reward2, done, {}

    def get_invalid_actions(self, player_index: int) -> List[int]:
        invalid_actions = [a for a in range(self.action_space.n) if self.board[a] != 0]
        return invalid_actions

    def render_terminal(self, **kwargs) -> None:
        print(self.env.render(mode="ansi"))

    def render_gui(self, **kwargs) -> None:
        raise NotImplementedError()

    def render_rgb_array(self, **kwargs) -> np.ndarray:
        raise NotImplementedError()

    def backup(self) -> Any:
        return [
            pickle.dumps(self.env),
            self.board[:],
            self._player_index,
        ]

    def restore(self, data: Any) -> None:
        self.env = pickle.loads(data[0])
        self.board = data[1][:]
        self._player_index = data[2]

    def make_worker(self, name: str) -> Optional[RLWorker]:
        if name == "negamax":
            return NegaMax()
        return None


class NegaMax(RuleBaseWorker):
    def __init__(self):
        pass  #

    def call_on_reset(self, env) -> None:
        pass  #

    def call_policy(self, env: ConnectX) -> Any:
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
            shape=(3, env.H, env.W),
        )
        return observation_space, EnvObservationType.SHAPE3

    def process_observation(self, observation: np.ndarray, env: ConnectX) -> np.ndarray:
        # Layer0: player1 field (0 or 1)
        # Layer1: player2 field (0 or 1)
        # Layer2: player_index (all0 or all1)
        _field = np.zeros((3, env.H, env.W))
        for y in range(env.H):
            for x in range(env.W):
                idx = x + y * env.W
                if env.board[idx] == 1:
                    _field[0][y][x] = 1
                elif env.board[idx] == 2:
                    _field[1][y][x] = 1
        _field[2] = env.player_index
        return _field
