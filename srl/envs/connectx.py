import logging
import math
import pickle
from typing import Any, Callable, List, Optional, Tuple, cast

import gym
import gym.envs.registration
import gym.spaces
import numpy as np
from srl.base.define import EnvObservationType
from srl.base.env import registration
from srl.base.env.genre.turnbase import TurnBase2PlayerActionDiscrete
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
except ModuleNotFoundError:
    logger.debug("kaggle env didn't read.")


class ConnectX(TurnBase2PlayerActionDiscrete):
    def __init__(self):
        super().__init__()

        self.env = kaggle_environments.make("connectx", debug=True)
        self._action_num = self.env.configuration["columns"]
        self.W = self.env.configuration["columns"]
        self.H = self.env.configuration["rows"]

        self._player_index = 0

    @property
    def action_num(self) -> int:
        return self._action_num

    @property
    def observation_space(self) -> gym.spaces.Space:
        return gym.spaces.Box(low=0, high=2, shape=(self.H * self.W + 1,))

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
        return np.array(self.board + [self.player_index])

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

        return np.array(self.board + [self.player_index]), reward1, reward2, done, {}

    def get_invalid_actions(self, player_index: int) -> List[int]:
        nvalid_actions = [a for a in range(self.action_num) if self.board[a] != 0]
        return nvalid_actions

    def render_terminal(self, **kwargs) -> None:
        print(self.env.render(mode="ansi"))

    def render_gui(self, **kwargs) -> None:
        raise NotImplementedError()

    def render_rgb_array(self, **kwargs) -> np.ndarray:
        raise NotImplementedError()

    def backup(self) -> Any:
        return pickle.dumps(self.env)

    def restore(self, data: Any) -> None:
        self.env = pickle.loads(data)

    def make_worker(self, name: str) -> Optional[RLWorker]:
        if name == "negamax":
            return NegaMax()
        return None


class NegaMax(RuleBaseWorker):
    def __init__(self):
        pass

    def call_on_reset(self, env) -> None:
        pass

    def call_policy(self, env: ConnectX) -> Any:
        observation = env.env.state[0]["observation"]
        configuration = env.env.configuration
        action = negamax_agent(observation, configuration)
        return action

    def call_render(self, env: ConnectX) -> None:
        pass
