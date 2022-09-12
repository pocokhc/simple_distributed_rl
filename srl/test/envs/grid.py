import enum
import json
import logging
import random
from dataclasses import dataclass
from typing import Any, List, Tuple, cast

import numpy as np
from srl.base.define import EnvObservationType, RLObservationType
from srl.base.env import registration
from srl.base.env.base import EnvRun, SpaceBase
from srl.base.env.genre import SinglePlayEnv
from srl.base.env.spaces import ArrayDiscreteSpace, BoxSpace, DiscreteSpace
from srl.base.rl.processor import Processor

logger = logging.getLogger(__name__)


registration.register(
    id="TestGrid",
    entry_point=__name__ + ":Grid",
    kwargs={
        "move_reward": -0.04,
        "move_prob": 0.8,
    },
)


class Action(enum.Enum):
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3


@dataclass
class Grid(SinglePlayEnv):

    move_prob: float = 0.8
    move_reward: float = -0.04

    def __post_init__(self):
        self.base_field = [
            [9, 9, 9, 9, 9, 9],
            [9, 0, 0, 0, 1, 9],
            [9, 0, 9, 0, -1, 9],
            [9, 0, 0, 0, 0, 9],
            [9, 9, 9, 9, 9, 9],
        ]

        # 遷移確率
        self.action_probs = {
            Action.UP: {
                Action.UP: self.move_prob,
                Action.DOWN: 0,
                Action.RIGHT: (1 - self.move_prob) / 2,
                Action.LEFT: (1 - self.move_prob) / 2,
            },
            Action.DOWN: {
                Action.UP: 0,
                Action.DOWN: self.move_prob,
                Action.RIGHT: (1 - self.move_prob) / 2,
                Action.LEFT: (1 - self.move_prob) / 2,
            },
            Action.RIGHT: {
                Action.UP: (1 - self.move_prob) / 2,
                Action.DOWN: (1 - self.move_prob) / 2,
                Action.RIGHT: self.move_prob,
                Action.LEFT: 0,
            },
            Action.LEFT: {
                Action.UP: (1 - self.move_prob) / 2,
                Action.DOWN: (1 - self.move_prob) / 2,
                Action.RIGHT: 0,
                Action.LEFT: self.move_prob,
            },
        }

    @property
    def action_space(self) -> SpaceBase:
        return DiscreteSpace(len(Action))

    @property
    def observation_space(self) -> SpaceBase:
        return ArrayDiscreteSpace(2, low=0, high=[self.W, self.H])

    @property
    def observation_type(self) -> EnvObservationType:
        return EnvObservationType.DISCRETE

    @property
    def max_episode_steps(self) -> int:
        return 50

    def call_reset(self) -> List[int]:
        self.player_pos = (1, 3)
        self.action = Action.DOWN
        return list(self.player_pos)

    def backup(self) -> Any:
        return self.player_pos

    def restore(self, data: Any) -> None:
        self.player_pos = data

    def call_step(self, action_: int) -> Tuple[List[int], float, bool, dict]:
        action = Action(action_)

        items = self.action_probs[action].items()
        actions = [a for a, prob in items]
        probs = [prob for a, prob in items]
        self.action = actions[np.random.choice(len(probs), p=probs)]

        self.player_pos = self._move(self.player_pos, self.action)
        reward, done = self.reward_done_func(self.player_pos)

        return list(self.player_pos), reward, done, {}

    def render_terminal(self):
        for y in range(self.H):
            s = ""
            for x in range(self.W):
                n = self.base_field[y][x]
                if self.player_pos[0] == x and self.player_pos[1] == y:
                    s += "P"
                elif n == 0:  # 道
                    s += " "
                elif n == 1:  # goal
                    s += "G"
                elif n == -1:  # 穴
                    s += "X"
                elif n == 9:  # 壁
                    s += "."
                else:
                    s += str(n)
            print(s)
        print("")

    def action_to_str(self, action) -> str:
        if Action.DOWN.value == action:
            return "↓"
        if Action.LEFT.value == action:
            return "←"
        if Action.RIGHT.value == action:
            return "→"
        if Action.UP.value == action:
            return "↑"
        return str(action)

    # ------------------------------------
    @property
    def W(self) -> int:
        return len(self.base_field[0])

    @property
    def H(self) -> int:
        return len(self.base_field)

    @property
    def actions(self):
        return [a for a in Action]

    @property
    def states(self):
        states = []
        for y in range(self.H):
            for x in range(self.W):
                states.append((x, y))
        return states

    def can_action_at(self, state):
        if self.base_field[state[1]][state[0]] == 0:
            return True
        else:
            return False

    # 次の状態遷移確率
    def transitions_at(self, state, action):
        transition_probs = {}
        for a in self.actions:
            prob = self.action_probs[action][a]
            next_state = self._move(state, a)
            if next_state not in transition_probs:
                transition_probs[next_state] = 0
            transition_probs[next_state] += prob
        return transition_probs

    def _move(self, state, action):
        next_state = list(state)

        if action == Action.UP:
            next_state[1] -= 1
        elif action == Action.DOWN:
            next_state[1] += 1
        elif action == Action.LEFT:
            next_state[0] -= 1
        elif action == Action.RIGHT:
            next_state[0] += 1
        else:
            raise ValueError()

        # check
        if not (0 <= next_state[0] < self.W):
            next_state = state
        if not (0 <= next_state[1] < self.H):
            next_state = state

        # 移動できない
        if self.base_field[next_state[1]][next_state[0]] == 9:
            next_state = state

        return tuple(next_state)

    def reward_done_func(self, state):
        reward = self.move_reward
        done = False

        attribute = self.base_field[state[1]][state[0]]
        if attribute == 1:
            reward = 1
            done = True
        elif attribute == -1:
            reward = -1
            done = True

        return reward, done


class LayerProcessor(Processor):
    def change_observation_info(
        self,
        env_observation_space: SpaceBase,
        env_observation_type: EnvObservationType,
        rl_observation_type: RLObservationType,
        _env: EnvRun,
    ) -> Tuple[SpaceBase, EnvObservationType]:
        env = cast(Grid, _env.get_original_env())
        observation_space = BoxSpace(
            low=0,
            high=1,
            shape=(1, env.H, env.W),
        )
        return observation_space, EnvObservationType.SHAPE3

    def process_observation(self, observation: np.ndarray, _env: EnvRun) -> np.ndarray:
        env = cast(Grid, _env.get_original_env())

        px = env.player_pos[0]
        py = env.player_pos[1]

        _field = np.zeros((1, env.H, env.W))
        for y in range(env.H):
            for x in range(env.W):
                if y == py and x == px:
                    _field[0][y][x] = 1
        return _field
