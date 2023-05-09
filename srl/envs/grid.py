import enum
import logging
import os
from dataclasses import dataclass
from typing import Any, List, Tuple, cast

import numpy as np

from srl.base.define import EnvObservationType, KeyBindType, RLObservationType
from srl.base.env import registration
from srl.base.env.base import EnvRun, SpaceBase
from srl.base.env.genre import SinglePlayEnv
from srl.base.env.spaces import ArrayDiscreteSpace, BoxSpace, DiscreteSpace
from srl.base.rl.processor import Processor

logger = logging.getLogger(__name__)


registration.register(
    id="Grid",
    entry_point=__name__ + ":Grid",
    kwargs={
        "move_reward": -0.04,
        "move_prob": 0.8,
        "reward_baseline": 0.65,  # # 0.7318 ぐらい
    },
)

registration.register(
    id="EasyGrid",
    entry_point=__name__ + ":Grid",
    kwargs={
        "move_reward": 0.0,
        "move_prob": 1.0,
        "reward_baseline": 0.9,
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
    reward_baseline: float = 0.6

    goal_reward: float = 1.0
    hole_reward: float = -1.0

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

        self.screen = None

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

    @property
    def reward_info(self) -> dict:
        r_min = (self.max_episode_steps - 1) * self.move_reward - 1
        r_max = 5 * self.move_reward + 1
        return {
            "range": (r_min, r_max),
            "baseline": self.reward_baseline,
        }

    def call_reset(self) -> Tuple[List[int], dict]:
        self.player_pos = (1, 3)
        self.action = Action.DOWN
        return list(self.player_pos), {}

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

    def render_rgb_array(self, **kwargs) -> np.ndarray:
        from srl.utils import pygame_wrapper as pw

        cell_size = 32
        WIDTH = cell_size * self.W
        HEIGHT = cell_size * self.H
        if self.screen is None:
            self.screen = pw.create_surface(WIDTH, HEIGHT)

            pw.load_image("cell", os.path.join(os.path.dirname(__file__), "img/cell.png"))
            pw.load_image("goal", os.path.join(os.path.dirname(__file__), "img/goal.png"))
            pw.load_image("hole", os.path.join(os.path.dirname(__file__), "img/hole.png"))
            pw.load_image("wall", os.path.join(os.path.dirname(__file__), "img/wall.png"))
            pw.load_image("player_down", os.path.join(os.path.dirname(__file__), "img/player_down.png"))
            pw.load_image("player_left", os.path.join(os.path.dirname(__file__), "img/player_left.png"))
            pw.load_image("player_right", os.path.join(os.path.dirname(__file__), "img/player_right.png"))
            pw.load_image("player_up", os.path.join(os.path.dirname(__file__), "img/player_up.png"))

        pw.draw_fill(self.screen, color=(255, 255, 255))

        for y in range(self.H):
            for x in range(self.W):
                x_pos = x * cell_size
                y_pos = y * cell_size
                pw.draw_image(self.screen, "cell", x_pos, y_pos)

                n = self.base_field[y][x]
                if self.player_pos[0] == x and self.player_pos[1] == y:
                    if self.action == Action.DOWN:
                        pw.draw_image(self.screen, "player_down", x_pos, y_pos)
                    elif self.action == Action.RIGHT:
                        pw.draw_image(self.screen, "player_right", x_pos, y_pos)
                    elif self.action == Action.LEFT:
                        pw.draw_image(self.screen, "player_left", x_pos, y_pos)
                    elif self.action == Action.UP:
                        pw.draw_image(self.screen, "player_up", x_pos, y_pos)
                elif n == 0:  # 道
                    pass
                elif n == 1:  # goal
                    pw.draw_image(self.screen, "goal", x_pos, y_pos)
                elif n == -1:  # 穴
                    pw.draw_image(self.screen, "hole", x_pos, y_pos)
                elif n == 9:  # 壁
                    pw.draw_image(self.screen, "wall", x_pos, y_pos)

        return pw.get_rgb_array(self.screen)

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

    def get_key_bind(self) -> KeyBindType:
        return {
            "": Action.LEFT.value,
            "a": Action.LEFT.value,
            "d": Action.RIGHT.value,
            "w": Action.UP.value,
            "s": Action.DOWN.value,
        }

    @property
    def render_interval(self) -> float:
        return 1000 / 1

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
            reward = self.goal_reward
            done = True
        elif attribute == -1:
            reward = self.hole_reward
            done = True

        return reward, done

    # ------------------------------------

    def value_iteration(self, discount: float = 0.9, threshold: float = 0.000001):
        V = {s: 0 for s in self.states}

        # 学習
        i = 0
        for i in range(100):  # for safety
            delta = 0

            # 全状態をループ
            for s in self.states:
                # アクションが実行できる状態のみ実施
                if not self.can_action_at(s):
                    continue

                # 各アクションでの報酬期待値を計算
                expected_reward = []
                for a in self.actions:
                    # 報酬期待値を計算
                    r = 0
                    for next_state, state_prob in self.transitions_at(s, a).items():
                        reward, done = self.reward_done_func(next_state)
                        if done:
                            gain = reward
                        else:
                            gain = reward + discount * V[next_state]
                        r += state_prob * gain
                    expected_reward.append(r)

                # greedyな方策
                maxq = np.max(expected_reward)
                delta = max(delta, abs(V[s] - maxq))  # 学習打ち切り用に差分を保存
                V[s] = maxq

            # 更新差分が閾値以下になったら学習終了
            if delta < threshold:
                break

        print("training end. iterator count: {}".format(i))
        return V

    def calc_action_values(self, discount: float = 0.9, threshold: float = 0.000001):
        V = self.value_iteration(discount, threshold)
        Q = {}

        # 全状態をループ
        for s in self.states:
            # アクションが実行できる状態のみ実施
            if not self.can_action_at(s):
                continue

            Q[s] = {}

            # 各アクションにて、選択後の状態価値の期待値を計算
            for a in self.actions:
                r = 0
                for next_state, state_prob in self.transitions_at(s, a).items():
                    reward, done = self.reward_done_func(next_state)
                    if done:
                        gain = reward
                    else:
                        gain = reward + discount * V[next_state]
                    r += state_prob * gain
                Q[s][a.value] = r

        return V, Q

    def print_state_values(self, V):
        for y in range(1, self.H - 1):
            s = ""
            for x in range(1, self.W - 1):
                if (x, y) in V:
                    v = V[(x, y)]
                else:
                    v = 0
                s += "{:9.6f} ".format(v)
            print(s)

    def print_action_values(self, Q):
        def _Q(x, y, a):
            if (x, y) in Q:
                return Q[(x, y)][a.value]
            else:
                return 0

        print("-" * 60)
        for y in range(1, self.H - 1):
            # 上
            s = ""
            for x in range(1, self.W - 1):
                s += "   {:6.3f}    |".format(_Q(x, y, Action.UP))
            print(s)
            # 左右
            s = ""
            for x in range(1, self.W - 1):
                s += "{:6.3f} {:6.3f}|".format(_Q(x, y, Action.LEFT), _Q(x, y, Action.RIGHT))
            print(s)
            # 下
            s = ""
            for x in range(1, self.W - 1):
                s += "   {:6.3f}    |".format(_Q(x, y, Action.DOWN))
            print(s)
            print("-" * 60)

    def reward_prediction(self, Q, times=1000):
        rewards = []
        for _ in range(times):
            game = Grid()
            state = game.reset()
            done = False
            total_reward = 0
            while not done:
                if state in Q:
                    q = []
                    for a in self.actions:
                        q.append(Q[state][a.value])
                    action = np.argmax(q)
                else:
                    action = np.random.choice(self.actions)
                state, reward, done, _ = game.step(action)
                total_reward += reward

            rewards.append(total_reward)
        return np.mean(rewards)


class LayerProcessor(Processor):
    def change_observation_info(
        self,
        env_observation_space: SpaceBase,
        env_observation_type: EnvObservationType,
        rl_observation_type: RLObservationType,
        env: EnvRun,
    ) -> Tuple[SpaceBase, EnvObservationType]:
        _env = cast(Grid, env.get_original_env())
        observation_space = BoxSpace(
            low=0,
            high=1,
            shape=(1, _env.H, _env.W),
        )
        return observation_space, EnvObservationType.SHAPE3

    def process_observation(self, observation: np.ndarray, env: EnvRun) -> np.ndarray:
        _env = cast(Grid, env.get_original_env())

        px = _env.player_pos[0]
        py = _env.player_pos[1]

        _field = np.zeros((1, _env.H, _env.W))
        for y in range(_env.H):
            for x in range(_env.W):
                if y == py and x == px:
                    _field[0][y][x] = 1
        return _field
