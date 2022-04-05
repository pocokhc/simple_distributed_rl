import enum
import json
import logging
import random
from dataclasses import dataclass
from typing import Any, List, Tuple

import gym
import gym.envs.registration
import gym.spaces
import numpy as np
from srl.base.define import EnvObservationType
from srl.base.env import EnvBase

logger = logging.getLogger(__name__)


gym.envs.registration.register(
    id="Grid-v0",
    entry_point=__name__ + ":Grid",
    kwargs={
        "move_reward": -0.04,
        "move_prob": 0.8,
        "state_type": "pos",
    },
)


gym.envs.registration.register(
    id="NeonGrid-v0",
    entry_point=__name__ + ":Grid",
    kwargs={
        "move_reward": -0.04,
        "move_prob": 0.8,
        "state_type": "neon",
    },
)

gym.envs.registration.register(
    id="ImageGrid-v0",
    entry_point=__name__ + ":Grid",
    kwargs={
        "move_reward": -0.04,
        "move_prob": 0.8,
        "state_type": "2d",
    },
)


class Action(enum.Enum):
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3


@dataclass
class Grid(EnvBase):

    move_prob: float = 0.8
    move_reward: float = -0.04
    state_type: str = "pos"

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

        self._action_space = gym.spaces.Discrete(len(Action))

        if self.state_type == "pos":
            self._observation_space = gym.spaces.Box(
                low=-0,
                high=np.maximum(self.H, self.W),
                shape=(2,),
            )
            self._observation_type = EnvObservationType.DISCRETE
        elif self.state_type in ["2d", "neon"]:
            self._observation_space = gym.spaces.Box(
                low=-1,
                high=9,
                shape=(self.H, self.W),
            )
            self._observation_type = EnvObservationType.SHAPE2
        else:
            raise ValueError()

    # override
    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    # override
    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    # override
    @property
    def observation_type(self) -> EnvObservationType:
        return self._observation_type

    # override
    @property
    def max_episode_steps(self) -> int:
        return 50

    # override
    def fetch_valid_actions(self) -> List[int]:
        return [e.value for e in Action]

    # override
    def reset(self) -> Any:
        self.player_pos = (1, 3)
        self.return_state = self._create_field(self.player_pos, self.state_type)
        return self.return_state

    def _create_field(self, player_pos, state_type) -> Any:
        if state_type == "pos":
            return tuple(player_pos)

        field = json.loads(json.dumps(self.base_field))  # deepcopy

        px = player_pos[0]
        py = player_pos[1]
        field[py][px] = 2

        if state_type == "neon":
            # 9は3～9にランダムに変わる
            for y in range(self.H):
                for x in range(self.W):
                    if field[y][x] == 9:
                        field[y][x] = random.randint(-1, 9)
        return tuple(map(tuple, field))

    # override
    def step(self, action_: int) -> Tuple[Any, float, bool, dict]:
        action = Action(action_)

        items = self.action_probs[action].items()
        actions = [a for a, prob in items]
        probs = [prob for a, prob in items]
        action = actions[np.random.choice(len(probs), p=probs)]

        self.player_pos = self._move(self.player_pos, action)
        reward, done = self.reward_done_func(self.player_pos)

        self.return_state = self._create_field(self.player_pos, self.state_type)
        return self.return_state, reward, done, {}

    # override
    def render(self, mode="human"):
        if self.state_type == "pos":
            state = self._create_field(self.player_pos, "2d")
        else:
            state = self.return_state
        for y in range(self.H):
            s = ""
            for x in range(self.W):
                n = state[y][x]
                if n == 2:  # player
                    s += "P"
                elif n == 0:  # 道
                    s += "."
                elif n == 1:  # goal
                    s += "G"
                elif n == -1:  # 穴
                    s += "X"
                else:
                    s += str(n)
            print(s)
        print("")

    # override
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

    # override
    def backup(self) -> Any:
        return json.dumps(
            [
                self.player_pos,
                self.return_state,
            ]
        )

    # override
    def restore(self, data: Any) -> None:
        d = json.loads(data)
        self.player_pos = d[0]
        self.return_state = d[1]

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
                s = self._create_field((x, y), self.state_type)
                states.append(s)
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

    # ------------------------------------

    def value_iteration(self, gamma: float = 0.9, threshold: float = 0.000001):
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
                            gain = reward + gamma * V[next_state]
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

    def calc_action_values(self, gamma: float = 0.9, threshold: float = 0.000001):
        V = self.value_iteration(gamma, threshold)
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
                        gain = reward + gamma * V[next_state]
                    r += state_prob * gain
                Q[s][a.value] = r

        return V, Q

    def print_state_values(self, V):
        for y in range(1, self.H - 1):
            s = ""
            for x in range(1, self.W - 1):
                state = self._create_field((x, y), self.state_type)
                if state in V:
                    v = V[state]
                else:
                    v = 0
                s += "{:9.6f} ".format(v)
            print(s)

    def print_action_values(self, Q):
        def _Q(x, y, a):
            s = self._create_field((x, y), self.state_type)
            if s in Q:
                return Q[s][a.value]
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
                    action = random.choice(self.actions)
                state, reward, done, _ = game.step(action)
                total_reward += reward

            rewards.append(total_reward)
        return np.mean(rewards)


if __name__ == "__main__":

    game = Grid()
    game.reset()
    done = False
    total_reward = 0
    step = 0
    game.render()

    while not done:
        valid_actions = game.fetch_valid_actions()
        action = random.choice(valid_actions)
        state, reward, done, _ = game.step(action)
        total_reward += reward
        step += 1
        print(f"step {step}, action {action}")
        game.render()

    print(total_reward)
