import enum
import logging
import os
import random
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, List, Optional, Tuple, cast

import numpy as np

import srl
from srl.base.define import KeyBindType, SpaceTypes
from srl.base.env import registration
from srl.base.env.base import EnvBase
from srl.base.env.env_run import EnvRun, SpaceBase
from srl.base.env.processor import EnvProcessor
from srl.base.spaces import ArrayDiscreteSpace, BoxSpace, DiscreteSpace
from srl.base.spaces.space import TObsSpace, TObsType

logger = logging.getLogger(__name__)


registration.register(
    id="Grid",
    entry_point=__name__ + ":Grid",
    kwargs={
        "move_reward": -0.04,
        "move_prob": 0.8,
    },
    check_duplicate=False,
)

registration.register(
    id="Grid-layer",
    entry_point=__name__ + ":GridLayer",
    kwargs={
        "move_reward": -0.04,
        "move_prob": 0.8,
    },
    check_duplicate=False,
)

registration.register(
    id="EasyGrid",
    entry_point=__name__ + ":Grid",
    kwargs={
        "move_reward": 0.0,
        "move_prob": 1.0,
        "reward_baseline_": {"episode": 100, "baseline": 0.9},
    },
    check_duplicate=False,
)


registration.register(
    id="EasyGrid-layer",
    entry_point=__name__ + ":GridLayer",
    kwargs={
        "move_reward": 0.0,
        "move_prob": 1.0,
        "reward_baseline_": {"episode": 100, "baseline": 0.9},
    },
    check_duplicate=False,
)


class Action(enum.Enum):
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3


@dataclass
class _GridBase(EnvBase[DiscreteSpace, int, TObsSpace, TObsType], Generic[TObsSpace, TObsType]):
    move_prob: float = 0.8
    move_reward: float = -0.04
    # 0.7318 ぐらい
    reward_baseline_: dict = field(default_factory=lambda: {"episode": 200, "baseline": 0.65})

    goal_reward: float = 1.0
    hole_reward: float = -1.0

    field: List[List[int]] = field(
        default_factory=lambda: [
            [9, 9, 9, 9, 9, 9],
            [9, 0, 0, 0, 1, 9],
            [9, 0, 9, 0, -1, 9],
            [9, 2, 0, 0, 0, 9],
            [9, 9, 9, 9, 9, 9],
        ],
    )

    def __post_init__(self):
        super().__init__()

        self.start_pos_list = []
        for y in range(self.H):
            for x in range(self.W):
                if self.field[y][x] == 2:
                    self.start_pos_list.append((x, y))
        assert len(self.start_pos_list) > 0, "There is no initial position. Enter '2' locations in the field."

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

        self.action_count = {}
        self.screen = None

    @property
    def action_space(self) -> DiscreteSpace:
        return DiscreteSpace(len(Action))

    @property
    def player_num(self) -> int:
        return 1

    @property
    def max_episode_steps(self) -> int:
        return 50

    @property
    def reward_range(self) -> Tuple[float, float]:
        r_min = (self.max_episode_steps - 1) * self.move_reward - 1
        r_max = 5 * self.move_reward + 1
        return r_min, r_max

    @property
    def reward_baseline(self) -> dict:
        return self.reward_baseline_

    def reset(self, seed: Optional[int] = None, **kwargs) -> Any:
        self.player_pos = random.choice(self.start_pos_list)
        self.action = Action.DOWN
        return self._create_state()

    def backup(self) -> Any:
        return self.player_pos[:]

    def restore(self, data: Any) -> None:
        self.player_pos = data[:]

    @abstractmethod
    def _create_state(self):
        raise NotImplementedError()

    def step(self, action) -> Tuple[Any, float, bool, bool]:
        action = Action(action)

        if self.training:
            k = tuple(self.player_pos)
            if k not in self.action_count:
                self.action_count[k] = {}
            if action.value not in self.action_count[k]:
                self.action_count[k][action.value] = 0
            self.action_count[k][action.value] += 1

        items = self.action_probs[action].items()
        actions = [a for a, prob in items]
        probs = [prob for a, prob in items]
        self.action = actions[np.random.choice(len(probs), p=probs)]

        self.player_pos = self._move(self.player_pos, self.action)
        reward, done = self.reward_done_func(self.player_pos)

        return self._create_state(), reward, done, False

    def render_terminal(self):
        for y in range(self.H):
            s = ""
            for x in range(self.W):
                n = self.field[y][x]
                if self.player_pos[0] == x and self.player_pos[1] == y:
                    s += "P"
                elif n == 0:  # 道
                    s += " "
                elif n == 1:  # goal
                    s += "G"
                elif n == 2:  # start
                    s += "S"
                elif n == -1:  # 穴
                    s += "X"
                elif n == 9:  # 壁
                    s += "."
                else:
                    s += str(n)
            print(s)

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

                n = self.field[y][x]
                if self.player_pos[0] == x and self.player_pos[1] == y:
                    if self.action == Action.DOWN:
                        pw.draw_image(self.screen, "player_down", x_pos, y_pos)
                    elif self.action == Action.RIGHT:
                        pw.draw_image(self.screen, "player_right", x_pos, y_pos)
                    elif self.action == Action.LEFT:
                        pw.draw_image(self.screen, "player_left", x_pos, y_pos)
                    elif self.action == Action.UP:
                        pw.draw_image(self.screen, "player_up", x_pos, y_pos)
                elif n == 0 or n == 2:  # 道
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

    def get_key_bind(self) -> Optional[KeyBindType]:
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
        return len(self.field[0])

    @property
    def H(self) -> int:
        return len(self.field)

    @property
    def actions(self) -> List[Action]:
        return [a for a in Action]

    @property
    def states(self):
        states = []
        for y in range(self.H):
            for x in range(self.W):
                states.append((x, y))
        return states

    def can_action_at(self, state):
        if self.field[state[1]][state[0]] in [0, 2]:
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
        if self.field[next_state[1]][next_state[0]] == 9:
            next_state = state

        return tuple(next_state)

    def reward_done_func(self, state):
        reward = self.move_reward
        done = False

        attribute = self.field[state[1]][state[0]]
        if attribute == 1:
            reward = self.goal_reward
            done = True
        elif attribute == -1:
            reward = self.hole_reward
            done = True

        return reward, done

    # ------------------------------------

    def calc_state_values(self, discount: float = 0.9, threshold: float = 0.000001):
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
        V = self.calc_state_values(discount, threshold)
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

        return Q

    def print_action_count(self):
        def _Q(x, y, a):
            q = 0
            if (x, y) in self.action_count:
                if a.value in self.action_count[(x, y)]:
                    q = self.action_count[(x, y)][a.value]
            return q

        print("-" * self.W * 14)
        for y in range(0, self.H):
            # 上
            s = ""
            for x in range(0, self.W):
                s += "   {:6d}    |".format(_Q(x, y, Action.UP))
            print(s)
            # 左右
            s = ""
            for x in range(0, self.W):
                s += "{:6d} {:6d}|".format(_Q(x, y, Action.LEFT), _Q(x, y, Action.RIGHT))
            print(s)
            # 下
            s = ""
            for x in range(0, self.W):
                s += "   {:6d}    |".format(_Q(x, y, Action.DOWN))
            print(s)
            print("-" * self.W * 14)

    def print_state_values(self, V):
        for y in range(0, self.H):
            s = ""
            for x in range(0, self.W):
                if (x, y) in V:
                    v = V[(x, y)]
                elif f"{x},{y}" in V:
                    v = V[f"{x},{y}"]
                else:
                    v = 0
                s += "{:9.6f} ".format(float(v))
            print(s)

    def plot_state_values(self, V):
        import matplotlib.pyplot as plt

        # 状態価値を格納する2D配列を初期化
        grid = np.zeros((self.H, self.W))

        for y in range(0, self.H):
            for x in range(0, self.W):
                if (x, y) in V:
                    grid[y, x] = V[(x, y)]
                elif f"{x},{y}" in V:
                    grid[y, x] = V[f"{x},{y}"]

        # ヒートマップを描画
        plt.figure(figsize=(8, 6))
        plt.imshow(grid, cmap="afmhot", origin="upper")
        plt.colorbar(label="State Value")
        plt.title("State Value map")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.xticks(ticks=range(self.W), labels=range(0, self.W))
        plt.yticks(ticks=range(self.H), labels=range(0, self.H))
        plt.grid(False)
        plt.show()
        plt.close()

    def print_action_values(self, Q):
        def _Q(x, y, a):
            if (x, y) in Q:
                q = Q[(x, y)][a.value]
            elif f"{x},{y}" in Q:
                q = Q[f"{x},{y}"][a.value]
            else:
                q = 0
            return float(q)

        print("-" * self.W * 14)
        for y in range(0, self.H):
            # 上
            s = ""
            for x in range(0, self.W):
                s += "   {:6.3f}    |".format(_Q(x, y, Action.UP))
            print(s)
            # 左右
            s = ""
            for x in range(0, self.W):
                s += "{:6.3f} {:6.3f}|".format(_Q(x, y, Action.LEFT), _Q(x, y, Action.RIGHT))
            print(s)
            # 下
            s = ""
            for x in range(0, self.W):
                s += "   {:6.3f}    |".format(_Q(x, y, Action.DOWN))
            print(s)
            print("-" * self.W * 14)

    def plot_action_values(self, Q):
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from matplotlib.patches import FancyArrow

        # 値の正規化用
        all_values = [value for actions in self.action_count.values() for value in actions.values()]
        norm = Normalize(vmin=min(all_values, default=0), vmax=max(all_values, default=1))
        cmap = cm.Blues

        fig, ax = plt.subplots(figsize=(self.W, self.H))
        ax.set_xlim(0, self.W)
        ax.set_ylim(0, self.H)
        ax.set_xticks(np.arange(self.W))
        ax.set_yticks(np.arange(self.H))
        ax.set_xticks(np.arange(self.W) + 0.5, minor=True)
        ax.set_yticks(np.arange(self.H) + 0.5, minor=True)
        ax.grid(which="minor", color="gray", linestyle="-", linewidth=1)
        ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
        fontsize = 12
        arrow_size = 0.15
        arrow_width = 0.08

        def _Q(x, y, a):
            if (x, y) in Q:
                q = Q[(x, y)][a]
            elif f"{x},{y}" in Q:
                q = Q[f"{x},{y}"][a]
            else:
                q = 0
            return q

        for y in range(self.H):
            for x in range(self.W):
                # 各アクションの値を取得
                up = _Q(x, y, 3)
                down = _Q(x, y, 1)
                left = _Q(x, y, 0)
                right = _Q(x, y, 2)

                # 中心座標
                cx, cy = x + 0.5, self.H - y - 0.5

                # 上矢印
                if up > 0:
                    color = cmap(norm(up))
                    ax.add_patch(FancyArrow(cx, cy, 0, arrow_size, width=arrow_width, color=color))
                    # ax.text(cx - 0, cy + 0.2, f"{up}", fontsize=fontsize, ha="center", va="bottom")

                # 下矢印
                if down > 0:
                    color = cmap(norm(down))
                    ax.add_patch(FancyArrow(cx, cy, 0, -arrow_size, width=arrow_width, color=color))
                    # ax.text(cx + 0, cy - 0.2, f"{down}", fontsize=fontsize, ha="center", va="top")

                # 左矢印
                if left > 0:
                    color = cmap(norm(left))
                    ax.add_patch(FancyArrow(cx, cy, -arrow_size, 0, width=arrow_width, color=color))
                    # ax.text(cx - 0.2, cy + 0, f"{left}", fontsize=fontsize, ha="right", va="center")

                # 右矢印
                if right > 0:
                    color = cmap(norm(right))
                    ax.add_patch(FancyArrow(cx, cy, arrow_size, 0, width=arrow_width, color=color))
                    # ax.text(cx + 0.2, cy - 0, f"{right}", fontsize=fontsize, ha="left", va="center")

        plt.show()
        plt.close()

    def plot_action_count(self):
        self.plot_action_values(self.action_count)

    def prediction_reward(self, Q, times=1000):
        rewards = []
        for _ in range(times):
            game = Grid()
            state = game.reset()
            done = False
            total_reward = 0
            while not done:
                state = tuple(state)
                if state in Q:
                    q = []
                    for a in self.actions:
                        q.append(Q[state][a.value])
                    action = np.argmax(q)
                else:
                    action = np.random.choice([a.value for a in self.actions])
                state, reward, done, _ = game.step(int(action))
                total_reward += reward

            rewards.append(total_reward)
        return np.mean(rewards)

    def verify_grid_policy(self, runner: srl.Runner):
        from srl.envs.grid import Grid

        env = srl.make_env("Grid")
        env_org = cast(Grid, env.unwrapped)
        worker = runner.make_worker()

        V, _Q = env_org.calc_action_values()
        Q = {}
        for k, v in _Q.items():
            new_k = worker.state_encode(k, env, append_recent_state=False)
            new_k = to_str_observation(new_k)
            Q[new_k] = v

        # 数ステップ回してactionを確認
        for _ in range(100):
            if env.done:
                env.reset()
                worker.on_reset(0)

            # action
            pred_a = worker.policy()

            # -----------
            # policyのアクションと最適アクションが等しいか確認
            key = to_str_observation(np.asarray(env.state))
            true_a = np.argmax(list(Q[key].values()))
            print(f"{env.state}: {true_a} == {pred_a}")
            assert true_a == pred_a
            # -----------

            # env step
            env.step(pred_a)

            # rl step
            worker.on_step()


class Grid(_GridBase[ArrayDiscreteSpace, List[int]]):
    @property
    def observation_space(self):
        return ArrayDiscreteSpace(2, low=0, high=[self.W, self.H])

    def _create_state(self):
        return list(self.player_pos)


class GridLayer(_GridBase[BoxSpace, np.ndarray]):
    @property
    def observation_space(self):
        return BoxSpace(
            shape=(self.H, self.W, 1),
            low=0,
            high=1,
            dtype=np.uint8,
            stype=SpaceTypes.IMAGE,
        )

    def _create_state(self):
        px = self.player_pos[0]
        py = self.player_pos[1]

        _field = np.zeros((self.H, self.W, 1))
        for y in range(self.H):
            for x in range(self.W):
                if y == py and x == px:
                    _field[y][x][0] = 1
        return _field


class LayerProcessor(EnvProcessor):
    def remap_observation_space(self, prev_space: SpaceBase, env_run: EnvRun, **kwargs):
        _env = cast(Grid, env_run.unwrapped)
        observation_space = BoxSpace(
            shape=(_env.H, _env.W, 1),
            low=0,
            high=1,
            dtype=np.uint8,
            stype=SpaceTypes.IMAGE,
        )
        return observation_space

    def remap_observation(self, state, prev_space: SpaceBase, new_space: SpaceBase, env_run: EnvRun, **kwargs):
        env = cast(Grid, env_run.unwrapped)
        px = env.player_pos[0]
        py = env.player_pos[1]

        _field = np.zeros((env.H, env.W, 1))
        for y in range(env.H):
            for x in range(env.W):
                if y == py and x == px:
                    _field[y][x][0] = 1
        return _field
