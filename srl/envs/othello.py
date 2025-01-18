import logging
import pickle
import time
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, List, Optional, Tuple, cast

import numpy as np

from srl.base.define import SpaceTypes
from srl.base.env.base import EnvBase
from srl.base.env.env_run import EnvRun
from srl.base.env.registration import register
from srl.base.rl.algorithms.env_worker import EnvWorker
from srl.base.spaces.array_discrete import ArrayDiscreteSpace
from srl.base.spaces.box import BoxSpace
from srl.base.spaces.discrete import DiscreteSpace
from srl.base.spaces.space import TObsSpace, TObsType

logger = logging.getLogger(__name__)

register(
    id="Othello",
    entry_point=__name__ + ":Othello",
    kwargs={"W": 8, "H": 8},
    check_duplicate=False,
)
register(
    id="Othello6x6",
    entry_point=__name__ + ":Othello",
    kwargs={"W": 6, "H": 6},
    check_duplicate=False,
)
register(
    id="Othello4x4",
    entry_point=__name__ + ":Othello",
    kwargs={"W": 4, "H": 4},
    check_duplicate=False,
)

register(
    id="Othello-layer",
    entry_point=__name__ + ":OthelloLayer",
    kwargs={"W": 8, "H": 8},
    check_duplicate=False,
)
register(
    id="Othello6x6-layer",
    entry_point=__name__ + ":OthelloLayer",
    kwargs={"W": 6, "H": 6},
    check_duplicate=False,
)
register(
    id="Othello4x4-layer",
    entry_point=__name__ + ":OthelloLayer",
    kwargs={
        "W": 4,
        "H": 4,
        # [0.3, 0.9] ぐらい
        "reward_baseline_": [
            {"episode": 50, "players": [None, "random"], "baseline": [0.1, None]},
            {"episode": 50, "players": ["random", None], "baseline": [None, 0.5]},
        ],
    },
    check_duplicate=False,
)


@dataclass
class _OthelloBase(EnvBase[DiscreteSpace, int, TObsSpace, TObsType], Generic[TObsSpace, TObsType]):
    W: int = 8
    H: int = 8
    reward_baseline_: list = field(default_factory=lambda: [])

    def __post_init__(self):
        super().__init__()
        self.screen = None

    def get_field(self, x: int, y: int) -> int:
        if x < 0:
            return 9
        if x >= self.W:
            return 9
        if y < 0:
            return 9
        if y >= self.H:
            return 9
        return self.field[self.W * y + x]

    def set_field(self, x: int, y: int, n: int):
        self.field[self.W * y + x] = n

    def pos(self, x: int, y: int) -> int:
        return self.W * y + x

    def pos_decode(self, a: int) -> Tuple[int, int]:
        return a % self.W, a // self.W

    # ---------------------------------

    @property
    def action_space(self) -> DiscreteSpace:
        return DiscreteSpace(self.W * self.H)

    @property
    def player_num(self) -> int:
        return 2

    @property
    def max_episode_steps(self) -> int:
        return self.W * self.H

    @property
    def reward_range(self) -> Tuple[float, float]:
        return -1, 1

    @property
    def reward_baseline(self) -> dict:
        return self.reward_baseline_

    def reset(self, *, seed: Optional[int] = None, **kwargs) -> Any:
        self.action = 0

        self.next_player = 0
        self.field = [0] * (self.W * self.H)
        center_x = int(self.W / 2) - 1
        center_y = int(self.H / 2) - 1
        self.set_field(center_x, center_y, 1)
        self.set_field(center_x + 1, center_y + 1, 1)
        self.set_field(center_x + 1, center_y, -1)
        self.set_field(center_x, center_y + 1, -1)
        self.movable_dirs = [
            self._calc_movable_dirs(0),
            self._calc_movable_dirs(1),
        ]

        return self._create_obs()

    def backup(self) -> Any:
        return pickle.dumps(
            [
                self.next_player,
                self.W,
                self.H,
                self.field,
                self.movable_dirs,
            ]
        )

    def restore(self, data: Any) -> None:
        d = pickle.loads(data)
        self.next_player = d[0]
        self.W = d[1]
        self.H = d[2]
        self.field = d[3]
        self.movable_dirs = d[4]

    @abstractmethod
    def _create_obs(self):
        raise NotImplementedError()

    def _calc_movable_dirs(self, player_index) -> List[List[int]]:
        my_color = 1 if player_index == 0 else -1
        enemy_color = -my_color

        dirs_list = [[] for _ in range(self.W * self.H)]
        for y in range(self.H):
            for x in range(self.W):
                # 石が置ける場所のみ
                if self.get_field(x, y) != 0:
                    continue

                # (x, y, dir) dirはテンキーに対応
                for diff_x, diff_y, dir_ in [
                    (-1, 1, 1),
                    (0, 1, 2),
                    (1, 1, 3),
                    (1, 0, 6),
                    (1, -1, 9),
                    (0, -1, 8),
                    (-1, -1, 7),
                    (-1, 0, 4),
                ]:
                    tmp_x = x + diff_x
                    tmp_y = y + diff_y

                    # 1つは相手の駒がある
                    if self.get_field(tmp_x, tmp_y) != enemy_color:
                        continue
                    tmp_x += diff_x
                    tmp_y += diff_y

                    # 相手の駒移動
                    while self.get_field(tmp_x, tmp_y) == enemy_color:
                        tmp_x += diff_x
                        tmp_y += diff_y

                    # 相手の駒の後に自分の駒があるか
                    if self.get_field(tmp_x, tmp_y) == my_color:
                        dirs_list[self.pos(x, y)].append(dir_)

        return dirs_list

    def step(self, action) -> Tuple[Any, List[float], bool, bool]:
        self.action = action

        # --- error action
        if len(self.movable_dirs[self.next_player][action]) == 0:
            if self.next_player == 0:
                return self._create_obs(), [-1.0, 0.0], True, False
            else:
                return self._create_obs(), [0.0, -1.0], True, False

        # --- step
        self._step(action)

        # --- 終了判定
        enemy_player = 1 if self.next_player == 0 else 0
        my_player = 0 if self.next_player == 0 else 1
        enemy_put_num = self.action_space.n - len(self.get_invalid_actions(enemy_player))
        my_put_num = self.action_space.n - len(self.get_invalid_actions(my_player))
        # 互いに置けないなら終了
        if enemy_put_num == 0 and my_put_num == 0:
            p1_count = len([f for f in self.field if f == 1])
            p2_count = len([f for f in self.field if f == -1])
            if p1_count > p2_count:
                r1 = 1.0
                r2 = -1.0
            elif p1_count < p2_count:
                r1 = -1.0
                r2 = 1.0
            else:
                r1 = r2 = 0.0
            self.info["P1"] = p1_count
            self.info["P2"] = p2_count
            return self._create_obs(), [r1, r2], True, False

        # 相手が置けないならpass
        if enemy_put_num == 0:
            return self._create_obs(), [0.0, 0.0], False, False

        # 手番交代
        self.next_player = enemy_player
        return self._create_obs(), [0.0, 0.0], False, False

    def _step(self, action):
        # --- update
        x, y = self.pos_decode(action)
        my_color = 1 if self.next_player == 0 else -1
        self.field[action] = my_color

        # 移動方向はテンキー
        move_diff = {
            1: (-1, 1),
            2: (0, 1),
            3: (1, 1),
            6: (1, 0),
            9: (1, -1),
            8: (0, -1),
            7: (-1, -1),
            4: (-1, 0),
        }
        for movable_dir in self.movable_dirs[self.next_player][action]:
            diff_x, diff_y = move_diff[movable_dir]
            tmp_x = x + diff_x
            tmp_y = y + diff_y
            while self.get_field(tmp_x, tmp_y) != my_color:
                a = self.pos(tmp_x, tmp_y)
                self.field[a] = my_color
                tmp_x += diff_x
                tmp_y += diff_y

        # 置ける場所を更新
        self.movable_dirs = [
            self._calc_movable_dirs(0),
            self._calc_movable_dirs(1),
        ]

    def get_invalid_actions(self, player_index) -> List[int]:
        return [a for a in range(self.H * self.W) if len(self.movable_dirs[player_index][a]) == 0]

    def render_terminal(self, **kwargs) -> None:
        invalid_actions = self.get_invalid_actions(self.next_player)
        p1_count = len([f for f in self.field if f == 1])
        p2_count = len([f for f in self.field if f == -1])

        print("-" * (1 + self.W * 3))
        for y in range(self.H):
            s = "|"
            for x in range(self.W):
                a = self.pos(x, y)
                if self.field[a] == 1:
                    if self.action == a:
                        s += "*o|"
                    else:
                        s += " o|"
                elif self.field[a] == -1:
                    if self.action == a:
                        s += "*x|"
                    else:
                        s += " x|"
                elif a not in invalid_actions:
                    s += "{:2d}|".format(a)
                else:
                    s += "  |"
            print(s)
        print("-" * (1 + self.W * 3))
        print(f"O: {p1_count}, X: {p2_count}")
        if self.next_player == 0:
            print("next player: O")
        else:
            print("next player: X")

    def render_rgb_array(self, **kwargs) -> Optional[np.ndarray]:
        from srl.utils import pygame_wrapper as pw

        WIDTH = 400
        HEIGHT = 400
        if self.screen is None:
            self.screen = pw.create_surface(WIDTH, HEIGHT)

        w_margin = 10
        h_margin = 10
        cell_w = int((WIDTH - w_margin * 2) / self.W)
        cell_h = int((HEIGHT - h_margin * 2) / self.H)
        invalid_actions = self.get_invalid_actions(self.next_player)

        pw.draw_fill(self.screen, color=(255, 255, 255))

        # --- cell
        for y in range(self.H):
            for x in range(self.W):
                center_x = int(w_margin + x * cell_w + cell_w / 2)
                center_y = int(h_margin + y * cell_h + cell_h / 2)
                left_top_x = w_margin + x * cell_w
                left_top_y = h_margin + y * cell_h

                pw.draw_box(
                    self.screen,
                    left_top_x,
                    left_top_y,
                    cell_w,
                    cell_h,
                    fill_color=(0, 200, 0),
                    width=4,
                    line_color=(0, 0, 0),
                )

                a = x + y * self.W
                if self.field[a] == 1:  # o
                    if self.action == a:
                        width = 4
                        line_color = (200, 0, 0)
                    else:
                        width = 0
                        line_color = (0, 0, 0)
                    pw.draw_circle(
                        self.screen,
                        center_x,
                        center_y,
                        int(cell_w * 0.3),
                        filled=True,
                        fill_color=(0, 0, 0),
                        width=width,
                        line_color=line_color,
                    )
                elif self.field[a] == -1:  # x
                    if self.action == a:
                        width = 4
                        line_color = (200, 0, 0)
                    else:
                        width = 0
                        line_color = (0, 0, 0)
                    pw.draw_circle(
                        self.screen,
                        center_x,
                        center_y,
                        int(cell_w * 0.3),
                        filled=True,
                        fill_color=(255, 255, 255),
                        width=width,
                        line_color=line_color,
                    )
                elif a not in invalid_actions:
                    if self.next_player == 0:
                        color = (0, 0, 0)
                    else:
                        color = (255, 255, 255)
                    pw.draw_circle(
                        self.screen,
                        center_x,
                        center_y,
                        int(cell_w * 0.1),
                        filled=True,
                        fill_color=color,
                    )

        return pw.get_rgb_array(self.screen)

    @property
    def render_interval(self) -> float:
        return 1000 / 1

    def make_worker(self, name: str, **kwargs) -> Optional[EnvWorker]:
        if name == "cpu":
            return Cpu(**kwargs)
        return None


@dataclass
class Othello(_OthelloBase[ArrayDiscreteSpace, List[int]]):
    @property
    def observation_space(self):
        return ArrayDiscreteSpace(self.W * self.H, low=-1, high=1)

    def _create_obs(self):
        return self.field


@dataclass
class OthelloLayer(_OthelloBase[BoxSpace, np.array]):
    @property
    def observation_space(self):
        return BoxSpace(low=0, high=1, shape=(self.H, self.W, 2), stype=SpaceTypes.IMAGE)

    def _create_obs(self):
        # Layer0: my_player field (0 or 1)
        # Layer1: enemy_player field (0 or 1)
        if self.next_player == 0:
            my_field = 1
            enemy_field = -1
        else:
            my_field = -1
            enemy_field = 1
        _field = np.zeros((self.H, self.W, 2))
        for y in range(self.H):
            for x in range(self.W):
                idx = x + y * self.W
                if self.field[idx] == my_field:
                    _field[y][x][0] = 1
                elif self.field[idx] == enemy_field:
                    _field[y][x][1] = 1
        return _field


class Cpu(EnvWorker):
    cache = {}

    def call_on_reset(self, env: EnvRun) -> dict:
        _env = cast(Othello, env.unwrapped)
        self.max_depth = 2
        self.eval_field = None

        if _env.W == 8:
            self.max_depth = 2
            self.eval_field = [
                [30, -12, 0, -1, -1, 0, -12, 30],
                [-12, -15, -3, -3, -3, -3, -15, -12],
                [0, -3, 0, -1, -1, 0, -3, 0],
                [-1, -3, -1, -1, -1, -1, -3, -1],
                [-1, -3, -1, -1, -1, -1, -3, -1],
                [0, -3, 0, -1, -1, 0, -3, 0],
                [-12, -15, -3, -3, -3, -3, -15, -12],
                [30, -12, 0, -1, -1, 0, -12, 30],
            ]
            self.eval_field = np.array(self.eval_field).flatten()
            assert self.eval_field.shape == (64,)

        elif _env.W == 6:
            self.max_depth = 3
            self.eval_field = [
                [30, -12, 0, 0, -12, 30],
                [-12, -15, -3, -3, -15, -12],
                [0, -3, 0, 0, -3, 0],
                [0, -3, 0, 0, -3, 0],
                [-12, -15, -3, -3, -15, -12],
                [30, -12, 0, 0, -12, 30],
            ]
            self.eval_field = np.array(self.eval_field).flatten()
            assert self.eval_field.shape == (36,)

        elif _env.W == 4:
            self.max_depth = 6

        return {}

    def call_policy(self, env: EnvRun) -> Tuple[int, dict]:
        self._count = 0
        self.t0 = time.time()
        _env = cast(Othello, env.unwrapped)
        scores = self._negamax(cast(Othello, _env.copy()))
        self._render_scores = scores
        self._render_count = self._count
        self._render_time = time.time() - self.t0

        scores = np.array(scores)
        action = int(np.random.choice(np.where(scores == scores.max())[0]))
        return action, {}

    def _negamax(self, env: Othello, depth: int = 0):
        key = str(env.field)
        if key in Cpu.cache:
            return Cpu.cache[key]

        self._count += 1

        env_dat = env.backup()
        player_index = env.next_player
        valid_actions = env.get_valid_actions(player_index)

        scores = [-999.0 for _ in range(env.action_space.n)]
        for a in valid_actions:
            env.restore(env_dat)

            # env stepを実施
            _, (r1, r2), done, _ = env.step(a)
            if done:
                # 終了状態なら報酬をスコアにする
                if player_index == 0:
                    scores[a] = r1 * 500
                else:
                    scores[a] = r2 * 500
            elif depth > self.max_depth:
                # 評価値を返す
                if self.eval_field is None:
                    scores[a] = 0
                else:
                    scores[a] = np.sum(np.asarray(self.eval_field) * np.asarray(env.field))
                if player_index != 0:
                    scores[a] = -scores[a]
            else:
                is_enemy = player_index != env.next_player
                n_scores = self._negamax(env, depth + 1)
                if is_enemy:
                    scores[a] = -np.max(n_scores)
                else:
                    scores[a] = np.max(n_scores)

        Cpu.cache[key] = scores
        return scores

    def render_terminal(self, worker, **kwargs) -> None:
        _env = cast(Othello, worker.env.unwrapped)
        valid_actions = worker.env.get_valid_actions(_env.next_player)

        print(f"- MinMax count: {self._render_count}, {self._render_time:.3f}s -")
        for y in range(_env.H):
            s = "|"
            for x in range(_env.W):
                a = x + y * _env.W
                if a in valid_actions:
                    s += "{:6.1f}|".format(self._render_scores[a])
                else:
                    s += " " * 6 + "|"
            print(s)
        print()
