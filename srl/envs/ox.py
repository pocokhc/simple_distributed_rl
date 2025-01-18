import logging
import time
from abc import abstractmethod
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
    id="OX",
    entry_point=__name__ + ":OX",
    kwargs={},
    check_duplicate=False,
)

register(
    id="OX-layer",
    entry_point=__name__ + ":OXLayer",
    kwargs={},
    check_duplicate=False,
)


class _OXBase(EnvBase[DiscreteSpace, int, TObsSpace, TObsType], Generic[TObsSpace, TObsType]):
    _scores_cache = {}

    def __init__(self):
        super().__init__()
        self.screen = None

        self.W = 3
        self.H = 3

    @property
    def action_space(self) -> DiscreteSpace:
        return DiscreteSpace(self.W * self.H)

    @property
    def player_num(self) -> int:
        return 2

    @property
    def max_episode_steps(self) -> int:
        return 10

    @property
    def reward_range(self) -> Tuple[float, float]:
        return -1, 1

    @property
    def reward_baseline(self):
        # [0.987, 0.813] ぐらい
        return [
            {"episode": 200, "players": [None, "random"], "baseline": [0.8, None]},
            {"episode": 200, "players": ["random", None], "baseline": [None, 0.65]},
        ]

    def reset(self, *, seed: Optional[int] = None, **kwargs) -> Any:
        self.field = [0 for _ in range(self.W * self.H)]
        self.next_player = 0
        return self._create_state()

    def backup(self) -> Any:
        return [self.field[:], self.next_player]

    def restore(self, data: Any) -> None:
        self.field = data[0][:]
        self.next_player = data[1]

    @abstractmethod
    def _create_state(self):
        raise NotImplementedError()

    def step(self, action) -> Tuple[Any, List[float], bool, bool]:
        reward1, reward2, done = self._step(action)

        if not done:
            if self.next_player == 0:
                self.next_player = 1
            else:
                self.next_player = 0

        return self._create_state(), [reward1, reward2], done, False

    def _step(self, action):
        # error action
        if self.field[action] != 0:
            if self.next_player == 0:
                return -1, 0.0, True
            else:
                return 0.0, -1.0, True

        # update
        if self.next_player == 0:
            self.field[action] = 1
        else:
            self.field[action] = -1

        reward1, reward2, done = self._check(self.field)

        return reward1, reward2, done

    def _check(self, field):
        for pos in [
            # 横
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            # 縦
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8],
            # 斜め
            [0, 4, 8],
            [2, 4, 6],
        ]:
            # 3つとも同じか
            if field[pos[0]] == field[pos[1]] and field[pos[1]] == field[pos[2]]:
                if field[pos[0]] == 1:
                    # player1 win
                    return 1.0, -1.0, True
                if field[pos[0]] == -1:
                    # player2 win
                    return -1.0, 1.0, True

        # 置く場所がなければdraw
        if sum([1 if v == 0 else 0 for v in field]) == 0:
            return 0.0, 0.0, True

        return 0.0, 0.0, False

    def get_invalid_actions(self, player_index: int = -1) -> List[int]:
        actions = []
        for a in range(self.H * self.W):
            if self.field[a] != 0:
                # x = a % self.W
                # y = a // self.W
                actions.append(a)
        return actions

    def render_terminal(self, **kwargs) -> None:
        print("-" * 10)
        for y in range(self.H):
            s = "|"
            for x in range(self.W):
                a = x + y * self.W
                if self.field[a] == 1:
                    s += " o|"
                elif self.field[a] == -1:
                    s += " x|"
                else:
                    s += "{:2d}|".format(a)
            print(s)
            print("-" * 10)
        if self.next_player == 0:
            print("next player: O")
        else:
            print("next player: X")

    def render_rgb_array(self, **kwargs) -> Optional[np.ndarray]:
        from srl.utils import pygame_wrapper as pw

        WIDTH = 200
        HEIGHT = 200
        if self.screen is None:
            self.screen = pw.create_surface(WIDTH, HEIGHT)

        w_margin = 10
        h_margin = 10
        cell_w = int((WIDTH - w_margin * 2) / self.W)
        cell_h = int((HEIGHT - h_margin * 2) / self.H)

        pw.draw_fill(self.screen)

        # --- line
        width = 5
        for i in range(4):
            x = w_margin + i * cell_w
            pw.draw_line(self.screen, x, h_margin, x, HEIGHT - h_margin, width=width)
        for i in range(4):
            y = h_margin + i * cell_h
            pw.draw_line(self.screen, w_margin, y, WIDTH - w_margin, y, width=width)

        # --- field
        for y in range(self.H):
            for x in range(self.W):
                center_x = int(w_margin + x * cell_w + cell_w / 2)
                center_y = int(h_margin + y * cell_h + cell_h / 2)

                a = x + y * self.W
                if self.field[a] == 1:  # o
                    pw.draw_circle(self.screen, center_x, center_y, int(cell_w * 0.3), line_color=(200, 0, 0), width=5)
                elif self.field[a] == -1:  # x
                    color = (0, 0, 200)
                    width = 5
                    diff = int(cell_w * 0.3)
                    pw.draw_line(
                        self.screen,
                        center_x - diff,
                        center_y - diff,
                        center_x + diff,
                        center_y + diff,
                        color=color,
                        width=width,
                    )
                    pw.draw_line(
                        self.screen,
                        center_x - diff,
                        center_y + diff,
                        center_x + diff,
                        center_y - diff,
                        color=color,
                        width=width,
                    )

        return pw.get_rgb_array(self.screen)

    @property
    def render_interval(self) -> float:
        return 1000 / 1

    def make_worker(self, name: str, **kwargs):
        if name == "cpu":
            return Cpu(**kwargs)
        return None

    # --------------------------------------------

    def calc_scores(self) -> List[float]:
        self._scores_count = 0
        t0 = time.time()
        scores = self._negamax(cast(_OXBase, self.copy()))
        self._scores_time = time.time() - t0
        return scores

    def _negamax(self, env: "_OXBase") -> List[float]:
        key = str(env.field)
        if key in _OXBase._scores_cache:
            return _OXBase._scores_cache[key]

        self._scores_count += 1
        env_dat = env.backup()

        scores = [-9.0 for _ in range(env.action_space.n)]
        for a in self.get_valid_actions():
            a = cast(int, a)
            env.restore(env_dat)

            _, rewards, done, _ = env.step(a)
            if done:
                if env.next_player == 0:
                    scores[a] = rewards[0]
                else:
                    scores[a] = rewards[1]
            else:
                # 次の状態へ
                n_scores = self._negamax(env)
                scores[a] = -np.max(n_scores)

        _OXBase._scores_cache[key] = scores
        return scores


class OX(_OXBase[ArrayDiscreteSpace, List[int]]):
    @property
    def observation_space(self):
        return ArrayDiscreteSpace(self.H * self.W, low=-1, high=1)

    def _create_state(self):
        return self.field


class OXLayer(_OXBase[BoxSpace, np.ndarray]):
    @property
    def observation_space(self):
        return BoxSpace(low=0, high=1, shape=(3, 3, 2), dtype=np.float32, stype=SpaceTypes.IMAGE)

    def _create_state(self):
        # Layer0: player1 field (0 or 1)
        # Layer1: player2 field (0 or 1)
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
    def call_policy(self, env: EnvRun) -> Tuple[int, dict]:
        _env = cast(OX, env.unwrapped)
        scores = _env.calc_scores()
        self._render_scores = scores
        self._render_count = _env._scores_count
        self._render_time = _env._scores_time

        action = int(np.random.choice(np.where(scores == np.max(scores))[0]))
        return action, {}

    def render_render(self, worker, **kwargs) -> None:
        _env = cast(OX, worker.env.unwrapped)

        print(f"- alphabeta({self._render_count}, {self._render_time:.3f}s) -")
        print("-" * 10)
        for y in range(_env.H):
            s = "|"
            for x in range(_env.W):
                a = x + y * _env.W
                s += "{:2.0f}|".format(self._render_scores[a])
            print(s)
            print("-" * 10)
