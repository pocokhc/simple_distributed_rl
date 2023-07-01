import logging
import time
from typing import Any, List, Optional, Tuple, cast

import numpy as np

from srl.base.define import EnvObservationTypes
from srl.base.env.env_run import EnvRun, SpaceBase
from srl.base.env.genre import TurnBase2Player
from srl.base.env.registration import register
from srl.base.rl.algorithms.env_worker import EnvWorker
from srl.base.rl.config import RLConfig
from srl.base.rl.processor import Processor
from srl.base.spaces import ArrayDiscreteSpace, BoxSpace, DiscreteSpace

logger = logging.getLogger(__name__)

register(
    id="OX",
    entry_point=__name__ + ":OX",
    kwargs={},
)


class OX(TurnBase2Player):
    _scores_cache = {}

    def __init__(self):
        self.W = 3
        self.H = 3

        self.screen = None
        self._next_player_index = 0

    @property
    def action_space(self) -> DiscreteSpace:
        return DiscreteSpace(self.W * self.H)

    @property
    def observation_space(self) -> ArrayDiscreteSpace:
        return ArrayDiscreteSpace(self.H * self.W, low=-1, high=1)

    @property
    def observation_type(self) -> EnvObservationTypes:
        return EnvObservationTypes.DISCRETE

    @property
    def max_episode_steps(self) -> int:
        return 10

    @property
    def reward_info(self) -> dict:
        return {
            "min": -1,
            "max": 1,
            "baseline": (0, 0),
            "type": int,
        }

    @property
    def next_player_index(self) -> int:
        return self._next_player_index

    def call_reset(self) -> Tuple[List[int], dict]:
        self.field = [0 for _ in range(self.W * self.H)]
        self._next_player_index = 0
        return self.field, {}

    def backup(self) -> Any:
        return [self.field[:], self._next_player_index]

    def restore(self, data: Any) -> None:
        self.field = data[0][:]
        self._next_player_index = data[1]

    def call_step(self, action: int) -> Tuple[List[int], float, float, bool, dict]:
        reward1, reward2, done = self._step(action)

        if not done:
            if self._next_player_index == 0:
                self._next_player_index = 1
            else:
                self._next_player_index = 0

        return self.field, reward1, reward2, done, {}

    def _step(self, action):
        # error action
        if self.field[action] != 0:
            if self._next_player_index == 0:
                return -1, 0, True
            else:
                return 0, -1, True

        # update
        if self._next_player_index == 0:
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
                    return 1, -1, True
                if field[pos[0]] == -1:
                    # player2 win
                    return -1, 1, True

        # 置く場所がなければdraw
        if sum([1 if v == 0 else 0 for v in field]) == 0:
            return 0, 0, True

        return 0, 0, False

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
        if self._next_player_index == 0:
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
        scores = self._negamax(cast(OX, self.copy()))
        self._scores_time = time.time() - t0
        return scores

    def _negamax(self, env: "OX") -> List[float]:
        key = str(env.field)
        if key in OX._scores_cache:
            return OX._scores_cache[key]

        self._scores_count += 1
        env_dat = env.backup()

        scores = [-9.0 for _ in range(env.action_space.n)]
        for a in self.get_valid_actions():
            a = cast(int, a)
            env.restore(env_dat)

            _, r1, r2, done, _ = env.call_step(a)
            if done:
                if env.next_player_index == 0:
                    scores[a] = r1
                else:
                    scores[a] = r2
            else:
                # 次の状態へ
                n_scores = self._negamax(env)
                scores[a] = -np.max(n_scores)

        OX._scores_cache[key] = scores
        return scores


class Cpu(EnvWorker):
    def call_policy(self, env: EnvRun) -> Tuple[int, dict]:
        _env = cast(OX, env.get_original_env())
        scores = _env.calc_scores()
        self._render_scores = scores
        self._render_count = _env._scores_count
        self._render_time = _env._scores_time

        action = int(np.random.choice(np.where(scores == np.max(scores))[0]))
        return action, {}

    def render_render(self, worker, **kwargs) -> None:
        _env = cast(OX, worker.env.get_original_env())

        print(f"- alphabeta({self._render_count}, {self._render_time:.3f}s) -")
        print("-" * 10)
        for y in range(_env.H):
            s = "|"
            for x in range(_env.W):
                a = x + y * _env.W
                s += "{:2.0f}|".format(self._render_scores[a])
            print(s)
            print("-" * 10)


class LayerProcessor(Processor):
    def preprocess_observation_space(
        self,
        env_observation_space: SpaceBase,
        env_observation_type: EnvObservationTypes,
        env: EnvRun,
        rl_config: RLConfig,
    ) -> Tuple[SpaceBase, EnvObservationTypes]:
        observation_space = BoxSpace(
            low=0,
            high=1,
            shape=(2, 3, 3),
        )
        return observation_space, EnvObservationTypes.SHAPE3

    def preprocess_observation(self, observation: np.ndarray, env: EnvRun) -> np.ndarray:
        _env = cast(OX, env.get_original_env())

        # Layer0: player1 field (0 or 1)
        # Layer1: player2 field (0 or 1)
        if _env.next_player_index == 0:
            my_field = 1
            enemy_field = -1
        else:
            my_field = -1
            enemy_field = 1
        _field = np.zeros((2, _env.H, _env.W))
        for y in range(_env.H):
            for x in range(_env.W):
                idx = x + y * _env.W
                if observation[idx] == my_field:
                    _field[0][y][x] = 1
                elif observation[idx] == enemy_field:
                    _field[1][y][x] = 1
        return _field
