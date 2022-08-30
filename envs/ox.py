import logging
import random
import time
from typing import Any, List, Optional, Tuple, cast

import numpy as np
from srl.base.define import EnvAction, EnvObservationType, RLObservationType
from srl.base.env.base import EnvRun, SpaceBase
from srl.base.env.genre import TurnBase2Player
from srl.base.env.registration import register
from srl.base.env.spaces import BoxSpace, DiscreteSpace
from srl.base.rl.base import RuleBaseWorker, WorkerRun
from srl.base.rl.processor import Processor
from srl.utils.viewer import Viewer

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

        self._player_index = 0
        self.viewer = None

    @property
    def action_space(self) -> SpaceBase:
        return DiscreteSpace(self.W * self.H)

    @property
    def observation_space(self) -> SpaceBase:
        return BoxSpace(
            low=-1,
            high=1,
            shape=(self.H * self.W,),
        )

    @property
    def observation_type(self) -> EnvObservationType:
        return EnvObservationType.DISCRETE

    @property
    def max_episode_steps(self) -> int:
        return 10

    @property
    def player_index(self) -> int:
        return self._player_index

    def call_reset(self) -> np.ndarray:
        self.field = [0 for _ in range(self.W * self.H)]
        self._player_index = 0
        return self._encode_state()

    # 観測用の状態を返す
    def _encode_state(self):
        # (turn,) + field
        return np.array(self.field)

    def backup(self) -> Any:
        return [self.field[:], self._player_index]

    def restore(self, data: Any) -> None:
        self.field = data[0][:]
        self._player_index = data[1]

    def call_step(self, action: int) -> Tuple[np.ndarray, float, float, bool, dict]:

        reward1, reward2, done = self._step(action)

        if not done:
            if self._player_index == 0:
                self._player_index = 1
            else:
                self._player_index = 0

        return (
            self._encode_state(),
            reward1,
            reward2,
            done,
            {},
        )

    def _step(self, action):

        # error action
        if self.field[action] != 0:
            if self.player_index == 0:
                return -1, 0, True
            else:
                return 0, -1, True

        # update
        if self.player_index == 0:
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

    def get_invalid_actions(self, player_index: int = 0) -> List[int]:
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
        if self.player_index == 0:
            print("next player: O")
        else:
            print("next player: X")

    def render_rgb_array(self, **kwargs) -> np.ndarray:

        WIDTH = 200
        HEIGHT = 200
        if self.viewer is None:
            self.viewer = Viewer(WIDTH, HEIGHT)

        w_margin = 10
        h_margin = 10
        cell_w = int((WIDTH - w_margin * 2) / self.W)
        cell_h = int((HEIGHT - h_margin * 2) / self.H)

        self.viewer.draw_fill()

        # --- line
        width = 5
        for i in range(4):
            x = w_margin + i * cell_w
            self.viewer.draw_line(x, h_margin, x, HEIGHT - h_margin, width=width)
        for i in range(4):
            y = h_margin + i * cell_h
            self.viewer.draw_line(w_margin, y, WIDTH - w_margin, y, width=width)

        # --- field
        for y in range(self.H):
            for x in range(self.W):
                center_x = int(w_margin + x * cell_w + cell_w / 2)
                center_y = int(h_margin + y * cell_h + cell_h / 2)

                a = x + y * self.W
                if self.field[a] == 1:  # o
                    self.viewer.draw_circle(center_x, center_y, int(cell_w * 0.3), line_color=(200, 0, 0), width=5)
                elif self.field[a] == -1:  # x
                    color = (0, 0, 200)
                    width = 5
                    diff = int(cell_w * 0.3)
                    self.viewer.draw_line(
                        center_x - diff, center_y - diff, center_x + diff, center_y + diff, color=color, width=width
                    )
                    self.viewer.draw_line(
                        center_x - diff, center_y + diff, center_x + diff, center_y - diff, color=color, width=width
                    )

        return self.viewer.get_rgb_array()

    def make_worker(self, name: str) -> Optional[RuleBaseWorker]:
        if name == "cpu":
            return Cpu()
        return None

    # --------------------------------------------

    def calc_scores(self) -> List[float]:
        self._scores_count = 0
        t0 = time.time()
        scores = self._negamax(self.copy())
        self._scores_time = time.time() - t0
        return scores

    def _negamax(self, env: "OX") -> List[float]:
        key = str(env.field)
        if key in OX._scores_cache:
            return OX._scores_cache[key]

        self._scores_count += 1
        env_dat = env.backup()

        scores = [-9.0 for _ in range(env.action_space.n)]
        for a in self.get_valid_actions(self.player_index):
            a = cast(int, a)
            env.restore(env_dat)

            _, r1, r2, done, _ = env.call_step(a)
            if done:
                if env.player_index == 0:
                    scores[a] = r1
                else:
                    scores[a] = r2
            else:
                # 次の状態へ
                n_scores = self._negamax(env)
                scores[a] = -np.max(n_scores)

        OX._scores_cache[key] = scores
        return scores


class Cpu(RuleBaseWorker):
    cache = {}

    def call_on_reset(self, env: EnvRun, worker: WorkerRun) -> None:
        pass  #

    def call_policy(self, _env: EnvRun, worker: WorkerRun) -> EnvAction:
        env = cast(OX, _env.get_original_env())
        scores = env.calc_scores()
        self._render_scores = scores
        self._render_count = env._scores_count
        self._render_time = env._scores_time

        action = int(random.choice(np.where(scores == np.max(scores))[0]))
        return action

    def render_render(self, _env: EnvRun, worker: WorkerRun, **kwargs) -> None:
        env = cast(OX, _env.get_original_env())

        print(f"- alphabeta({self._render_count}, {self._render_time:.3f}s) -")
        print("-" * 10)
        for y in range(env.H):
            s = "|"
            for x in range(env.W):
                a = x + y * env.W
                s += "{:2.0f}|".format(self._render_scores[a])
            print(s)
            print("-" * 10)


class LayerProcessor(Processor):
    def change_observation_info(
        self,
        env_observation_space: SpaceBase,
        env_observation_type: EnvObservationType,
        rl_observation_type: RLObservationType,
        env: OX,
    ) -> Tuple[SpaceBase, EnvObservationType]:
        observation_space = BoxSpace(
            low=0,
            high=1,
            shape=(2, 3, 3),
        )
        return observation_space, EnvObservationType.SHAPE3

    def process_observation(self, observation: np.ndarray, _env: EnvRun) -> np.ndarray:
        env = cast(OX, _env.get_original_env())

        # Layer0: player1 field (0 or 1)
        # Layer1: player2 field (0 or 1)
        if env.player_index == 0:
            my_field = 1
            enemy_field = -1
        else:
            my_field = -1
            enemy_field = 1
        _field = np.zeros((2, env.H, env.W))
        for y in range(env.H):
            for x in range(env.W):
                idx = x + y * env.W
                if observation[idx] == my_field:
                    _field[0][y][x] = 1
                elif observation[idx] == enemy_field:
                    _field[1][y][x] = 1
        return _field
