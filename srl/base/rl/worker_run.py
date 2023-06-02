import logging
from typing import Optional, Union

import numpy as np

from srl.base.define import EnvActionType, InfoType, PlayRenderModes
from srl.base.env.env_run import EnvRun
from srl.base.render import Render
from srl.base.rl.worker import WorkerBase

logger = logging.getLogger(__name__)


class WorkerRun:
    def __init__(
        self,
        worker: WorkerBase,
        font_name: str = "",
        font_size: int = 12,
    ):
        self.worker = worker
        self._render = Render(worker, font_name, font_size)
        self._player_index = 0
        self._info = None
        self._step_reward = 0

    # ------------------------------------
    # episode functions
    # ------------------------------------
    @property
    def training(self) -> bool:
        return self.worker.training

    @property
    def distributed(self) -> bool:
        return self.worker.distributed

    @property
    def player_index(self) -> int:
        return self._player_index

    @property
    def info(self) -> Optional[InfoType]:
        return self._info

    @property
    def reward(self) -> float:
        return self._step_reward

    def on_reset(
        self,
        env: EnvRun,
        player_index: int = 0,
        render_mode: Union[str, PlayRenderModes] = "",
    ) -> None:
        self._player_index = player_index
        self._info = None
        self._is_reset = False
        self._step_reward = 0

        # --- render
        self._render.cache_reset()
        self._render.reset(render_mode, interval=-1)

    def policy(self, env: EnvRun) -> Optional[EnvActionType]:
        # 初期化していないなら初期化する
        if not self._is_reset:
            self._info = self.worker.on_reset(env, self)
            self._is_reset = True
        else:
            # 2週目以降はpolicyの実行前にstepを実行
            self._info = self.worker.on_step(env, self)
            self._step_reward = 0

        # worker policy
        env_action, info = self.worker.policy(env, self)
        self._info.update(info)
        self._render.cache_reset()

        return env_action

    def on_step(self, env: EnvRun) -> None:
        # 初期化前はskip
        if not self._is_reset:
            return

        # 相手の番のrewardも加算
        self._step_reward += env.step_rewards[self.player_index]

        # 終了ならon_step実行
        if env.done:
            self._info = self.worker.on_step(env, self)
            self._step_reward = 0
            self._render.cache_reset()

    # ------------------------------------
    # render functions
    # ------------------------------------
    def render(self, env: EnvRun, **kwargs) -> Union[None, str, np.ndarray]:
        # 初期化前はskip
        if not self._is_reset:
            return self._render.get_dummy()

        return self._render.render(env=env, worker=self, **kwargs)

    def render_terminal(self, env: EnvRun, return_text: bool = False, **kwargs):
        # 初期化前はskip
        if not self._is_reset:
            if return_text:
                return ""
            return

        return self._render.render_terminal(return_text, env=env, worker=self, **kwargs)

    def render_rgb_array(self, env: EnvRun, **kwargs) -> np.ndarray:
        # 初期化前はskip
        if not self._is_reset:
            return np.zeros((4, 4, 3), dtype=np.uint8)  # dummy image

        return self._render.render_rgb_array(env=env, worker=self, **kwargs)

    def render_window(self, env: EnvRun, **kwargs) -> np.ndarray:
        # 初期化前はskip
        if not self._is_reset:
            return np.zeros((4, 4, 3), dtype=np.uint8)  # dummy image

        return self._render.render_window(env=env, worker=self, **kwargs)
