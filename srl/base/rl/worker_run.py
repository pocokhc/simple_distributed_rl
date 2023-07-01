import logging
from typing import List, Optional, Tuple, Union, cast

import numpy as np

from srl.base.define import (
    EnvActionType,
    EnvObservationType,
    InfoType,
    InvalidActionsType,
    InvalidActionType,
    PlayRenderModes,
    RLActionType,
    RLObservationType,
    RLTypes,
)
from srl.base.env.env_run import EnvRun
from srl.base.render import Render
from srl.base.rl.config import RLConfig
from srl.base.rl.worker import WorkerBase

logger = logging.getLogger(__name__)


class WorkerRun:
    def __init__(
        self,
        worker: WorkerBase,
        env: EnvRun,
        distributed: bool = False,
        actor_id: int = 0,
    ):
        worker.config.reset(env)
        worker._set_worker_run(self)

        self._worker = worker
        self._config = worker.config
        self._env = env
        self._distributed = distributed
        self._actor_id = actor_id

        self._training = False
        self._rendering = False
        self._player_index = 0
        self._info = {}
        self._state: RLObservationType = self._config.create_dummy_state()
        self._reward = 0
        self._step_reward = 0
        self._done = False
        self._render = Render(worker)

        if self._config.window_length > 1:
            self._dummy_state = self._config.create_dummy_state(is_one=True)

    # ------------------------------------
    # episode functions
    # ------------------------------------
    @property
    def worker(self) -> WorkerBase:
        return self._worker

    @property
    def config(self) -> RLConfig:
        return self._config

    @property
    def env(self) -> EnvRun:
        return self._env

    @property
    def training(self) -> bool:
        return self._training

    @property
    def distributed(self) -> bool:
        return self._distributed

    @property
    def rendering(self) -> bool:
        return self._rendering

    @property
    def actor_id(self) -> int:
        return self._actor_id

    @property
    def player_index(self) -> int:
        return self._player_index

    @property
    def info(self) -> Optional[InfoType]:
        return self._info

    @property
    def state(self) -> RLObservationType:
        return self._state

    @property
    def reward(self) -> float:
        return self._reward

    @property
    def done(self) -> bool:
        return self._done

    def on_reset(
        self,
        player_index: int,
        training: bool,
        render_mode: Union[str, PlayRenderModes] = "",
    ) -> None:
        self._player_index = player_index
        self._training = training
        self._rendering = PlayRenderModes.is_rendering(render_mode)

        self._is_reset = False
        self._step_reward = 0

        self._info = {}
        self._state = self._config.create_dummy_state()
        self._reward = 0
        self._done = False

        if self._config.window_length > 1:
            self._recent_states: List[RLObservationType] = [
                self._dummy_state for _ in range(self._config.window_length)
            ]

        if self._rendering:
            self._render.cache_reset()
            self._render.reset(render_mode)

    def policy(self) -> EnvActionType:
        if not self._is_reset:
            # 1週目は reset -> policy
            self._state = self.state_encode(self.env.state, self._env, append_recent_state=True)
            self._info = self._worker.on_reset(self)
            self._is_reset = True
        else:
            # 2週目以降は step -> policy
            self._on_step()

        self._invalid_actions = self._env.get_invalid_actions()

        # worker policy
        action, info = self._worker.policy(self)
        action = self.action_decode(action)
        self._info.update(info)

        if self._rendering:
            self._render.cache_reset()

        return action

    def on_step(self) -> None:
        # 初期化前はskip
        if not self._is_reset:
            return

        # 相手の番のrewardも加算
        self._step_reward += self._env.step_rewards[self.player_index]

        # 終了ならon_step実行
        if self._env.done:
            self._on_step()
            if self._rendering:
                self._render.cache_reset()

    def _on_step(self):
        self._state = self.state_encode(self._env.state, self._env, append_recent_state=True)
        self._reward = self.reward_encode(self._step_reward, self._env)
        self._done = self._env.done
        self._info = self._worker.on_step(self)
        self._step_reward = 0

    # ------------------------------
    # encode/decode
    # ------------------------------
    def state_encode(self, state: EnvObservationType, env: EnvRun, append_recent_state: bool) -> RLObservationType:
        if self._config.enable_state_encode:
            for processor in self._config.run_processors:
                state = processor.preprocess_observation(state, env)

            if self._config.observation_type == RLTypes.DISCRETE:
                state = self._config.observation_space.encode_to_int_np(state)
            elif self._config.observation_type == RLTypes.CONTINUOUS:
                state = self._config.observation_space.encode_to_np(state)
            else:
                # not coming
                state = np.asarray(state, dtype=np.float32)
            if state.shape == ():
                state = state.reshape((1,))
        else:
            state = cast(RLObservationType, state)
        if self._config.window_length > 1:
            if append_recent_state:
                self._recent_states.pop(0)
                self._recent_states.append(state)
                state = np.asarray(self._recent_states)
            else:
                state = np.asarray(self._recent_states[:-1] + [state])
        return state

    def action_encode(self, action: EnvActionType) -> RLActionType:
        if self._config.enable_action_decode:
            if self._config.action_type == RLTypes.DISCRETE:
                action = self._config.action_space.encode_to_int(action)
            elif self._config.action_type == RLTypes.CONTINUOUS:
                action = self._config.action_space.encode_to_list_float(action)
            else:
                # do nothing
                action = cast(RLActionType, action)
        else:
            action = cast(RLActionType, action)
        return action

    def action_decode(self, action: RLActionType) -> EnvActionType:
        if self._config.enable_action_decode:
            if self._config.action_type == RLTypes.DISCRETE:
                assert not isinstance(action, list)
                env_action = self._config.action_space.decode_from_int(int(action))
            elif self._config.action_type == RLTypes.CONTINUOUS:
                if isinstance(action, list):
                    action = [float(a) for a in action]
                else:
                    action = [float(action)]
                env_action = self._config.action_space.decode_from_list_float(action)
            else:
                env_action: EnvActionType = action  # not coming
        else:
            env_action: EnvActionType = action
        return env_action

    def reward_encode(self, reward: float, env: EnvRun) -> float:
        if self._config.enable_reward_encode:
            for processor in self._config.run_processors:
                reward = processor.preprocess_reward(reward, env)
        return reward

    # ------------------------------------
    # invalid
    # ------------------------------------
    def get_invalid_actions(self, env: Optional[EnvRun] = None) -> InvalidActionsType:
        if self._config.action_type == RLTypes.DISCRETE:
            if env is None:
                env = self._env
            if self._config.enable_action_decode:
                return [
                    cast(InvalidActionType, self.action_encode(a)) for a in env.get_invalid_actions(self.player_index)
                ]
            else:
                return env.get_invalid_actions(self.player_index)
        else:
            return []

    # def get_valid_actions(self, env=None) -> InvalidActionsType:
    #    raise NotImplementedError()  # TODO: bugがあったのでいったん保留

    # ------------------------------------
    # render functions
    # ------------------------------------
    def set_render_options(
        self,
        interval: float = -1,  # ms
        scale: float = 1.0,
        font_name: str = "",
        font_size: int = 12,
    ) -> float:
        self._render.interval = interval
        self._render.scale = scale
        self._render.font_name = font_name
        self._render.font_size = font_size
        return interval

    def render(self, **kwargs) -> Union[None, str, np.ndarray]:
        # 初期化前はskip
        if not self._is_reset:
            return self._render.get_dummy()

        return self._render.render(env=self.env, worker=self, **kwargs)

    def render_terminal(self, return_text: bool = False, **kwargs):
        # 初期化前はskip
        if not self._is_reset:
            if return_text:
                return ""
            return

        return self._render.render_terminal(return_text, env=self.env, worker=self, **kwargs)

    def render_rgb_array(self, **kwargs) -> np.ndarray:
        # 初期化前はskip
        if not self._is_reset:
            return np.zeros((4, 4, 3), dtype=np.uint8)  # dummy image

        return self._render.render_rgb_array(env=self.env, worker=self, **kwargs)

    def render_window(self, **kwargs) -> np.ndarray:
        # 初期化前はskip
        if not self._is_reset:
            return np.zeros((4, 4, 3), dtype=np.uint8)  # dummy image

        return self._render.render_window(env=self.env, worker=self, **kwargs)

    # ------------------------------------
    # utils
    # ------------------------------------
    def sample_action(self) -> RLActionType:
        action = self._config.action_space.sample(self.get_invalid_actions())
        return self.action_encode(action)

    def sample_action_for_env(self) -> EnvActionType:
        return self._env.sample_action()

    def env_step(self, env: EnvRun, action: RLActionType, **step_kwargs) -> Tuple[RLObservationType, List[float]]:
        """RLActionを入力として、envを1step進める。戻り値はRL側の状態。
        Worker自身の内部状態は変更しない
        """

        # 内部状態が変わるwindow_lengthは未対応
        assert self._config.window_length == 1, "window_length is not supported."

        env_action = self.action_decode(action)
        env.step(env_action, **step_kwargs)
        next_state = self.state_encode(env.state, env, append_recent_state=False)
        rewards = [self.reward_encode(r, env) for r in env.step_rewards.tolist()]

        return next_state, rewards
