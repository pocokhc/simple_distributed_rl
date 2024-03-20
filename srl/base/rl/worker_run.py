import logging
from typing import Generic, List, Optional, Tuple, TypeVar, Union, cast

import numpy as np

from srl.base.define import (
    DoneTypes,
    EnvActionType,
    EnvObservationType,
    InfoType,
    ObservationModes,
    RenderModes,
    RLActionType,
    RLInvalidActionType,
    RLObservationType,
    SpaceTypes,
)
from srl.base.env.env_run import EnvRun
from srl.base.render import Render
from srl.base.rl.config import RLConfig
from srl.base.rl.parameter import RLParameter
from srl.base.rl.worker import RLWorker
from srl.base.spaces.space import SpaceBase

logger = logging.getLogger(__name__)

_TActSpace = TypeVar("_TActSpace")
_TObsSpace = TypeVar("_TObsSpace")


class WorkerRun(Generic[_TActSpace, _TObsSpace]):
    def __init__(
        self,
        worker: RLWorker[RLConfig, RLParameter, _TActSpace, _TObsSpace],
        env: EnvRun,
        distributed: bool = False,
        actor_id: int = 0,
    ):
        worker.config.setup(env, enable_log=False)
        worker._set_worker_run(self)

        self._worker = worker
        self._config: RLConfig[SpaceBase, SpaceBase] = worker.config
        self._env = env
        self._distributed = distributed
        self._actor_id = actor_id

        self._training: bool = False
        self._rendering: bool = False
        self._player_index: int = 0
        self._info: dict = {}
        self._prev_state: RLObservationType = []  # None
        self._state: RLObservationType = []  # None
        self._prev_action: RLActionType = 0
        self._reward: float = 0
        self._step_reward: float = 0
        self._prev_invalid_actions: List[RLInvalidActionType] = []
        self._invalid_actions: List[RLInvalidActionType] = []
        self._render = Render(worker)

        self._total_step: int = 0
        self._dummy_rl_states_one_step = [s.get_default() for s in self._config.observation_spaces_one_step]

    # ------------------------------------
    # episode functions
    # ------------------------------------
    @property
    def worker(self) -> RLWorker:
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
    def prev_state(self) -> RLObservationType:
        return self._prev_state

    @property
    def state(self):
        return self._state

    @property
    def prev_action(self) -> RLActionType:
        return self._prev_action

    @property
    def reward(self) -> float:
        return self._reward

    @property
    def done(self) -> bool:
        return self._env._done != DoneTypes.NONE

    @property
    def terminated(self) -> bool:
        return self._env._done == DoneTypes.TERMINATED

    @property
    def done_type(self) -> DoneTypes:
        return self._env._done

    @property
    def done_reason(self) -> str:
        return self._env._done_reason

    @property
    def prev_invalid_actions(self) -> List[RLInvalidActionType]:
        return self._prev_invalid_actions

    @property
    def invalid_actions(self) -> List[RLInvalidActionType]:
        return self._invalid_actions

    @property
    def total_step(self) -> int:
        return self._total_step

    def on_reset(
        self,
        player_index: int,
        training: bool,
        render_mode: Union[str, RenderModes] = "",
    ) -> None:
        self._player_index = player_index
        self._training = training
        self._is_reset = False

        if self._config.window_length > 1:
            self._recent_states: List[List[RLObservationType]] = [
                self._dummy_rl_states_one_step for _ in range(self._config.window_length)
            ]

        self._info = {}
        self._state = self.state_encode(
            self._dummy_rl_states_one_step,
            self._env,
            create_env_sate=False,
            enable_state_encode=False,
            append_recent_state=True,
        )
        self._prev_state = self._state
        self._prev_action = 0
        self._reward = 0
        self._step_reward = 0
        self._set_invalid_actions()

        [r.on_reset(self._env) for r in self._config.episode_processors]

        self._rendering = RenderModes.is_rendering(render_mode)
        self._render.reset(render_mode)

    def policy(self) -> EnvActionType:
        if not self._is_reset:
            # 1週目は reset -> policy
            self._set_invalid_actions()
            self._prev_state = self._state
            self._state = self.state_encode(
                self.env.state,
                self._env,
                create_env_sate=True,
                enable_state_encode=self._config.enable_state_encode,
                append_recent_state=True,
            )
            self._info = self._worker.on_reset(self)
            self._is_reset = True
        else:
            # 2週目以降は step -> policy
            self._on_step()

        # worker policy
        self._prev_action, info = self._worker.policy(self)
        if self._config.enable_assertion:
            assert self._config.action_space.check_val(self._prev_action)
        elif self._config.enable_sanitize:
            self._prev_action = self._config.action_space.sanitize(self._prev_action)
        env_action = self.action_decode(self._prev_action)
        self._info.update(info)

        if self._rendering:
            self._render.cache_reset()

        return env_action

    def on_step(self) -> None:
        # 初期化前はskip
        if not self._is_reset:
            return
        self._total_step += 1

        # 相手の番のrewardも加算
        self._step_reward += self._env.step_rewards[self.player_index]

        # 終了ならon_step実行
        if self.done:
            self._on_step()
            if self._rendering:
                self._render.cache_reset()

    def _on_step(self):
        # encode -> set invalid -> on_step -> reward=0
        self._set_invalid_actions()
        self._prev_state = self._state
        self._state = self.state_encode(
            self.env.state,
            self._env,
            create_env_sate=True,
            enable_state_encode=self._config.enable_state_encode,
            append_recent_state=True,
        )
        self._reward = self.reward_encode(self._step_reward, self._env)
        self._env._done = self.done_encode(self._env._done, self._env)
        self._info = self._worker.on_step(self)
        self._step_reward = 0

    def _set_invalid_actions(self):
        self._prev_invalid_actions = self._invalid_actions
        self._invalid_actions = [
            cast(RLInvalidActionType, self.action_encode(a)) for a in self._env.get_invalid_actions(self.player_index)
        ]

    def on_start(self):
        self._worker.on_start(self)

    def on_end(self):
        self._worker.on_end(self)

    # ------------------------------
    # encode/decode
    # ------------------------------
    def state_encode(
        self,
        env_state: EnvObservationType,
        env: EnvRun,
        create_env_sate: bool,
        enable_state_encode: bool,
        append_recent_state: bool,
    ) -> RLObservationType:

        # --- create env state
        if create_env_sate:
            env_states = []
            if self._config.observation_mode & ObservationModes.ENV:
                if self._config.is_env_obs_multi:
                    assert isinstance(env_state, list)
                    env_states.extend(env_state)
                else:
                    env_states.append(env_state)
            if self._config.observation_mode & ObservationModes.RENDER_IMAGE:
                env_states.append(self._env.render_rgb_array())
            if self._config.observation_mode & ObservationModes.RENDER_TERMINAL:
                env_states.append(self._env.render_ansi())
        else:
            env_states = cast(List[EnvObservationType], env_state)

        # --- encode
        if enable_state_encode:
            rl_states = []
            for i in range(len(env_states)):
                for p in self._config.observation_processors_list[i]:
                    env_states[i] = p.preprocess_observation(env_states[i], env)

                env_space = self._config.observation_spaces_of_env[i]
                rl_space = self._config.observation_spaces_one_step[i]
                if rl_space.stype == SpaceTypes.DISCRETE:
                    rl_states.append(env_space.encode_to_list_int(env_states[i]))
                elif rl_space.stype == SpaceTypes.CONTINUOUS:
                    rl_states.append(env_space.encode_to_np(env_states[i], np.float32))
                elif rl_space.stype == SpaceTypes.IMAGE:
                    rl_states.append(env_space.encode_to_np(env_states[i], np.float32))
                else:
                    # do nothing
                    rl_states.append(cast(RLObservationType, env_states[i]))
        else:
            rl_states = cast(List[RLObservationType], env_states)

        # --- create rl state
        if self._config.window_length > 1:
            if append_recent_state:
                self._recent_states.pop(0)
                self._recent_states.append(rl_states)
                _recent_state = self._recent_states
            else:
                _recent_state = self._recent_states[1:] + [rl_states]

            # 各配列毎に積み重ねる
            rl_states2 = []
            for i in range(len(rl_states)):
                space = self._config.observation_spaces_one_step[i]
                s = space.encode_stack([r[i] for r in _recent_state])
                rl_states2.append(s)
            rl_states = rl_states2

        if self._config.observation_space.stype == SpaceTypes.MULTI:
            return rl_states
        else:
            return rl_states[0]

    def action_encode(self, env_action: EnvActionType) -> RLActionType:
        if self._config.enable_action_decode:
            env_actions = []
            if self._config.action_space.stype == SpaceTypes.MULTI:
                assert isinstance(env_action, list)
                env_actions = env_action
            else:
                env_actions = [env_action]

            rl_actions = []
            for i in range(len(env_actions)):
                env_space = self._config.action_spaces_of_env[i]
                rl_space = self._config.action_spaces[i]
                if rl_space.stype == SpaceTypes.DISCRETE:
                    a = env_space.encode_to_int(env_actions[i])
                elif rl_space.stype == SpaceTypes.CONTINUOUS:
                    a = env_space.encode_to_list_float(env_actions[i])
                elif rl_space.stype == SpaceTypes.IMAGE:
                    a = env_space.encode_to_np(env_actions[i], np.uint8)
                else:
                    # do nothing
                    a = cast(RLActionType, env_action)
                rl_actions.append(a)

            if self._config.action_space.stype == SpaceTypes.MULTI:
                return rl_actions
            else:
                return rl_actions[0]

        return cast(RLActionType, env_action)

    def action_decode(self, rl_action: RLActionType) -> EnvActionType:
        if self._config.enable_action_decode:
            if self._config.action_space.stype == SpaceTypes.MULTI:
                assert isinstance(rl_action, list)
                rl_actions = rl_action
            else:
                rl_actions = [rl_action]

            env_actions = []
            for i in range(len(rl_actions)):
                env_space = self._config.action_spaces_of_env[i]
                rl_space = self._config.action_spaces[i]
                a = rl_actions[i]
                if rl_space.stype == SpaceTypes.DISCRETE:
                    a = env_space.decode_from_int(int(a))
                elif rl_space.stype == SpaceTypes.CONTINUOUS:
                    if isinstance(a, list):
                        a = [float(a2) for a2 in a]
                    else:
                        a = [float(a)]
                    a = env_space.decode_from_list_float(a)
                elif SpaceTypes.is_image(rl_space.stype):
                    a = env_space.decode_from_np(np.asarray(a))
                else:
                    # do nothing
                    a = cast(EnvActionType, a)
                env_actions.append(a)

            if len(self._config.action_spaces_of_env) > 1:
                # MULTI
                return env_actions
            else:
                return env_actions[0]

        return cast(EnvActionType, rl_action)

    def reward_encode(self, reward: float, env: EnvRun) -> float:
        if self._config.enable_reward_encode:
            for p in self._config.episode_processors:
                reward = p.preprocess_reward(reward, env)
        return reward

    def done_encode(self, done: DoneTypes, env: EnvRun) -> DoneTypes:
        if self._config.enable_done_encode:
            for p in self._config.episode_processors:
                done = p.preprocess_done(done, env)
        return done

    # ------------------------------------
    # invalid
    # ------------------------------------
    def get_invalid_actions(self, env: Optional[EnvRun] = None) -> List[RLInvalidActionType]:
        return self._invalid_actions

    def get_valid_actions(self, env: Optional[EnvRun] = None) -> List[RLInvalidActionType]:
        return self.config.action_space.get_valid_actions(self.get_invalid_actions(env))

    def add_invalid_actions(self, invalid_actions: List[RLInvalidActionType]) -> None:
        self._invalid_actions += invalid_actions
        self._invalid_actions = list(set(self._invalid_actions))

    # ------------------------------------
    # render functions
    # ------------------------------------
    def set_render_options(
        self,
        interval: float = -1,  # ms
        scale: float = 1.0,
        font_name: str = "",
        font_size: int = 18,
    ):
        self._render.set_render_options(interval, scale, font_name, font_size)

    def render(self, **kwargs):
        if not self._is_reset:
            return
        # workerはterminalのみ表示
        self._render.render(render_window=False, worker=self, **kwargs)

    def render_ansi(self, **kwargs) -> str:
        if not self._is_reset:
            return ""  # dummy
        return self._render.render_ansi(worker=self, **kwargs)

    def render_rgb_array(self, **kwargs) -> np.ndarray:
        if not self._is_reset:
            return np.zeros((4, 4, 3), dtype=np.uint8)  # dummy
        return self._render.render_rgb_array(worker=self, **kwargs)

    # ------------------------------------
    # utils
    # ------------------------------------
    def sample_action(self) -> RLActionType:
        action = self._config.action_space_of_env.sample(self.get_invalid_actions())
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
        next_state = self.state_encode(
            env.state,
            env,
            create_env_sate=True,
            enable_state_encode=self._config.enable_state_encode,
            append_recent_state=False,
        )
        rewards = [self.reward_encode(r, env) for r in env.step_rewards.tolist()]

        return next_state, rewards
