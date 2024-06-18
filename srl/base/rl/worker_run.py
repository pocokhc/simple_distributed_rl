import logging
from typing import Any, Generic, List, Optional, Tuple, cast

import numpy as np

from srl.base.context import RunContext
from srl.base.define import (
    DoneTypes,
    EnvActionType,
    EnvObservationType,
    InfoType,
    ObservationModes,
    SpaceTypes,
    TActSpace,
    TActType,
    TObsSpace,
    TObsType,
)
from srl.base.env.env_run import EnvRun
from srl.base.exception import SRLError
from srl.base.render import Render
from srl.base.rl.config import RLConfig
from srl.base.rl.parameter import RLParameter
from srl.base.rl.worker import RLWorkerGeneric
from srl.base.spaces.multi import MultiSpace
from srl.base.spaces.space import SpaceBase

logger = logging.getLogger(__name__)


class WorkerRun(Generic[TActSpace, TActType, TObsSpace, TObsType]):
    def __init__(
        self,
        worker: RLWorkerGeneric[RLConfig, RLParameter, TActSpace, TActType, TObsSpace, TObsType],
        env: EnvRun,
    ):
        worker.config.setup(env, enable_log=False)
        worker._set_worker_run(self)

        self._worker = worker
        self._config: RLConfig[SpaceBase, SpaceBase] = worker.config
        self._env = env
        self._context = RunContext()
        self._render = Render(worker)
        self._has_start = False

        self._player_index: int = 0
        self._info: dict = {}
        self._prev_state: TObsType = cast(TObsType, [])  # None
        self._state: TObsType = cast(TObsType, [])  # None
        self._prev_action: TActType = cast(TActType, 0)
        self._action: TActType = cast(TActType, 0)
        self._reward: float = 0
        self._step_reward: float = 0
        self._done: DoneTypes = DoneTypes.RESET
        self._done_reason: str = ""
        self._prev_invalid_actions: List[TActType] = []
        self._invalid_actions: List[TActType] = []
        self._total_step: int = 0
        self._dummy_rl_states_one_step = self._config.observation_space_one_step.get_default()

        self._processors_on_reset: Any = [c for c in self._config.episode_processors if hasattr(c, "remap_on_reset")]
        self._processors_reward: Any = [c for c in self._config.episode_processors if hasattr(c, "remap_reward")]

    # ------------------------------------
    # episode functions
    # ------------------------------------
    @property
    def worker(self):
        return self._worker

    @property
    def config(self) -> RLConfig:
        return self._config

    @property
    def env(self) -> EnvRun:
        return self._env

    @property
    def context(self) -> RunContext:
        return self._context

    @property
    def training(self) -> bool:
        return self._context.training

    @property
    def distributed(self) -> bool:
        return self._context.distributed

    @property
    def rendering(self) -> bool:
        return self._context.rendering

    @property
    def actor_id(self) -> int:
        return self._context.actor_id

    @property
    def player_index(self) -> int:
        return self._player_index

    @property
    def info(self) -> InfoType:
        return self._info

    @property
    def prev_state(self) -> TObsType:
        return self._prev_state

    @property
    def state(self) -> TObsType:
        return self._state

    @property
    def prev_action(self) -> TActType:
        return self._prev_action

    @property
    def action(self) -> TActType:
        return self._action

    @property
    def reward(self) -> float:
        return self._reward

    @property
    def done(self) -> bool:
        return self._done != DoneTypes.NONE

    @property
    def terminated(self) -> bool:
        return self._done == DoneTypes.TERMINATED

    @property
    def done_type(self) -> DoneTypes:
        return self._done

    @property
    def done_reason(self) -> str:
        return self._done_reason

    @property
    def prev_invalid_actions(self) -> List[TActType]:
        return self._prev_invalid_actions

    @property
    def invalid_actions(self) -> List[TActType]:
        return self._invalid_actions

    @property
    def total_step(self) -> int:
        return self._total_step

    def on_start(self, context: RunContext):
        self._context = context
        self._render.set_render_mode(context.render_mode)

        # --- worker
        self._worker.on_start(self, context)
        self._has_start = True

    def on_reset(self, player_index: int) -> None:
        if not self._has_start:
            raise SRLError("Cannot call worker.on_reset() before calling worker.on_start(context)")

        self._player_index = player_index
        self._is_reset = False

        if self._config.window_length > 1:
            self._recent_states: List[TObsType] = [
                self._dummy_rl_states_one_step for _ in range(self._config.window_length)
            ]

        self._info = {}
        self._state = self.state_encode(
            self._dummy_rl_states_one_step,
            self._env,
            create_env_sate=False,
            enable_state_encode=False,
            append_recent_state=True,
            is_dummy=True,
        )
        self._prev_state = self._state
        self._action = cast(TActType, 0)
        self._prev_action = cast(TActType, 0)
        self._reward = 0
        self._step_reward = 0
        self._done = DoneTypes.NONE
        self._done_reason = ""
        self._set_invalid_actions()

        [r.remap_on_reset(self, self._env) for r in self._processors_on_reset]

    def policy(self) -> EnvActionType:
        if not self._is_reset:
            # 1週目は reset -> policy
            self._set_invalid_actions()
            self._prev_state = self._state
            self._state = self.state_encode(
                self._env.state,
                self._env,
                create_env_sate=True,
                enable_state_encode=self._config.enable_state_encode,
                append_recent_state=True,
                is_dummy=False,
            )
            self._info = self._worker.on_reset(self)
            if self._info is None:
                self._info = {}
            self._is_reset = True
        else:
            # 2週目以降は step -> policy
            self._on_step()

        # worker policy
        action, info = self._worker.policy(self)
        self._prev_action = self._action
        self._action = cast(TActType, action)
        if self._config.enable_assertion:
            assert self._config.action_space.check_val(self._action)
        elif self._config.enable_sanitize:
            self._action = self._config.action_space.sanitize(self._action)
        env_action = self.action_decode(self._action)
        self._info.update(info)

        if self._context.rendering:
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
        if self._env._done != DoneTypes.NONE:
            self._on_step()
            if self._context.rendering:
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
            is_dummy=False,
        )
        self._reward = self.reward_encode(self._step_reward, self._env)
        self._done = self._env._done
        self._done_reason = self._env._done_reason
        self._info = self._worker.on_step(self)
        if self._info is None:
            self._info = {}
        self._step_reward = 0

    def _set_invalid_actions(self):
        self._prev_invalid_actions = self._invalid_actions
        self._invalid_actions = [self.action_encode(a) for a in self._env.get_invalid_actions(self.player_index)]

    def on_end(self):
        self._worker.on_end(self)

    # ------------------------------
    # encode/decode
    # ------------------------------
    def _state_encode_sub(self, rl_stype: SpaceTypes, env_space: SpaceBase, env_state: EnvObservationType) -> TObsType:
        if rl_stype == SpaceTypes.DISCRETE:
            rl_state = env_space.encode_to_list_int(env_state)
        elif rl_stype == SpaceTypes.CONTINUOUS:
            rl_state = env_space.encode_to_np(env_state, self.config.dtype)
        elif SpaceTypes.is_image(rl_stype):
            rl_state = env_space.encode_to_np(env_state, self.config.dtype)
        elif rl_stype == SpaceTypes.MULTI:
            rl_state = env_state
        else:
            rl_state = env_state
        return cast(TObsType, rl_state)

    def state_encode(
        self,
        env_state: EnvObservationType,
        env: EnvRun,
        create_env_sate: bool,
        enable_state_encode: bool,
        append_recent_state: bool,
        is_dummy: bool,
    ) -> TObsType:

        if is_dummy:
            rl_state = env_state
        else:
            # --- create env state
            # listにして処理する
            if create_env_sate:
                env_states = []
                if self._config.observation_mode & ObservationModes.ENV:
                    if self._config._is_env_obs_multi:
                        assert isinstance(env_state, list)
                        env_states.extend(env_state)
                    else:
                        env_states.append(env_state)
                if self._config.observation_mode & ObservationModes.RENDER_IMAGE:
                    env_states.append(self._env.render_rgb_array())
                if self._config.observation_mode & ObservationModes.RENDER_TERMINAL:
                    env_states.append(self._env.render_ansi())
            else:
                env_states = cast(List[EnvObservationType], [env_state])

            # --- processor
            for i in range(len(env_states)):
                for p in self._config.observation_processors_list[i]:
                    p = cast(Any, p)
                    env_states[i] = p.remap_observation(env_states[i], self, env)

            # --- encode
            env_space = self._config.observation_space_of_env
            if not isinstance(env_space, MultiSpace):
                env_states = env_states[0]
            if enable_state_encode:
                rl_space = self._config.observation_space_one_step
                if isinstance(rl_space, MultiSpace):
                    if isinstance(env_space, MultiSpace):
                        # : assert len(rl_spaces) == len(env_spaces)
                        rl_state = []
                        for i in range(rl_space.space_size):
                            rl_state.append(
                                self._state_encode_sub(
                                    rl_space.spaces[i].stype,
                                    env_space.spaces[i],
                                    env_states[i],
                                )
                            )
                    else:
                        # : assert len(rl_spaces) == 1
                        rl_state = [
                            self._state_encode_sub(
                                rl_space.spaces[0].stype,
                                env_space,
                                env_states,
                            )
                        ]
                else:
                    rl_state = self._state_encode_sub(
                        rl_space.stype,
                        env_space,
                        env_states,
                    )
            else:
                rl_state = env_states
        rl_state = cast(TObsType, rl_state)

        # --- create rl state
        if self._config.window_length > 1:
            if append_recent_state:
                self._recent_states.pop(0)
                self._recent_states.append(rl_state)
                _recent_state = self._recent_states
            else:
                _recent_state = self._recent_states[1:] + [rl_state]

            # 各配列毎に積み重ねる
            rl_state = self._config.observation_space_one_step.encode_stack(_recent_state)

        return rl_state

    def _action_encode_sub(self, rl_stype: SpaceTypes, env_space: SpaceBase, env_act) -> TActType:
        if rl_stype == SpaceTypes.DISCRETE:
            rl_act = env_space.encode_to_int(env_act)
        elif rl_stype == SpaceTypes.CONTINUOUS:
            rl_act = env_space.encode_to_list_float(env_act)
        elif SpaceTypes.is_image(rl_stype):
            rl_act = env_space.encode_to_np(env_act, np.uint8)
        elif rl_stype == SpaceTypes.MULTI:
            rl_act = env_act
        else:
            rl_act = env_act
        return cast(TActType, rl_act)

    def action_encode(self, env_action: EnvActionType) -> TActType:
        if self._config.enable_action_decode:
            env_space = self._config.action_space_of_env
            rl_space = self._config.action_space
            if isinstance(rl_space, MultiSpace):
                if isinstance(env_space, MultiSpace):
                    env_action_multi = cast(list, env_action)

                    # : assert len(rl_spaces) == len(env_spaces)
                    rl_act = []
                    for i in range(rl_space.space_size):
                        rl_act.append(
                            self._action_encode_sub(
                                rl_space.spaces[i].stype,
                                env_space.spaces[i],
                                env_action_multi[i],
                            )
                        )
                else:
                    # : assert len(rl_spaces) == 1
                    rl_act = [
                        self._action_encode_sub(
                            rl_space.spaces[0].stype,
                            env_space,
                            env_action,
                        )
                    ]
            else:
                rl_act = self._action_encode_sub(
                    rl_space.stype,
                    env_space,
                    env_action,
                )
        else:
            rl_act = env_action
        return cast(TActType, rl_act)

    def _action_decode_sub(self, rl_stype: SpaceTypes, env_space: SpaceBase, rl_act) -> EnvActionType:
        if rl_stype == SpaceTypes.DISCRETE:
            return env_space.decode_from_int(int(rl_act))
        elif rl_stype == SpaceTypes.CONTINUOUS:
            if isinstance(rl_act, list):
                rl_act = [float(a2) for a2 in rl_act]
            else:
                rl_act = [float(rl_act)]
            return env_space.decode_from_list_float(rl_act)
        elif SpaceTypes.is_image(rl_stype):
            return env_space.decode_from_np(np.asarray(rl_act))
        elif rl_stype == SpaceTypes.MULTI:
            pass
        return cast(EnvActionType, rl_act)

    def action_decode(self, rl_action: TActType) -> EnvActionType:
        if self._config.enable_action_decode:
            env_space = self._config.action_space_of_env
            rl_space = self._config.action_space
            if isinstance(rl_space, MultiSpace):
                rl_action_multi = cast(list, rl_action)
                if isinstance(env_space, MultiSpace):
                    # : assert len(rl_spaces) == len(env_spaces)
                    env_act = []
                    for i in range(rl_space.space_size):
                        env_act.append(
                            self._action_decode_sub(
                                rl_space.spaces[i].stype,
                                env_space.spaces[i],
                                rl_action_multi[i],
                            )
                        )
                else:
                    # : assert len(rl_spaces) == 1
                    env_act = self._action_decode_sub(
                        rl_space.spaces[0].stype,
                        env_space,
                        rl_action_multi[0],
                    )

            else:
                env_act = self._action_decode_sub(
                    rl_space.stype,
                    env_space,
                    rl_action,
                )
        else:
            env_act = cast(EnvActionType, rl_action)
        return env_act

    def reward_encode(self, reward: float, env: EnvRun) -> float:
        if self._config.enable_reward_encode:
            for p in self._processors_reward:
                reward = p.remap_reward(reward, self, env)
        return reward

    # ------------------------------------
    # invalid
    # ------------------------------------
    def get_invalid_actions(self, env: Optional[EnvRun] = None) -> List[TActType]:
        return self._invalid_actions

    def get_valid_actions(self, env: Optional[EnvRun] = None) -> List[TActType]:
        return self.config.action_space.get_valid_actions(self.get_invalid_actions(env))

    def add_invalid_actions(self, invalid_actions: List[TActType]) -> None:
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

    def render_rl_image(self) -> Optional[np.ndarray]:
        if not SpaceTypes.is_image(self._config._rl_obs_space_one_step.stype):
            return None
        if self._config.window_length > 1:
            img = cast(np.ndarray, self._recent_states[-1]).copy()
        else:
            img = cast(np.ndarray, self._state).copy()
        if img.max() <= 1:
            img *= 255
        if self._config._rl_obs_space_one_step.stype == SpaceTypes.GRAY_2ch:
            img = img[..., np.newaxis]
            img = np.tile(img, (1, 1, 3))
        elif self._config._rl_obs_space_one_step.stype == SpaceTypes.GRAY_3ch:
            img = np.tile(img, (1, 1, 3))
        elif self._config._rl_obs_space_one_step.stype == SpaceTypes.COLOR:
            pass
        elif self._config._rl_obs_space_one_step.stype == SpaceTypes.IMAGE:
            if len(img.shape) == 3 and img.shape[-1] == 3:
                pass
            else:
                return None
        else:
            return None
        return img.astype(np.uint8)

    # ------------------------------------
    # utils
    # ------------------------------------
    def sample_action(self) -> TActType:
        action = self._config.action_space_of_env.sample(self.get_invalid_actions())
        return self.action_encode(action)

    def sample_action_for_env(self) -> EnvActionType:
        return self._env.sample_action()

    def env_step(self, env: EnvRun, action: TActType, **step_kwargs) -> Tuple[TObsType, List[float]]:
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
            is_dummy=False,
        )
        rewards = [self.reward_encode(r, env) for r in env.step_rewards.tolist()]

        return next_state, rewards
