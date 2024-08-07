import logging
from typing import Any, Generic, List, Optional, Tuple, cast

import numpy as np

from srl.base.context import RunContext
from srl.base.define import (
    DoneTypes,
    EnvActionType,
    EnvObservationType,
    ObservationModes,
    SpaceTypes,
    TActSpace,
    TActType,
    TObsSpace,
    TObsType,
)
from srl.base.env.env_run import EnvRun
from srl.base.exception import NotSupportedError, SRLError, UndefinedError
from srl.base.info import Info
from srl.base.render import Render
from srl.base.rl.config import RLConfig
from srl.base.rl.parameter import RLParameter
from srl.base.rl.worker import RLWorkerGeneric
from srl.base.spaces.space import SpaceBase
from srl.utils import render_functions as render_funcs

logger = logging.getLogger(__name__)


class WorkerRun(Generic[TActSpace, TActType, TObsSpace, TObsType]):
    def __init__(
        self,
        worker: RLWorkerGeneric[RLConfig, RLParameter, TActSpace, TActType, TObsSpace, TObsType],
        env: EnvRun,
    ):
        # - restore/backup用に状態は意識して管理
        # - env関係の値の保存はprocessorで変更がありそうなものはこちらが持つ
        #   そうじゃないものはenvの値をそのまま利用
        # - baseの中はクラスの中身まで把握しているものとして速度優先でprivateな変数のアクセスをする
        #   例えば self.env.info と self._env.env.info は同じだけど基本後半を使う方針

        worker.config.setup(env, enable_log=False)
        worker._set_worker_run(self)

        self._worker = worker
        self._config: RLConfig[SpaceBase, SpaceBase] = worker.config
        self._env = env
        self._render = Render(worker)
        self._has_start = False

        self._on_start_val(RunContext())
        self._on_reset_val(0)

        # --- processor
        self._processors = [c.copy() for c in self._config._episode_processors]
        self._processors_on_reset: Any = [c for c in self._processors if hasattr(c, "remap_on_reset")]
        self._processors_reward: Any = [c for c in self._processors if hasattr(c, "remap_reward")]

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
    def distributed(self) -> bool:
        return self._context.distributed

    @property
    def training(self) -> bool:
        return self._context.training

    @property
    def train_only(self) -> bool:
        return self._context.train_only

    @property
    def rollout(self) -> bool:
        return self._context.rollout

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
    def info(self) -> Info:
        return self._worker.info

    @property
    def prev_state(self) -> TObsType:
        return self._prev_state

    @property
    def state(self) -> TObsType:
        return self._state

    @property
    def state_one_step(self) -> TObsType:
        return self._recent_states[-1] if self._config.window_length > 1 else self._state

    @property
    def prev_render_img_state(self) -> np.ndarray:
        return self._prev_render_img_state

    @property
    def render_img_state(self) -> np.ndarray:
        return self._render_img_state

    @property
    def render_img_state_one_step(self) -> np.ndarray:
        return (
            self._recent_render_img_states[-1]
            if self._config.render_image_window_length > 1
            else self._render_img_state
        )

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
        return self._env._done != DoneTypes.NONE

    @property
    def terminated(self) -> bool:
        return self._env._done == DoneTypes.TERMINATED

    @property
    def done_type(self) -> DoneTypes:
        return self._env._done

    @property
    def done_reason(self) -> str:
        return self._env.env.done_reason

    @property
    def prev_invalid_actions(self) -> List[TActType]:
        return self._prev_invalid_actions

    @property
    def invalid_actions(self) -> List[TActType]:
        return self._invalid_actions

    @property
    def total_step(self) -> int:
        return self._total_step

    @property
    def episode_seed(self) -> Optional[int]:
        return self._episode_seed

    def on_start(self, context: RunContext):
        self._on_start_val(context)
        self._render.set_render_mode(context.render_mode, enable_window=False)
        self._worker.on_start(self, context)
        self._has_start = True

    def _on_start_val(self, context: RunContext):
        self._context = context
        self._total_step: int = 0

    def on_end(self):
        self._worker.on_end(self)
        self._has_start = False

    def on_reset(self, player_index: int, seed: Optional[int] = None) -> None:
        if not self._has_start:
            raise SRLError("Cannot call worker.on_reset() before calling worker.on_start(context)")

        if self._context.rendering:
            self._render.cache_reset()

        self._on_reset_val(player_index, seed)
        [r.remap_on_reset(self, self._env) for r in self._processors_on_reset]

    def _on_reset_val(self, player_index: int, seed: Optional[int] = None):
        self._player_index = player_index
        self._episode_seed = seed
        self._is_reset = False

        self._state = self._config.observation_space.get_default()
        self._prev_state = self._state
        if self._config.window_length > 1:
            self._recent_states: List[TObsType] = [
                self._config.observation_space_one_step.get_default() for _ in range(self._config.window_length)
            ]
        if self._config.use_render_image_state():
            self._render_img_state = self._config.obs_render_img_space.get_default()
            self._prev_render_img_state = self._render_img_state
            if self._config.render_image_window_length > 1:
                self._recent_render_img_states: List[np.ndarray] = [
                    self._config.obs_render_img_space_one_step.get_default()
                    for _ in range(self._config.render_image_window_length)
                ]
        self._action = cast(TActType, 0)
        self._prev_action = cast(TActType, 0)
        self._reward: float = 0
        self._step_reward: float = 0
        self._invalid_actions: List[TActType] = []
        self._set_invalid_actions()

    def policy(self) -> EnvActionType:
        # 1週目は reset -> policy
        # 2週目以降は step -> policy
        self._on_step(on_reset=(not self._is_reset))

        # worker policy
        action = self._worker.policy(self)
        self._prev_action = self._action
        self._action = cast(TActType, action)
        if self._config.enable_assertion:
            assert self._config.action_space.check_val(self._action)
        elif self._config.enable_sanitize:
            self._action = self._config.action_space.sanitize(self._action)
        env_action = self.action_decode(self._action)

        if self._context.rendering:
            self._render.cache_reset()

        return env_action

    def on_step(self) -> None:
        # 初期化前はskip
        if not self._is_reset:
            return
        self._total_step += 1

        # 相手の番のrewardも加算
        self._step_reward += self._env.rewards[self.player_index]

        # 終了ならon_step実行
        if self._env._done != DoneTypes.NONE:
            self._on_step()
            if self._context.rendering:
                self._render.cache_reset()

    def _on_step(self, on_reset: bool = False):
        # encode -> set invalid -> on_step -> reward=0
        self._set_invalid_actions()
        self._prev_state = self._state
        self._state = self.state_encode(
            self.env.state,
            self._env,
            enable_encode=self._config.enable_state_encode,
            append_recent_state=True,
        )
        if self._config.use_render_image_state():
            self._prev_render_img_state = self._render_img_state
            self._render_img_state = self.render_img_state_encode(
                self._env,
                enable_encode=self._config.enable_state_encode,
                append_recent_state=True,
            )

        if on_reset:
            self._worker.on_reset(self)
            self._is_reset = True
        else:
            self._reward = self.reward_encode(self._step_reward, self._env)
            self._step_reward = 0
            self._worker.on_step(self)

    def _set_invalid_actions(self):
        self._prev_invalid_actions = self._invalid_actions
        self._invalid_actions = [self.action_encode(a) for a in self._env.get_invalid_actions(self.player_index)]

    # ------------------------------
    # encode/decode
    # ------------------------------
    def state_encode(
        self,
        env_state: EnvObservationType,
        env: EnvRun,
        enable_encode: bool,
        append_recent_state: bool,
    ) -> TObsType:

        # --- observation_mode
        if self._config.observation_mode == ObservationModes.ENV:
            pass
        elif self._config.observation_mode == ObservationModes.RENDER_IMAGE:
            if self.config.obs_render_type == "rgb_array":
                env_state = cast(EnvObservationType, env.render_rgb_array())
            elif self.config.obs_render_type == "terminal":
                env_state = cast(EnvObservationType, env.render_terminal_text_to_image())
            else:
                raise NotSupportedError(self.config.obs_render_type)
        else:
            raise UndefinedError(self._config.observation_mode)

        if enable_encode:
            # --- processor
            for p in self._config._obs_processors:
                p = cast(Any, p)
                env_state = p.remap_observation(env_state, self, env)

            # --- encode
            rl_state: TObsType = self._config.observation_space_of_env.encode_to_space(
                env_state, self._config.observation_space_one_step
            )
        else:
            rl_state = cast(TObsType, env_state)

        # --- create rl state
        if self._config.window_length > 1:
            if append_recent_state:
                self._recent_states.pop(0)
                self._recent_states.append(rl_state)
                _recent_state = self._recent_states
            else:
                _recent_state = self._recent_states[1:] + [rl_state]
            rl_state = self._config.observation_space_one_step.encode_stack(_recent_state)

        return rl_state

    def render_img_state_encode(
        self,
        env: EnvRun,
        enable_encode: bool,
        append_recent_state: bool,
    ) -> np.ndarray:

        if self.config.obs_render_type == "rgb_array":
            img_state = env.render_rgb_array()
        elif self.config.obs_render_type == "terminal":
            img_state = env.render_terminal_text_to_image()
        else:
            raise NotSupportedError(self.config.obs_render_type)

        if enable_encode:
            for p in self._config._render_img_processors:
                p = cast(Any, p)
                img_state = p.remap_observation(img_state, self, env)
        img_state = cast(np.ndarray, img_state)

        if self._config.render_image_window_length > 1:
            if append_recent_state:
                self._recent_render_img_states.pop(0)
                self._recent_render_img_states.append(img_state)
                _states = self._recent_render_img_states
            else:
                _states = self._recent_render_img_states[1:] + [img_state]
            img_state = self._config.obs_render_img_space_one_step.encode_stack(_states)

        return img_state

    def action_encode(self, env_action: EnvActionType) -> TActType:
        if self._config.enable_action_decode:
            rl_act = self._config.action_space_of_env.encode_to_space(
                env_action,
                self._config.action_space,
            )
        else:
            rl_act = env_action
        return cast(TActType, rl_act)

    def action_decode(self, rl_action: TActType) -> EnvActionType:
        if self._config.enable_action_decode:
            env_act = self._config.action_space_of_env.decode_from_space(rl_action, self._config.action_space)
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
        return self._render.render(worker=self, **kwargs)

    def render_terminal_text(self, **kwargs) -> str:
        return self._render.render_terminal_text(worker=self, **kwargs)

    def render_terminal_text_to_image(self, **kwargs):
        return self._render.render_terminal_text_to_image(worker=self, **kwargs)

    def render_rgb_array(self, **kwargs) -> Optional[np.ndarray]:
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

    def create_render_image(
        self,
        add_terminal: bool = True,
        add_rgb_array: bool = True,
        add_rl_state: bool = True,
        info_text: str = "",
    ) -> np.ndarray:
        """
        ---------------------------------
        | BG white   | BG black         |
        | [env]      | [info]           |
        | [rl state] | [rl render text] |
        |            | [rl render rgb]  |
        ---------------------------------
        """
        padding = 1
        color1 = (0, 0, 0)

        # --- env image
        env_img = self._env.render_rgb_array()
        if env_img is None:
            env_img = self._env.render_terminal_text_to_image()
        assert env_img is not None
        env_img = render_funcs.add_padding(env_img, padding, padding, padding, padding, (111, 175, 0))

        # [rl state]
        if add_rl_state:
            # 同じ場合は省略
            if self.env.observation_space != self.config._rl_obs_space_one_step:
                rl_state_img = self.render_rl_image()
                if rl_state_img is not None:
                    rl_state_img = render_funcs.add_padding(
                        rl_state_img, padding, padding, padding, padding, (111, 175, 0)
                    )
                    rl_state_img = render_funcs.draw_text(rl_state_img, 0, 12, "RL")
                    env_img = render_funcs.hconcat(env_img, rl_state_img, color1)

        # [info]
        rl_img = None
        if info_text != "":
            info_img = render_funcs.text_to_rgb_array(info_text, self._render.font_name, self._render.font_size)
            info_img = render_funcs.add_padding(info_img, padding, padding, padding, padding)
            rl_img = info_img

        # [rl render text]
        if add_terminal:
            t_img = self.render_terminal_text_to_image()
            if t_img is not None:
                t_img = render_funcs.add_padding(t_img, padding, padding, padding, padding)
                if rl_img is None:
                    rl_img = t_img
                else:
                    rl_img = render_funcs.hconcat(rl_img, t_img)

        # [rl render rgb]
        if add_rgb_array:
            rl_render = self.render_rgb_array()
            if rl_render is not None:
                rl_render = render_funcs.add_padding(rl_render, padding, padding, padding, padding)
                if rl_img is None:
                    rl_img = rl_render
                else:
                    rl_img = render_funcs.hconcat(rl_img, rl_render)

        # --- env + rl
        if rl_img is not None:
            env_img = render_funcs.vconcat(env_img, rl_img, color1, (0, 0, 0))

        return env_img

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
            enable_encode=self._config.enable_state_encode,
            append_recent_state=False,
        )
        rewards = [self.reward_encode(r, env) for r in env.rewards]

        return next_state, rewards

    def backup(self) -> Any:
        d = [
            # on_start
            self._total_step,
            self._has_start,
            # on_reset
            self._player_index,
            self._episode_seed,
            self._is_reset,
            (
                [self._config.observation_space_one_step.copy_value(s) for s in self._recent_states]
                if self._config.window_length > 1
                else []
            ),
            self._config.observation_space.copy_value(self._state),
            self._config.observation_space.copy_value(self._prev_state),
            self._config.action_space.copy_value(self._action),
            self._config.action_space.copy_value(self._prev_action),
            self._reward,
            self._step_reward,
            self._invalid_actions[:],
            self._prev_invalid_actions[:],
            # env
            self._env.backup(),
        ]
        if self._config.use_render_image_state():
            d.append(
                (
                    [self._config.obs_render_img_space_one_step.copy_value(s) for s in self._recent_render_img_states]
                    if self._config.render_image_window_length > 1
                    else []
                )
            )
            d.append(self._config.obs_render_img_space.copy_value(self._render_img_state))
            d.append(self._config.obs_render_img_space.copy_value(self._prev_render_img_state))

        return d

    def restore(self, dat: Any):
        # on_start
        self._total_step = dat[0]
        self._has_start = dat[1]
        # on_reset
        self._player_index = dat[2]
        self._episode_seed = dat[3]
        self._is_reset = dat[4]
        if self._config.window_length > 1:
            self._recent_states = dat[5][:]
        self._state: TObsType = dat[6]
        self._prev_state: TObsType = dat[7]
        self._action: TActType = dat[8]
        self._prev_action = dat[9]
        self._reward = dat[10]
        self._step_reward = dat[11]
        self._invalid_actions = dat[12][:]
        self._prev_invalid_actions = dat[13]
        self._env.restore(dat[14])
        if self._config.use_render_image_state():
            if self._config.render_image_window_length > 1:
                self._recent_render_img_states = dat[15][:]
            self._render_img_state: np.ndarray = dat[16]
            self._prev_render_img_state: np.ndarray = dat[17]

        if self._context.rendering:
            self._render.cache_reset()
