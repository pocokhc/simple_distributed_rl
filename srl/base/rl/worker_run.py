import logging
from typing import List, Optional, Tuple, Union, cast

import numpy as np

from srl.base.define import (
    EnvActionType,
    EnvObservationType,
    InfoType,
    InvalidActionsType,
    RenderModes,
    RLActionType,
    RLObservationType,
    RLTypes,
)
from srl.base.env.env_run import EnvRun
from srl.base.render import Render
from srl.base.rl.base import RLWorker
from srl.base.rl.config import RLConfig

logger = logging.getLogger(__name__)


class WorkerRun:
    def __init__(
        self,
        worker: RLWorker,
        env: EnvRun,
        distributed: bool = False,
        actor_id: int = 0,
    ):
        worker.config.setup(env, enable_log=False)
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
        self._invalid_actions: InvalidActionsType = []
        self._render = Render(worker)

        self._total_step = 0

        if self._config.window_length > 1:
            self._dummy_state = self._config.create_dummy_state(is_one=True)

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
    def state(self) -> RLObservationType:
        return self._state

    @property
    def reward(self) -> float:
        return self._reward

    @property
    def done(self) -> bool:
        return self._done

    @property
    def invalid_actions(self) -> InvalidActionsType:
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
        self._rendering = RenderModes.is_rendering(render_mode)

        self._is_reset = False
        self._step_reward = 0

        self._info = {}
        self._state = self._config.create_dummy_state()
        self._reward = 0
        self._done = False
        self._set_invalid_actions()

        if self._config.window_length > 1:
            self._recent_states: List[RLObservationType] = [
                self._dummy_state for _ in range(self._config.window_length)
            ]

        [r.on_reset(self._env) for r in self._config._run_processors]
        self._render.reset(render_mode)

    def policy(self) -> EnvActionType:
        if not self._is_reset:
            # 1週目は reset -> policy
            self._set_invalid_actions()
            self._state = self.state_encode(self.env.state, self._env, append_recent_state=True)
            self._info = self._worker.on_reset(self)
            self._is_reset = True
        else:
            # 2週目以降は step -> policy
            self._on_step()

        # worker policy
        action, info = self._worker.policy(self)
        if self._config.enable_assertion_value:
            self.assert_action(action)
        elif self._config.enable_sanitize_value:
            action = self.sanitize_action(action)
        action = self.action_decode(action)
        self._info.update(info)

        if self._rendering:
            self._render.cache_reset()

        return action

    def on_step(self) -> None:
        # 初期化前はskip
        if not self._is_reset:
            return
        self._total_step += 1

        # 相手の番のrewardも加算
        self._step_reward += self._env.step_rewards[self.player_index]

        if self._done and not self._env.done:
            self._env._done = True
            self._env._done_reason = "RL"

        # 終了ならon_step実行
        if self._env.done:
            self._on_step()
            if self._rendering:
                self._render.cache_reset()

    def _on_step(self):
        # encode -> set invalid -> on_step -> reward=0
        self._state = self.state_encode(self._env.state, self._env, append_recent_state=True)
        self._reward = self.reward_encode(self._step_reward, self._env)
        self._done = self.done_encode(self._env.done, self._env)
        self._set_invalid_actions()
        self._info = self._worker.on_step(self)
        self._step_reward = 0

    def _set_invalid_actions(self):
        self._invalid_actions = cast(
            InvalidActionsType, [self.action_encode(a) for a in self._env.get_invalid_actions(self.player_index)]
        )

    # ------------------------------
    # encode/decode
    # ------------------------------
    def state_encode(self, state: EnvObservationType, env: EnvRun, append_recent_state: bool) -> RLObservationType:
        if self._config.enable_state_encode:
            for p in self.config.run_processors:
                state = p.preprocess_observation(state, env)

            if self._config.observation_type == RLTypes.DISCRETE:
                state = self._config.observation_one_step_space.encode_to_int_np(state)
            elif self._config.observation_type == RLTypes.CONTINUOUS:
                state = self._config.observation_one_step_space.encode_to_np(state)
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
            for p in self.config.run_processors:
                reward = p.preprocess_reward(reward, env)
        return reward

    def done_encode(self, done: bool, env: EnvRun) -> bool:
        if self._config.enable_done_encode:
            for p in self.config.run_processors:
                done = p.preprocess_done(done, env)
        return done

    # ------------------------------------
    # invalid
    # ------------------------------------
    def get_invalid_actions(self, env: Optional[EnvRun] = None) -> InvalidActionsType:
        return self._invalid_actions

    def get_valid_actions(self, env: Optional[EnvRun] = None) -> InvalidActionsType:
        if self.config.action_type == RLTypes.DISCRETE:
            return [a for a in range(self.config.action_num) if a not in self.get_invalid_actions(env)]
        else:
            raise NotImplementedError("not support")

    def add_invalid_actions(self, invalid_actions: InvalidActionsType) -> None:
        if self.config.action_type == RLTypes.DISCRETE:
            self._invalid_actions += invalid_actions
            self._invalid_actions = list(set(self._invalid_actions))
        else:
            raise NotImplementedError("not support")

    # ------------------------------------
    # check
    # ------------------------------------
    def sanitize_action(self, action: RLActionType) -> RLActionType:
        if self.config.action_type == RLTypes.DISCRETE:
            try:
                return int(cast(int, action))
            except Exception as e:
                logger.error(f"{action}({type(action)}), {e}")
            return 0
        else:
            try:
                if isinstance(action, list):
                    return [float(a) for a in action]
                if isinstance(action, tuple):
                    return [float(a) for a in action]
                return float(action)
            except Exception as e:
                logger.error(f"{action}({type(action)}), {e}")
            return 0.0

    def assert_action(self, action: RLActionType):
        if self.config.action_type == RLTypes.DISCRETE:
            assert isinstance(action, int), f"The type of action is different. {action}({type(action)})"
        elif self.config.action_type == RLTypes.CONTINUOUS:
            assert isinstance(action, float) or isinstance(
                action, list
            ), f"The type of action is different. {action}({type(action)})"
            if isinstance(action, list):
                for a in action:
                    assert isinstance(a, float), f"The type of action is different. {a}({type(a)})"
        elif self.config.action_type == RLTypes.ANY:
            assert (
                isinstance(action, int) or isinstance(action, float) or isinstance(action, list)
            ), f"The type of action is different. {action}({type(action)})"
            if isinstance(action, list):
                for a in action:
                    assert isinstance(a, float), f"The type of action is different. {a}({type(a)})"
        else:
            raise ValueError(self.config.action_type)

    # ------------------------------------
    # render functions
    # ------------------------------------
    def set_render_options(
        self,
        interval: float = -1,  # ms
        scale: float = 1.0,
        font_name: str = "",
        font_size: int = 12,
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
