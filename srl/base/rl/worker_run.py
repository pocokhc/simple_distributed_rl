import logging
from typing import Any, List, Optional, Tuple, Union, cast

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
        self._dummy_rl_one_step_states: List[RLObservationType] = [
            s.get_default() for s in self._config.observation_spaces_one_step
        ]

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
    def state(self) -> RLObservationType:
        return self._state

    @property
    def prev_action(self) -> RLActionType:
        return self._prev_action

    @property
    def reward(self) -> float:
        return self._reward

    @property
    def done(self) -> bool:
        return self._env.done

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
            self._recent_states: List[List[RLObservationType]] = [[[0]] for _ in range(self._config.window_length)]
            for _ in range(self._config.window_length):
                self._create_rl_state(self._dummy_rl_one_step_states, append_recent_state=True)

        self._info = {}
        self._state: RLObservationType = self._create_rl_state(
            self._dummy_rl_one_step_states, append_recent_state=True
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
            env_states = self._create_env_state(self.env.state)
            rl_states = self.state_encode(env_states, self._env)
            self._state = self._create_rl_state(rl_states, append_recent_state=True)
            self._info = self._worker.on_reset(self)
            self._is_reset = True
        else:
            # 2週目以降は step -> policy
            self._on_step()

        # worker policy
        self._prev_action, info = self._worker.policy(self)
        if self._config.enable_assertion_value:
            self.assert_action(self._prev_action)
        elif self._config.enable_sanitize_value:
            self._prev_action = self.sanitize_action(self._prev_action)
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
        env_states = self._create_env_state(self.env.state)
        rl_states = self.state_encode(env_states, self._env)
        self._state = self._create_rl_state(rl_states, append_recent_state=True)
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

    def _create_env_state(self, env_state: EnvObservationType) -> List[EnvObservationType]:
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
        return env_states

    def _create_rl_state(self, rl_states: List[RLObservationType], append_recent_state: bool) -> RLObservationType:
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
                s = [r[i] for r in _recent_state]
                if space.rl_type == RLTypes.DISCRETE:
                    rl_states2.append(s)
                elif space.rl_type == RLTypes.CONTINUOUS:
                    rl_states2.append(np.asarray(s, np.float32))
                elif space.rl_type == RLTypes.IMAGE:
                    rl_states2.append(np.asarray(s, np.float32))
                else:
                    # do nothing
                    rl_states2.append(s)
            rl_states = rl_states2

        if self._config.observation_space.rl_type == RLTypes.MULTI:
            return rl_states
        else:
            return rl_states[0]

    # ------------------------------
    # encode/decode
    # ------------------------------
    def state_encode(self, env_states: List[EnvObservationType], env: EnvRun) -> List[RLObservationType]:
        if self._config.enable_state_encode:
            states_rl = []
            for i in range(len(env_states)):
                for p in self._config.observation_processors_list[i]:
                    env_states[i] = p.preprocess_observation(env_states[i], env)

                space = self._config.observation_spaces_one_step[i]
                if space.rl_type == RLTypes.DISCRETE:
                    states_rl.append(space.encode_to_list_int(env_states[i]))
                elif space.rl_type == RLTypes.CONTINUOUS:
                    states_rl.append(space.encode_to_np(env_states[i], np.float32))
                elif space.rl_type == RLTypes.IMAGE:
                    states_rl.append(space.encode_to_np(env_states[i], np.float32))
                else:
                    # do nothing
                    states_rl.append(cast(RLObservationType, env_states[i]))
        else:
            states_rl = cast(List[RLObservationType], env_states)
        return states_rl

    def action_encode(self, action_env: EnvActionType) -> RLActionType:
        if self._config.enable_action_decode:
            if self._config.action_type == RLTypes.DISCRETE:
                action_rl = self._config.action_space.encode_to_int(action_env)
            elif self._config.action_type == RLTypes.CONTINUOUS:
                action_rl = self._config.action_space.encode_to_list_float(action_env)
            elif self._config.action_type == RLTypes.IMAGE:
                action_rl = self._config.action_space.encode_to_np(action_env, np.uint8)
            elif self._config.action_type == RLTypes.MULTI:
                action_rl = self._config.action_space.encode_to_list_space(action_env)
            else:
                # do nothing
                action_rl = cast(RLActionType, action_env)
        else:
            action_rl = cast(RLActionType, action_env)
        return action_rl

    def action_decode(self, action_rl: RLActionType) -> EnvActionType:
        if self._config.enable_action_decode:
            if self._config.action_type == RLTypes.DISCRETE:
                assert not isinstance(action_rl, list)
                action_env = self._config.action_space.decode_from_int(int(action_rl))
            elif self._config.action_type == RLTypes.CONTINUOUS:
                if isinstance(action_rl, list):
                    action_rl = [float(a) for a in action_rl]
                else:
                    action_rl = [float(action_rl)]
                action_env = self._config.action_space.decode_from_list_float(action_rl)
            elif self._config.action_type == RLTypes.IMAGE:
                assert isinstance(action_rl, np.ndarray)
                action_env = self._config.action_space.decode_from_np(action_rl)
            elif self._config.action_type == RLTypes.MULTI:
                action_env = self._config.action_space.decode_from_list_space(action_rl)
            else:
                action_env = cast(EnvActionType, action_rl)  # not coming
        else:
            action_env = cast(EnvActionType, action_rl)
        return action_env

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
        if self.config.action_type == RLTypes.DISCRETE:
            return [a for a in range(self.config.action_num) if a not in self.get_invalid_actions(env)]
        else:
            raise NotImplementedError("not support")

    def add_invalid_actions(self, invalid_actions: List[RLInvalidActionType]) -> None:
        if self.config.action_type == RLTypes.DISCRETE:
            self._invalid_actions += invalid_actions
            self._invalid_actions = list(set(self._invalid_actions))
        else:
            raise NotImplementedError("not support")

    # ------------------------------------
    # check
    # ------------------------------------
    def sanitize_action(self, action: Any) -> RLActionType:
        if self.config.action_type == RLTypes.DISCRETE:
            try:
                return int(cast(int, action))
            except Exception as e:
                logger.error(f"{action}({type(action)}), {e}")
        elif self.config.action_type == RLTypes.CONTINUOUS:
            try:
                if isinstance(action, list):
                    return [float(a) for a in action]
                if isinstance(action, tuple):
                    return [float(a) for a in action]
                return float(action)
            except Exception as e:
                logger.error(f"{action}({type(action)}), {e}")
        elif self.config.action_type == RLTypes.IMAGE:
            try:
                if isinstance(action, np.ndarray):
                    action = action.astype(np.uint8)
                    return action
                logger.error(f"sanitize fail. {action}({type(action)})")
            except Exception as e:
                logger.error(f"{action}({type(action)}), {e}")
        elif self.config.action_type == RLTypes.MULTI:
            try:
                if isinstance(action, tuple):
                    return list(action)
                if isinstance(action, list):
                    return action
                logger.error(f"sanitize fail. {action}({type(action)})")
            except Exception as e:
                logger.error(f"{action}({type(action)}), {e}")

        return self.config.action_space.get_default()

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
        elif self.config.action_type == RLTypes.IMAGE:
            assert isinstance(action, np.ndarray)
            assert len(action.shape) == 3
            assert action.shape[-1] == 3
        elif self.config.action_type == RLTypes.MULTI:
            assert isinstance(action, list)
        elif self.config.action_type == RLTypes.UNKNOWN:
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
        env_states = self._create_env_state(env.state)
        rl_states = self.state_encode(env_states, env)
        next_state = self._create_rl_state(rl_states, append_recent_state=False)
        rewards = [self.reward_encode(r, env) for r in env.step_rewards.tolist()]

        return next_state, rewards
