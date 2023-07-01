import logging
import pickle
import time
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Union

import numpy as np

from srl.base.define import (
    EnvActionType,
    EnvObservationType,
    EnvObservationTypes,
    InfoType,
    InvalidActionsType,
    KeyBindType,
    PlayRenderModes,
)
from srl.base.env.base import EnvBase
from srl.base.env.config import EnvConfig
from srl.base.render import Render
from srl.base.spaces.discrete import DiscreteSpace
from srl.base.spaces.space import SpaceBase

if TYPE_CHECKING:
    from srl.base.rl.worker_run import WorkerRun

logger = logging.getLogger(__name__)


class EnvRun:
    def __init__(self, env: EnvBase, config: EnvConfig) -> None:
        self.env = env
        self.config = config
        config._update_env_info(env)  # config update

        self._render = Render(self.env)
        self.set_render_options()

        self._reset_vals()
        self._is_direct_step = False
        self.init()

    def init(self):
        """reset前の状態を定義"""
        self._done = True

    # --- with
    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            import traceback

            logger.error(traceback.format_exc())

    # ------------------------------------
    # change internal state
    # ------------------------------------
    def reset(
        self,
        render_mode: Union[str, PlayRenderModes] = "",
        seed: Optional[int] = None,
    ) -> None:
        # --- seed
        self.env.set_seed(seed)

        # --- render
        if render_mode != "":
            self._render.cache_reset()
            self._render.reset(render_mode)

        # --- env reset
        self._reset_vals()
        self._state, self._info = self.env.reset()
        self._invalid_actions_list = [self.env.get_invalid_actions(i) for i in range(self.env.player_num)]
        if self.config.check_val:
            self._state = self.check_state(self._state, "state in env.reset may not be SpaceType.")
            for i, invalid_actions in enumerate(self._invalid_actions_list):
                self._invalid_actions_list[i] = self.check_invalid_actions(
                    self._invalid_actions_list[i], "invalid_actions in env.reset may not be SpaceType."
                )

    def _reset_vals(self):
        self._step_num = 0
        self._state = self.env.observation_space.get_default()
        self._done = False
        self._done_reason = ""
        self._prev_player_index = 0
        self._episode_rewards = np.zeros(self.player_num)
        self._step_rewards = np.zeros(self.player_num)
        self._invalid_actions_list = [[] for _ in range(self.env.player_num)]
        self._t0 = time.time()
        self._info = {}

    def step(
        self,
        action: EnvActionType,
        skip_function: Optional[Callable[[], None]] = None,
    ) -> None:
        assert not self.done, "It is in the done state. Please execute reset()."
        if self._is_direct_step:
            assert self.env.can_simulate_from_direct_step, "env does not support 'step' after 'direct_step'."

        # --- env step
        if self.config.check_action:
            action = self.check_action(action, "The format of 'action' entered in 'env.step' was wrong.")
        self._prev_player_index = self.env.next_player_index
        state, rewards, done, info = self.env.step(action)
        if self.config.check_val:
            state = self.check_state(state, "'state' in 'env.step' may not be SpaceType.")
            rewards = self.check_rewards(rewards, "'rewards' in 'env.step' may not be List[float].")
            done = self.check_done(done, "'done' in 'env.reset may' not be bool.")

        self._render.cache_reset()
        step_rewards = np.array(rewards, dtype=np.float32)

        # --- skip frame
        for _ in range(self.config.frameskip):
            assert self.player_num == 1, "not support"
            state, rewards, done, info = self.env.step(action)
            if self.config.check_val:
                state = self.check_state(state, "'state' in 'env.step' may not be SpaceType.")
                rewards = self.check_rewards(rewards, "'rewards' in 'env.step' may not be List[float].")
                done = self.check_done(done, "'done' in 'env.reset may' not be bool.")
            step_rewards += np.array(rewards, dtype=np.float32)
            self._render.cache_reset()
            if done:
                break

            if skip_function is not None:
                skip_function()

        return self._step(state, step_rewards, done, info)

    def _step(self, state: EnvObservationType, rewards: np.ndarray, done: bool, info: InfoType):
        self._state = state
        self._step_rewards = rewards
        self._done = done
        self._info = info

        invalid_actions = self.env.get_invalid_actions(self.next_player_index)
        if self.config.check_val:
            invalid_actions = self.check_invalid_actions(
                invalid_actions, "invalid_actions in env.reset may not be SpaceType."
            )
        self._invalid_actions_list[self.next_player_index] = invalid_actions
        self._step_num += 1
        self._episode_rewards += self.step_rewards

        # action check
        if not self.done and len(invalid_actions) > 0:
            assert len(invalid_actions) < self.action_space.n

        # done step
        if self.done:
            self._done_reason = "env"
        elif self.step_num > self.max_episode_steps:
            self._done = True
            self._done_reason = "episode max steps"
        elif self.config.episode_timeout > 0 and time.time() - self._t0 > self.config.episode_timeout:
            self._done = True
            self._done_reason = "timeout"

    def backup(self, include_env: bool = True) -> Any:
        logger.debug("env.backup")
        d = [
            self.step_num,
            self.episode_rewards,
            self.state,
            self.step_rewards,
            self.done,
            self.done_reason,
            self.prev_player_index,
            self._invalid_actions_list,
            self.info,
            self._t0,
            self._is_direct_step,
        ]
        data = [pickle.dumps(d)]
        if include_env:
            data.append(self.env.backup())
        return data

    def restore(self, data: Any) -> None:
        logger.debug("env.restore")
        d = pickle.loads(data[0])
        self._step_num = d[0]
        self._episode_rewards = d[1]
        self._state = d[2]
        self._step_rewards = d[3]
        self._done = d[4]
        self._done_reason = d[5]
        self._prev_player_index = d[6]
        self._invalid_actions_list = d[7]
        self._info = d[8]
        self._t0 = d[9]
        self._is_direct_step = d[10]
        if self._is_direct_step:
            if not self.env.can_simulate_from_direct_step:
                logger.warning("env does not support 'step' after 'direct_step'.")
        if len(data) == 2:
            self.env.restore(data[1])

    # ------------------------------------
    # check
    # ------------------------------------
    def check_action(self, action: EnvActionType, error_msg: str = "") -> EnvActionType:
        try:
            if action in self.get_invalid_actions():
                logger.error(f"{action}({type(action)}), {error_msg}, invalid action {self.get_invalid_actions()}")
            return self.env.action_space.convert(action)
        except Exception as e:
            logger.error(f"{action}({type(action)}), {error_msg}, {e}")
        return self.env.action_space.get_default()

    def check_state(self, state: EnvObservationType, error_msg: str = "") -> EnvObservationType:
        try:
            return self.env.observation_space.convert(state)
        except Exception as e:
            logger.error(f"{state}({type(state)}), {error_msg}, {e}")
        return self.env.observation_space.get_default()

    def check_rewards(self, rewards: List[float], error_msg: str = "") -> List[float]:
        try:
            for i, r in enumerate(rewards):
                try:
                    rewards[i] = float(r)
                except Exception as e:
                    logger.error(f"{rewards}({type(rewards)}, {type(r)}), {error_msg}, {e}")
                    rewards[i] = 0.0
            return rewards
        except Exception as e:
            logger.error(f"{rewards}({type(rewards)}), {error_msg}, {e}")
        return [0.0 for _ in range(self.player_num)]

    def check_done(self, done: bool, error_msg: str = "") -> bool:
        try:
            return bool(done)
        except Exception as e:
            logger.error(f"{done}({type(done)}), {error_msg}, {e}")
        return False

    def check_invalid_actions(self, invalid_actions: InvalidActionsType, error_msg: str = "") -> InvalidActionsType:
        try:
            for j, invalid_action in enumerate(invalid_actions):
                invalid_actions[j] = int(invalid_action)
            return invalid_actions
        except Exception as e:
            logger.error(f"{invalid_actions}, {error_msg}, {e}")
        return []

    # ------------------------------------
    # No internal state change
    # ------------------------------------
    # implement properties
    @property
    def action_space(self) -> SpaceBase:
        return self.env.action_space

    @property
    def observation_space(self) -> SpaceBase:
        return self.env.observation_space

    @property
    def observation_type(self) -> EnvObservationTypes:
        return self.env.observation_type

    @property
    def max_episode_steps(self) -> int:
        return self.config.max_episode_steps

    @property
    def player_num(self) -> int:
        return self.env.player_num

    @property
    def reward_info(self) -> dict:
        return self.env.reward_info

    # state properties
    @property
    def state(self) -> EnvObservationType:
        return self._state

    @property
    def prev_player_index(self) -> int:
        return self._prev_player_index

    @property
    def next_player_index(self) -> int:
        return self.env.next_player_index

    @property
    def step_num(self) -> int:
        return self._step_num

    @property
    def done(self) -> bool:
        return self._done

    @property
    def done_reason(self) -> str:
        return self._done_reason

    @property
    def episode_rewards(self) -> np.ndarray:
        return self._episode_rewards

    @property
    def step_rewards(self) -> np.ndarray:
        return self._step_rewards

    @property
    def info(self) -> InfoType:
        return self._info

    @property
    def reward(self) -> float:
        """直前のrewardを返す"""
        return self.step_rewards[self.prev_player_index]

    @property
    def invalid_actions(self) -> InvalidActionsType:
        """現プレイヤーのinvalid actionsを返す"""
        return self._invalid_actions_list[self.next_player_index]

    # invalid actions
    def get_invalid_actions(self, player_index: int = -1) -> InvalidActionsType:
        if isinstance(self.action_space, DiscreteSpace):
            if player_index == -1:
                player_index = self.next_player_index
            return self._invalid_actions_list[player_index]
        else:
            return []

    def get_valid_actions(self, player_index: int = -1) -> InvalidActionsType:
        if isinstance(self.action_space, DiscreteSpace):
            invalid_actions = self.get_invalid_actions(player_index)
            return [a for a in range(self.action_space.n) if a not in invalid_actions]
        else:
            assert False, "not support"

    def add_invalid_actions(self, invalid_actions: InvalidActionsType, player_index: int) -> None:
        self._invalid_actions_list[player_index] += invalid_actions
        self._invalid_actions_list[player_index] = list(set(self._invalid_actions_list[player_index]))

    # other functions
    def action_to_str(self, action: Union[str, EnvActionType]) -> str:
        return self.env.action_to_str(action)

    def get_key_bind(self) -> KeyBindType:
        return self.env.get_key_bind()

    def make_worker(
        self,
        name: str,
        distributed: bool = False,
        enable_raise: bool = True,
        env_worker_kwargs: dict = {},
        actor_id: int = 0,
    ) -> Optional["WorkerRun"]:
        env_worker_kwargs = env_worker_kwargs.copy()
        worker = self.env.make_worker(name, **env_worker_kwargs)
        if worker is None:
            if enable_raise:
                raise ValueError(f"'{name}' worker is not found.")
            return None

        from srl.base.rl.worker_run import WorkerRun

        return WorkerRun(worker, self, distributed, actor_id)

    def get_original_env(self) -> object:
        return self.env.get_original_env()

    def get_env_base(self) -> EnvBase:
        return self.env

    # ------------------------------------
    # render
    # ------------------------------------
    def set_render_options(
        self,
        interval: float = -1,  # ms
        scale: float = 1.0,
        font_name: str = "",
        font_size: int = 12,
    ) -> float:
        if interval > 0:
            pass
        elif self.config.render_interval > 0:
            interval = self.config.render_interval
        else:
            interval = self.env.render_interval

        self._render.interval = interval
        self._render.scale = scale
        self._render.font_name = font_name
        self._render.font_size = font_size
        return interval

    def render(self, **kwargs) -> Union[None, str, np.ndarray]:
        return self._render.render(**kwargs)

    def render_terminal(self, return_text: bool = False, **kwargs) -> Union[None, str]:
        return self._render.render_terminal(return_text, **kwargs)

    def render_rgb_array(self, **kwargs) -> np.ndarray:
        return self._render.render_rgb_array(**kwargs)

    def render_window(self, **kwargs) -> np.ndarray:
        return self._render.render_window(**kwargs)

    # ------------------------------------
    # direct
    # ------------------------------------
    def direct_step(self, *args, **kwargs) -> None:
        self._is_direct_step = True
        self.is_start_episode, state, player_index, info = self.env.direct_step(*args, **kwargs)
        if self.config.check_val:
            s = "'is_start_episode' in 'env.direct_step may' not be bool."
            self.is_start_episode = self.check_done(self.is_start_episode, s)
            state = self.check_state(state, "'state' in 'env.direct_step' may not be SpaceType.")

        if self.is_start_episode:
            self._reset_vals()
        self._step(state, np.zeros((self.player_num,)), False, info)

    def decode_action(self, action: EnvActionType) -> Any:
        if self.config.check_action:
            action = self.check_action(action, "The format of 'action' entered in 'env.decode_action' was wrong.")
        return self.env.decode_action(action)

    # ------------------------------------
    # util functions
    # ------------------------------------
    def sample_action(self, player_index: int = -1) -> EnvActionType:
        return self.action_space.sample(self.get_invalid_actions(player_index))

    def sample_observation(self, player_index: int = -1) -> EnvActionType:
        return self.observation_space.sample(self.get_invalid_actions(player_index))

    def copy(self):
        org_env = self.env.__class__()
        env = self.__class__(org_env, self.config)
        env.restore(self.backup())
        return env

    def to_dict(self) -> dict:
        conf = {}
        for k, v in self.__dict__.items():
            if type(v) in [int, float, bool, str]:
                conf[k] = v
        return conf
