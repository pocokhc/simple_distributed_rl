import logging
import pickle
import random
import time
import traceback
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Union

import numpy as np

from srl.base.define import (
    DoneTypes,
    EnvActionType,
    EnvObservationType,
    EnvObservationTypes,
    InfoType,
    InvalidActionsType,
    KeyBindType,
    RenderModes,
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
    def __init__(self, config: EnvConfig) -> None:
        from srl.base.env.registration import make_base

        self.config = config
        self.env = make_base(self.config)
        self.config._update_env_info(self.env)  # config update

        self._render = Render(self.env)
        self.set_render_options()

        self._reset_vals()
        self._is_direct_step = False
        self.init()

    def init(self):
        """reset前の状態を定義"""
        self._done = DoneTypes.RESET
        self._done_reason = ""

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
            logger.error(traceback.format_exc())

    def remake(self):
        from srl.base.env.registration import make_base

        self.close()
        self.env = make_base(self.config)

    # ------------------------------------
    # change internal state
    # ------------------------------------
    def reset(
        self,
        render_mode: Union[str, RenderModes] = "",
        seed: Optional[int] = None,
    ) -> None:
        # --- seed
        self.env.set_seed(seed)

        # --- render
        self._render.reset(render_mode)

        # --- env reset
        self._reset_vals()
        self._state, self._info = self.env.reset()
        if self.config.random_noop_max > 0:
            for _ in range(random.randint(0, self.config.random_noop_max)):
                self._state, rewards, env_done, self._info = self.env.step(self.env.action_space.get_default())
                assert not DoneTypes.done(env_done), "Terminated during noop step."
        self._invalid_actions_list = [self.env.get_invalid_actions(i) for i in range(self.env.player_num)]
        if self.config.enable_assertion_value:
            assert self.observation_space.check_val(self._state)
            [self.assert_invalid_actions(a) for a in self._invalid_actions_list]
        elif self.config.enable_sanitize_value:
            self._state = self.sanitize_state(self._state, "state in env.reset may not be SpaceType.")
            self._invalid_actions_list = [
                self.sanitize_invalid_actions(a, "invalid_actions in env.reset may not be SpaceType.")
                for a in self._invalid_actions_list
            ]

    def _reset_vals(self):
        self._step_num: int = 0
        self._state = self.env.observation_space.get_default()
        self._done: DoneTypes = DoneTypes.NONE
        self._done_reason: str = ""
        self._prev_player_index: int = 0
        self._episode_rewards = np.zeros(self.player_num)
        self._step_rewards = np.zeros(self.player_num)
        self._invalid_actions_list: list = [[] for _ in range(self.env.player_num)]
        self._t0 = time.time()
        self._info: dict = {}

    def step(self, action: EnvActionType, frameskip_function: Optional[Callable[[], None]] = None) -> None:
        assert self._done == DoneTypes.NONE, f"It is in the done state. Please execute reset(). ({self._done})"
        if self._is_direct_step:
            assert self.env.can_simulate_from_direct_step, "env does not support 'step' after 'direct_step'."

        # --- env step
        if self.config.enable_assertion_value:
            self.assert_action(action)
        elif self.config.enable_sanitize_value:
            action = self.sanitize_action(action, "The format of 'action' entered in 'env.step' was wrong.")
        self._prev_player_index = self.env.next_player_index

        f_except = None
        try:
            state, rewards, env_done, info = self.env.step(action)
        except Exception as e:
            f_except = e

        if self.config.enable_assertion_value:
            assert f_except is None, f_except
            self.assert_state(state)
            self.assert_rewards(rewards)
            self.assert_done(env_done)
        elif self.config.enable_sanitize_value:
            state = self.sanitize_state(state, "'state' in 'env.step' may not be SpaceType.")
            rewards = self.sanitize_rewards(rewards, "'rewards' in 'env.step' may not be List[float].")
            env_done = self.sanitize_done(env_done, "'done' in 'env.reset may' not be bool.")
        else:
            if f_except is not None:
                self.remake()
                self._done = DoneTypes.TRUNCATED
                self._done_reason = "step exception"
                return

        self._render.cache_reset()
        step_rewards = np.array(rewards, dtype=np.float32)

        # --- skip frame
        for _ in range(self.config.frameskip):
            assert self.player_num == 1, "not support"
            state, rewards, env_done, info = self.env.step(action)
            if self.config.enable_assertion_value:
                self.assert_state(state)
                self.assert_rewards(rewards)
                self.assert_done(env_done)
            elif self.config.enable_sanitize_value:
                state = self.sanitize_state(state, "'state' in 'env.step' may not be SpaceType.")
                rewards = self.sanitize_rewards(rewards, "'rewards' in 'env.step' may not be List[float].")
                env_done = self.sanitize_done(env_done, "'done' in 'env.reset may' not be bool.")

            step_rewards += np.array(rewards, dtype=np.float32)
            self._render.cache_reset()
            if DoneTypes.done(env_done):
                break

            if frameskip_function is not None:
                frameskip_function()

        return self._step(state, step_rewards, env_done, info)

    def _step(self, state: EnvObservationType, rewards: np.ndarray, env_done: Union[bool, DoneTypes], info: InfoType):
        self._state = state
        self._step_rewards = rewards
        self._done = DoneTypes.from_bool(env_done)
        self._info = info

        invalid_actions = self.env.get_invalid_actions(self.next_player_index)
        if self.config.enable_assertion_value:
            self.assert_invalid_actions(invalid_actions)
        elif self.config.enable_sanitize_value:
            invalid_actions = self.sanitize_invalid_actions(
                invalid_actions, "invalid_actions in env.reset may not be SpaceType."
            )
        self._invalid_actions_list[self.next_player_index] = invalid_actions
        self._step_num += 1
        self._episode_rewards += self.step_rewards

        # action check
        if not self._done and len(invalid_actions) > 0:
            assert len(invalid_actions) < self.action_space.n

        # done step
        if self._done == DoneTypes.NONE:
            if self.step_num > self.max_episode_steps:
                self._done = DoneTypes.TRUNCATED
                self._done_reason = "episode step over"
            elif self.config.episode_timeout > 0 and time.time() - self._t0 > self.config.episode_timeout:
                self._done = DoneTypes.TRUNCATED
                self._done_reason = "timeout"

    def backup(self, include_env: bool = True) -> Any:
        logger.debug("env.backup")
        d = [
            self._step_num,
            self._episode_rewards,
            self._state,
            self._step_rewards,
            self._done,
            self._done_reason,
            self._prev_player_index,
            self._invalid_actions_list,
            self._info,
            self._t0,
            self._is_direct_step,
        ]
        data: list = [pickle.dumps(d)]
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
    def sanitize_action(self, action: EnvActionType, error_msg: str = "") -> EnvActionType:
        try:
            if action in self.get_invalid_actions():
                logger.error(f"{action}({type(action)}), {error_msg}, invalid action {self.get_invalid_actions()}")
            return self.env.action_space.convert(action)
        except Exception as e:
            logger.error(f"{action}({type(action)}), {error_msg}, {e}")
        return self.env.action_space.get_default()

    def assert_action(self, action: EnvActionType):
        assert self.action_space.check_val(action), f"The type of action is different. {action}({type(action)})"

    def sanitize_state(self, state: EnvObservationType, error_msg: str = "") -> EnvObservationType:
        try:
            return self.env.observation_space.convert(state)
        except Exception as e:
            logger.error(f"{state}({type(state)}), {error_msg}, {e}")
        return self.env.observation_space.get_default()

    def assert_state(self, state: EnvObservationType):
        assert self.observation_space.check_val(state), f"The type of state is different. {state}({type(state)})"

    def sanitize_rewards(self, rewards: List[float], error_msg: str = "") -> List[float]:
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

    def assert_rewards(self, rewards: List[float]):
        assert isinstance(rewards, list), f"Rewards must be arrayed. {rewards}({type(rewards)})"
        assert (
            len(rewards) == self.env.player_num
        ), f"Array sizes are different. {len(rewards)} != {self.env.player_num}, {rewards}({type(rewards)})"
        for r in rewards:
            assert isinstance(r, float), f"The type of reward is different. {r}({type(r)}), {rewards}"

    def sanitize_done(self, done: Union[bool, DoneTypes], error_msg: str = "") -> DoneTypes:
        try:
            return DoneTypes.from_bool(done)
        except Exception as e:
            logger.error(f"{done}({type(done)}), {error_msg}, {e}")
        return DoneTypes.TRUNCATED

    def assert_done(self, done: Union[bool, DoneTypes]):
        if isinstance(done, bool):
            pass
        elif isinstance(done, DoneTypes):
            pass
        else:
            assert False, f"The type of reward is different. {done}({type(done)})"

    def sanitize_invalid_actions(self, invalid_actions: InvalidActionsType, error_msg: str = "") -> InvalidActionsType:
        try:
            for j, invalid_action in enumerate(invalid_actions):
                invalid_actions[j] = int(invalid_action)
            return invalid_actions
        except Exception as e:
            logger.error(f"{invalid_actions}, {error_msg}, {e}")
        return []

    def assert_invalid_actions(self, invalid_actions: InvalidActionsType):
        assert isinstance(
            invalid_actions, list
        ), f"invalid_actions must be arrayed. {invalid_actions}({type(invalid_actions)})"
        for a in invalid_actions:
            assert isinstance(a, int), f"The type of reward is different. {a}({type(a)})"

    # ------------------------------------
    # No internal state change
    # ------------------------------------
    @property
    def name(self) -> str:
        return self.config.name

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

    @property
    def info_types(self) -> dict:
        return self.env.info_types

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
        return self._done != DoneTypes.NONE

    @property
    def done_type(self) -> DoneTypes:
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
        if player_index == -1:
            player_index = self.next_player_index
        return self._invalid_actions_list[player_index]

    def get_valid_actions(self, player_index: int = -1) -> InvalidActionsType:
        invalid_actions = self.get_invalid_actions(player_index)
        return [a for a in range(self.action_space.n) if a not in invalid_actions]

    def add_invalid_actions(self, invalid_actions: InvalidActionsType, player_index: int) -> None:
        if self.config.enable_assertion_value:
            self.assert_invalid_actions(invalid_actions)
        elif self.config.enable_sanitize_value:
            invalid_actions = self.sanitize_invalid_actions(
                invalid_actions, "invalid_actions in 'env.add_invalid_actions' may not be SpaceType."
            )

        if isinstance(self.action_space, DiscreteSpace):
            self._invalid_actions_list[player_index] += invalid_actions
            self._invalid_actions_list[player_index] = list(set(self._invalid_actions_list[player_index]))
        else:
            assert False, "not support"

    # other functions
    def action_to_str(self, action: Union[str, EnvActionType]) -> str:
        return self.env.action_to_str(action)

    def get_key_bind(self) -> KeyBindType:
        return self.env.get_key_bind()

    def make_worker(
        self,
        name: str,
        env_worker_kwargs: dict = {},
        enable_raise: bool = True,
        distributed: bool = False,
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

    def get_original_env(self) -> Any:
        return self.env.get_original_env()

    def get_env_base(self) -> EnvBase:
        return self.env

    def set_done(self):
        self._done = DoneTypes.TRUNCATED
        self._done_reason = "set_done"

    # ------------------------------------
    # render
    # ------------------------------------
    def set_render_options(
        self,
        interval: float = -1,  # ms
        scale: float = 1.0,
        font_name: str = "",
        font_size: int = 18,
    ) -> float:
        if interval > 0:
            pass
        elif self.config.render_interval > 0:
            interval = self.config.render_interval
        else:
            interval = self.env.render_interval

        self._render.set_render_options(interval, scale, font_name, font_size)
        return interval

    def render(self, **kwargs):
        self._render.render(render_window=True, **kwargs)

    def render_ansi(self, **kwargs) -> str:
        return self._render.render_ansi(**kwargs)

    def render_rgb_array(self, **kwargs) -> np.ndarray:
        return self._render.render_rgb_array(**kwargs)

    # ------------------------------------
    # direct
    # ------------------------------------
    def direct_step(self, *args, **kwargs) -> None:
        self._is_direct_step = True
        self.is_start_episode, state, player_index, info = self.env.direct_step(*args, **kwargs)
        if self.config.enable_assertion_value:
            self.assert_done(self.is_start_episode)
            self.assert_state(state)
        if self.config.enable_sanitize_value:
            s = "'is_start_episode' in 'env.direct_step may' not be bool."
            self.is_start_episode = self.sanitize_done(self.is_start_episode, s)
            state = self.sanitize_state(state, "'state' in 'env.direct_step' may not be SpaceType.")

        if self.is_start_episode:
            self._reset_vals()
        self._step(state, np.zeros((self.player_num,)), False, info)

    def decode_action(self, action: EnvActionType) -> Any:
        if self.config.enable_assertion_value:
            self.assert_action(action)
        elif self.config.enable_sanitize_value:
            action = self.sanitize_action(action, "The format of 'action' entered in 'env.decode_action' was wrong.")
        return self.env.decode_action(action)

    # ------------------------------------
    # util functions
    # ------------------------------------
    def sample_action(self, player_index: int = -1) -> EnvActionType:
        return self.action_space.sample(self.get_invalid_actions(player_index))

    def sample_observation(self, player_index: int = -1) -> EnvActionType:
        return self.observation_space.sample(self.get_invalid_actions(player_index))

    def copy(self):
        env = self.__class__(self.config)
        env.restore(self.backup())
        return env

    def to_dict(self) -> dict:
        conf = {}
        for k, v in self.__dict__.items():
            if type(v) in [int, float, bool, str]:
                conf[k] = v
        return conf
