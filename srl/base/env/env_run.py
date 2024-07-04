import logging
import random
import time
import traceback
from typing import Any, Callable, List, Optional, Union

import numpy as np

from srl.base.context import RunContext
from srl.base.define import DoneTypes, EnvActionType, EnvObservationType, KeyBindType, RenderModes
from srl.base.env.config import EnvConfig
from srl.base.exception import SRLError
from srl.base.info import Info
from srl.base.render import Render
from srl.base.spaces.discrete import DiscreteSpace
from srl.base.spaces.space import SpaceBase

logger = logging.getLogger(__name__)


class EnvRun:
    def __init__(self, config: EnvConfig) -> None:
        # restore/backup用に状態は意識して管理

        from srl.base.env.registration import make_base

        self.config = config
        self.env = make_base(self.config)

        # --- processor
        self._processors = [c.copy() for c in self.config.processors]
        self._processors_reset: Any = [c for c in self._processors if hasattr(c, "remap_reset")]
        self._processors_step_action: Any = [c for c in self._processors if hasattr(c, "remap_step_action")]
        self._processors_step: Any = [c for c in self._processors if hasattr(c, "remap_step")]
        self._processors_step_invalid_actions: Any = [
            c for c in self._processors if hasattr(c, "remap_step_invalid_actions")
        ]

        # --- space
        self._action_space = self.env.action_space
        self._observation_space = self.env.observation_space
        for p in self._processors:
            self._action_space = p.remap_action_space(self._action_space, self)
            self._observation_space = p.remap_observation_space(self._observation_space, self)

        # --- init val
        self.config._update_env_info(self.env)  # config update
        self._render = Render(self.env)
        self._reset_vals()
        self._done = DoneTypes.RESET
        self._is_direct_step = False
        self._has_start = False

    def _reset_vals(self):
        self._step_num: int = 0
        self._state = self.env.observation_space.get_default()
        self._done: DoneTypes = DoneTypes.NONE
        self._done_reason: str = ""
        self._prev_player_index: int = 0
        self._episode_rewards = np.zeros(self.player_num)
        self._step_rewards = np.zeros(self.player_num)
        self._invalid_actions_list: List[List[EnvActionType]] = [[] for _ in range(self.env.player_num)]
        self._t0 = time.time()
        self._info = Info()

    def backup(self) -> Any:
        # - spaceは状態を持たない
        # - renderはcacheクリア
        d = [
            # reset_vals
            self._step_num,
            self._observation_space.copy_value(self._state),
            self._done,
            self._done_reason,
            self._prev_player_index,
            self._episode_rewards.copy(),
            self._step_rewards.copy(),
            [s[:] for s in self._invalid_actions_list],
            self._t0,
            self._info.copy(),
            # init val
            self._is_direct_step,
            self._has_start,
            # processor
            [p.backup() for p in self._processors],
            # env
            self.env.backup(),
        ]
        return d

    def restore(self, dat: Any) -> None:
        # reset_vals
        self._step_num = dat[0]
        self._state = dat[1]
        self._done = dat[2]
        self._done_reason = dat[3]
        self._prev_player_index = dat[4]
        self._episode_rewards = dat[5].copy()
        self._step_rewards = dat[6].copy()
        self._invalid_actions_list = dat[7][:]
        self._t0 = dat[8]
        self._info = dat[9].copy()
        # init val
        self._is_direct_step = dat[10]
        self._has_start = dat[11]
        # processor
        [p.restore(dat[12][i]) for i, p in enumerate(self._processors)]
        # env
        self.env.restore(dat[13])

        # render
        self._render.cache_reset()

        if self._is_direct_step:
            if not self.env.can_simulate_from_direct_step:
                logger.warning("env does not support 'step' after 'direct_step'.")

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
    # run functions
    # ------------------------------------
    def setup(self, context: Optional[RunContext] = None, render_mode: Union[str, RenderModes] = RenderModes.none):
        if context is None:
            context = RunContext(self.config, None)
        if render_mode != RenderModes.none:
            context.render_mode = render_mode

        # --- reset前の状態を設定
        self._done = DoneTypes.RESET
        self._done_reason = ""

        # --- render
        render_mode = context.render_mode
        if self.config.override_render_mode != RenderModes.none:
            render_mode = self.config.override_render_mode
        self._render.set_render_mode(render_mode)

        # --- processor
        [p.setup(self) for p in self._processors]

        # --- env
        self.env.setup(**context.to_dict())
        self._has_start = True

    def reset(self, seed: Optional[int] = None) -> None:
        if not self._has_start:
            raise SRLError("Cannot call env.reset() before calling env.setup()")

        # --- seed
        self.env.set_seed(seed)

        # --- env reset
        self._reset_vals()
        self._state, info = self.env.reset()
        if self.config.random_noop_max > 0:
            for _ in range(random.randint(0, self.config.random_noop_max)):
                self._state, rewards, env_done, info = self.env.step(self.env.action_space.get_default())
                assert not DoneTypes.done(env_done), "Terminated during noop step."
        self._invalid_actions_list = [self.env.get_invalid_actions(i) for i in range(self.env.player_num)]

        # --- processor
        for p in self._processors_reset:
            self._state, info = p.remap_reset(self._state, info, self)

        # info
        for k, v in info.items():
            self._info.set_scalar(k, v)

        # --- check
        if self.config.enable_assertion:
            assert self.observation_space.check_val(self._state)
            [self.assert_invalid_actions(a) for a in self._invalid_actions_list]
        elif self.config.enable_sanitize:
            self._state = self.sanitize_state(self._state, "state in env.reset may not be SpaceType.")
            self._invalid_actions_list = [
                self.sanitize_invalid_actions(a, "invalid_actions in env.reset may not be SpaceType.")
                for a in self._invalid_actions_list
            ]

    def step(
        self,
        action: EnvActionType,
        frameskip: int = 0,
        frameskip_function: Optional[Callable[[], None]] = None,
    ) -> None:
        if self._done == DoneTypes.TRUNCATED:
            return
        if self._done != DoneTypes.NONE:
            raise SRLError(f"It is in the done state. Please execute reset(). ({self._done})")
        if self._is_direct_step and (not self.env.can_simulate_from_direct_step):
            raise SRLError("env does not support 'step' after 'direct_step'.")

        # --- action processor
        for p in self._processors_step_action:
            action = p.remap_step_action(action, self)

        # --- env step
        if self.config.enable_assertion:
            self.assert_action(action)
        elif self.config.enable_sanitize:
            action = self.sanitize_action(action, "The format of 'action' entered in 'env.step' was wrong.")
        self._prev_player_index = self.env.next_player_index

        state, rewards, env_done, info = self._step1(action)
        self._render.cache_reset()
        step_rewards = np.array(rewards)

        # --- skip frame
        for _ in range(self.config.frameskip + frameskip):
            if DoneTypes.done(env_done):
                break
            state, rewards, env_done, info = self._step1(action)

            step_rewards += np.array(rewards)
            self._render.cache_reset()

            if frameskip_function is not None:
                frameskip_function()

        return self._step2(state, step_rewards, env_done, info)

    def _step1(self, action):
        f_except = None
        try:
            state, rewards, env_done, info = self.env.step(action)
            for p in self._processors_step:
                state, rewards, env_done, info = p.remap_step(state, rewards, env_done, info, self)
        except Exception:
            f_except = traceback.format_exc()

        if self.config.enable_assertion:
            assert f_except is None, f_except
            self.assert_state(state)
            self.assert_rewards(rewards)
            self.assert_done(env_done)
        elif f_except is not None:
            self.remake()
            self._done = DoneTypes.TRUNCATED
            self._done_reason = "step exception"
            s = "An exception occurred in env.step. Recreate.\n" + f_except
            print(s)
            logger.warning(s)
            return state, rewards, env_done, info
        elif self.config.enable_sanitize:
            state = self.sanitize_state(state, "'state' in 'env.step' may not be SpaceType.")
            rewards = self.sanitize_rewards(rewards, "'rewards' in 'env.step' may not be List[float].")
            env_done = self.sanitize_done(env_done, "'done' in 'env.reset may' not be bool.")
        return state, rewards, env_done, info

    def _step2(self, state: EnvObservationType, rewards: np.ndarray, env_done: Union[bool, DoneTypes], info: dict):
        self._state = state
        self._step_rewards = rewards
        self._done = DoneTypes.from_bool(env_done)

        # info
        for k, v in info.items():
            self._info.set_scalar(k, v)

        invalid_actions = self.env.get_invalid_actions(self.next_player_index)
        for p in self._processors_step_invalid_actions:
            invalid_actions = p.remap_step_invalid_actions(invalid_actions, self)
        if self.config.enable_assertion:
            self.assert_invalid_actions(invalid_actions)
        elif self.config.enable_sanitize:
            invalid_actions = self.sanitize_invalid_actions(
                invalid_actions, "invalid_actions in env.reset may not be SpaceType."
            )
        self._invalid_actions_list[self.next_player_index] = invalid_actions
        self._step_num += 1
        self._episode_rewards += self._step_rewards

        # action check
        if not self._done and len(invalid_actions) > 0:
            # 有効なアクションがある事
            if isinstance(self.action_space, DiscreteSpace):
                assert len(invalid_actions) < self.action_space.n

        # done step
        if self._done == DoneTypes.NONE:
            if self.step_num > self.max_episode_steps:
                self._done = DoneTypes.TRUNCATED
                self._done_reason = "episode step over"
            elif self.config.episode_timeout > 0 and time.time() - self._t0 > self.config.episode_timeout:
                self._done = DoneTypes.TRUNCATED
                self._done_reason = "timeout"

    # ------------------------------------
    # check
    # ------------------------------------
    def sanitize_action(self, action: EnvActionType, error_msg: str = "") -> EnvActionType:
        try:
            for inv_act in self.get_invalid_actions():
                if action == inv_act:
                    logger.error(f"{action}({type(action)}), {error_msg}, invalid action {self.get_invalid_actions()}")
                    break
            return self._action_space.sanitize(action)
        except Exception as e:
            logger.error(f"{action}({type(action)}), {error_msg}, {e}")
        return self._action_space.get_default()

    def assert_action(self, action: EnvActionType):
        assert self._action_space.check_val(action), f"The type of action is different. {action}({type(action)})"

    def sanitize_state(self, state: EnvObservationType, error_msg: str = "") -> EnvObservationType:
        try:
            return self._observation_space.sanitize(state)
        except Exception as e:
            logger.error(f"{state}({type(state)}), {error_msg}, {e}")
        return self._observation_space.get_default()

    def assert_state(self, state: EnvObservationType):
        assert self._observation_space.check_val(state), f"The type of state is different. {state}({type(state)})"

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

    def sanitize_invalid_actions(self, invalid_actions, error_msg: str = "") -> List[EnvActionType]:
        try:
            for j in range(len(invalid_actions)):
                invalid_actions[j] = self.action_space.sanitize(invalid_actions[j])
            return invalid_actions
        except Exception as e:
            logger.error(f"{invalid_actions}, {error_msg}, {e}")
        return []

    def assert_invalid_actions(self, invalid_actions):
        assert isinstance(
            invalid_actions, list
        ), f"invalid_actions must be arrayed. {invalid_actions}({type(invalid_actions)})"
        for a in invalid_actions:
            assert self.action_space.check_val(a), f"The type of invalid_action is different. {a}({type(a)})"

    # ------------------------------------
    # property
    # ------------------------------------
    @property
    def name(self) -> str:
        return self.config.name

    @property
    def action_space(self) -> SpaceBase:
        return self._action_space

    @property
    def observation_space(self) -> SpaceBase:
        return self._observation_space

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
    def elapsed_time(self) -> float:
        return time.time() - self._t0

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
    def info(self) -> Info:
        return self._info

    @property
    def reward(self) -> float:
        """直前のrewardを返す"""
        return self.step_rewards[self.prev_player_index]

    @property
    def invalid_actions(self) -> List[EnvActionType]:
        """現プレイヤーのinvalid actionsを返す"""
        return self._invalid_actions_list[self.next_player_index]

    # invalid actions
    def get_invalid_actions(self, player_index: int = -1) -> List[EnvActionType]:
        if player_index == -1:
            player_index = self.next_player_index
        return self._invalid_actions_list[player_index]

    def get_valid_actions(self, player_index: int = -1) -> List[EnvActionType]:
        if isinstance(self.action_space, DiscreteSpace):
            invalid_actions = self.get_invalid_actions(player_index)
            return [a for a in range(self.action_space.n) if a not in invalid_actions]
        return []

    def add_invalid_actions(self, invalid_actions: List[EnvActionType], player_index: int) -> None:
        if self.config.enable_assertion:
            self.assert_invalid_actions(invalid_actions)
        elif self.config.enable_sanitize:
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

    def get_key_bind(self) -> Optional[KeyBindType]:
        return self.env.get_key_bind()

    def make_worker(self, name: str, worker_kwargs: dict = {}, enable_raise: bool = True):
        from srl.base.rl.registration import make_env_worker

        return make_env_worker(self, name, worker_kwargs, enable_raise)

    @property
    def unwrapped(self) -> Any:
        return self.env.unwrapped

    def end_episode(self):
        self._done = DoneTypes.TRUNCATED
        self._done_reason = "call end_episode"

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
        if self.config.enable_assertion:
            self.assert_done(self.is_start_episode)
            self.assert_state(state)
        if self.config.enable_sanitize:
            s = "'is_start_episode' in 'env.direct_step may' not be bool."
            self.is_start_episode = self.sanitize_done(self.is_start_episode, s)
            state = self.sanitize_state(state, "'state' in 'env.direct_step' may not be SpaceType.")

        if self.is_start_episode:
            self._reset_vals()
        self._step2(state, np.zeros((self.player_num,)), False, info)

    def decode_action(self, action: EnvActionType) -> Any:
        if self.config.enable_assertion:
            self.assert_action(action)
        elif self.config.enable_sanitize:
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
