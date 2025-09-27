import logging
import random
import time
import traceback
from typing import TYPE_CHECKING, Any, Callable, Generic, List, Optional, Tuple, Union, cast

import numpy as np

from srl.base.context import RunContext
from srl.base.define import DoneTypes, EnvObservationType, KeyBindType, RenderModeType
from srl.base.env.base import EnvBase
from srl.base.env.config import EnvConfig
from srl.base.env.registration import make_base
from srl.base.exception import SRLError
from srl.base.info import Info
from srl.base.render import Render
from srl.base.spaces.discrete import DiscreteSpace
from srl.base.spaces.space import SpaceBase, TActSpace, TActType, TObsSpace, TObsType

if TYPE_CHECKING:
    from srl.base.define import RLActionType, RLObservationType
    from srl.base.rl.config import RLConfig


logger = logging.getLogger(__name__)


class EnvRun(Generic[TActSpace, TActType, TObsSpace, TObsType]):
    def __init__(self, config: EnvConfig, env: Optional[EnvBase[TActSpace, TActType, TObsSpace, TObsType]] = None) -> None:
        # restore/backup用に状態は意識して管理
        self.env: EnvBase[TActSpace, TActType, TObsSpace, TObsType] = None  # type: ignore
        self.config = config
        if env is not None:
            self.env = env
            self.env.set_env_run(self)
        else:
            self.remake()

        # --- processor
        self._processors = [c.copy() for c in self.config.processors]
        self._processors_action: List[Tuple[Any, SpaceBase, SpaceBase]] = []
        self._processors_invalid_actions: List[Tuple[Any, SpaceBase, SpaceBase]] = []
        self._processors_observation: List[Tuple[Any, SpaceBase, SpaceBase]] = []
        self._processors_reset: List[Any] = [c for c in self._processors if hasattr(c, "remap_reset")]
        self._processors_step: List[Any] = [c for c in self._processors if hasattr(c, "remap_step")]

        # --- space
        if self.config.override_action_space is None:
            act_space = self.env.action_space
        else:
            act_space = self.config.override_action_space
        if self.config.override_observation_space is None:
            obs_space = self.env.observation_space
        else:
            obs_space = self.config.override_observation_space
        for p in self._processors:
            # action
            new_space = p.remap_action_space(act_space, env_run=self)
            if new_space is not None:
                if hasattr(p, "remap_action"):
                    self._processors_action.append((p, act_space, new_space))
                if hasattr(p, "remap_invalid_actions"):
                    self._processors_invalid_actions.append((p, act_space, new_space))
                act_space = new_space

            # obs
            new_space = p.remap_observation_space(obs_space, env_run=self)
            if new_space is not None:
                if hasattr(p, "remap_observation"):
                    self._processors_observation.append((p, obs_space, new_space))
                obs_space = new_space
        self._remapped_act_space = act_space
        self._remapped_obs_space = obs_space

        # --- init val
        render_interval = 1000 / 60
        if self.config.render_interval > 0:
            render_interval = self.config.render_interval
        else:
            render_interval = self.env.render_interval
        render_interval *= self.config.frameskip + 1
        self._render = Render(self.env, render_interval)
        self._reset_vals()
        self.context: RunContext = RunContext(self.config)
        self.env.next_player = 0
        self._done = DoneTypes.RESET
        self._is_direct_step = False
        self._is_setup = False

    def _reset_vals(self):
        self._step_num: int = 0
        self._state = self._remapped_obs_space.get_default()
        self._action: TActType = self._remapped_act_space.get_default()
        self._done = DoneTypes.NONE
        self.env.done_reason = ""
        self._prev_player: int = 0
        self._episode_rewards = [0.0 for _ in range(self.env.player_num)]
        self._step_rewards = [0.0 for _ in range(self.env.player_num)]
        self._invalid_actions_list: List[List[TActType]] = [[] for _ in range(self.env.player_num)]
        self._t0 = time.time()
        self.env.info = Info()

    def remake(self):
        logger.debug("remake")
        self.close()
        self.env = make_base(self.config, self)

    def backup(self) -> Any:
        # - spaceは状態を持たない
        # - renderはcacheクリア
        logger.debug(f"backup: step={self._step_num}, done={self._done}({self.env.done_reason}) reward={self._step_rewards}")
        d = [
            # reset_vals
            self._step_num,
            self._remapped_obs_space.copy_value(self._state),
            self._remapped_act_space.copy_value(self._action),
            self._done,
            self.env.done_reason,
            self._prev_player,
            self.env.next_player,
            self._episode_rewards[:],
            self._step_rewards[:],
            [s[:] for s in self._invalid_actions_list],
            self._t0,
            self.env.info.copy(),
            # init val
            self._is_direct_step,
            self._is_setup,
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
        self._action = dat[2]
        self._done = dat[3]
        self.env.done_reason = dat[4]
        self._prev_player = dat[5]
        self.env.next_player = dat[6]
        self._episode_rewards = dat[7][:]
        self._step_rewards = dat[8][:]
        self._invalid_actions_list = dat[9][:]
        self._t0 = dat[10]
        self.env.info = dat[11].copy()
        logger.debug(f"restore: step={self._step_num}, done={self._done}({self.env.done_reason}) reward={self._step_rewards}")
        # init val
        self._is_direct_step = dat[12]
        self._is_setup = dat[13]
        # processor
        [p.restore(dat[14][i]) for i, p in enumerate(self._processors)]
        # env
        self.env.restore(dat[15])

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
        if self.env is not None:
            try:
                logger.debug("close")
                self.env.close()
            except Exception:
                logger.error(traceback.format_exc())

    # ------------------------------------
    # run functions
    # ------------------------------------
    def setup(
        self,
        context: Optional[RunContext] = None,
        render_mode: RenderModeType = "",
    ):
        logger.debug(f"setup: {render_mode=}")
        if context is not None:
            self.context = context
        if render_mode == "":
            render_mode = self.context.env_render_mode

        # --- reset前の状態を設定
        self._done: DoneTypes = DoneTypes.RESET
        self.env.done_reason = ""

        # --- render
        self._render.set_render_mode(render_mode)

        # --- processor
        [p.setup(env_run=self) for p in self._processors]

        # --- env
        kwargs = self.context.to_dict()
        kwargs["render_mode"] = render_mode
        self.env.setup(**kwargs)
        self._is_setup = True

    def teardown(self, **kwargs) -> None:
        logger.debug("teardown")
        self.env.teardown(**kwargs)
        self._is_setup = False

    def reset(self, *, seed: Optional[int] = None, **kwargs) -> None:
        logger.debug(f"reset: {seed=}")
        if not self._is_setup:
            raise SRLError("Cannot call env.reset() before calling env.setup()")

        # --- env reset
        self._reset_vals()
        self._state = self.env.reset(seed=seed, **kwargs)

        if self.config.random_noop_max > 0:
            for _ in range(random.randint(0, self.config.random_noop_max)):
                self._state, rewards, terminated, truncated = self.env.step(self.env.action_space.get_default())
                assert (not terminated) and (not truncated), "Terminated during noop step."

        self._invalid_actions_list = [self.env.get_invalid_actions(i) for i in range(self.env.player_num)]
        self._prev_player = self.env.next_player

        # --- processor
        for p in self._processors_observation:
            self._state = p[0].remap_observation(self._state, p[1], [2], env_run=self)
        for p in self._processors_reset:
            p.remap_reset(env_run=self)

        # --- check
        if self.config.enable_assertion:
            assert self.observation_space.check_val(self._state)
            [self.assert_invalid_actions(a) for a in self._invalid_actions_list]
        elif self.config.enable_sanitize:
            self._state = self.sanitize_state(self._state, "state in env.reset may not be SpaceType.")
            self._invalid_actions_list = [self.sanitize_invalid_actions(a, "invalid_actions in env.reset may not be SpaceType.") for a in self._invalid_actions_list]

        # render
        if self._render.rendering:
            self._render.cache_render()

    def step(
        self,
        action: TActType,
        frameskip: int = 0,
        frameskip_function: Optional[Callable[[], None]] = None,
    ) -> None:
        logger.debug(f"step: {action=}, {frameskip=}")
        if self._done != DoneTypes.NONE:
            raise SRLError(f"It is in the done state. Please execute reset(). ({self._done})")
        if self._is_direct_step and (not self.env.can_simulate_from_direct_step):
            raise SRLError("env does not support 'step' after 'direct_step'.")

        # --- action processor
        for p in self._processors_action:
            action = p[0].remap_action(action, p[1], [2], env_run=self)

        # --- env step
        self._prev_player = self.env.next_player
        if self.config.enable_assertion:
            self.assert_action(action)
        elif self.config.enable_sanitize:
            action = self.sanitize_action(action, "The format of 'action' entered in 'env.step' was wrong.")
        self._action = action
        state, rewards, done = self._step1(action)
        step_rewards = rewards

        # --- skip frame
        for _ in range(self.config.frameskip + frameskip):
            if done != DoneTypes.NONE:
                break
            state, rewards, done = self._step1(action)
            step_rewards = [step_rewards[i] + rewards[i] for i in range(self.env.player_num)]

            if frameskip_function is not None:
                self._render.cache_reset()
                frameskip_function()

        self._step2(state, step_rewards, done)
        self._render.cache_render()

    def _step1(self, action) -> Tuple[TObsType, List[float], DoneTypes]:
        """actionを元にenv.stepを実行"""
        f_except = None
        try:
            state, _rewards, terminated, truncated = self.env.step(action)
            rewards: List[float] = _rewards if isinstance(_rewards, list) else [_rewards]
            for p in self._processors_observation:
                state = p[0].remap_observation(state, p[1], [2], env_run=self)
            for p in self._processors_step:
                rewards, terminated, truncated = p.remap_step(rewards, terminated, truncated, env_run=self)

        except Exception:
            f_except = traceback.format_exc()

        if self.config.enable_assertion:
            assert f_except is None, f_except
            self.assert_state(state)
            self.assert_rewards(rewards)
            self.assert_bool(terminated)
            self.assert_bool(truncated)
        elif f_except is not None:
            self.remake()  # 例外が出たら作り直す
            done = DoneTypes.TRUNCATED
            self.env.done_reason = "step exception"
            s = "An exception occurred in env.step. Recreate.\n" + f_except
            print(s)
            logger.warning(s)
            return cast(TObsType, state), rewards, done
        elif self.config.enable_sanitize:
            state = self.sanitize_state(state, "'state' in 'env.step' may not be SpaceType.")
            rewards = self.sanitize_rewards(rewards, "'rewards' in 'env.step' may not be List[float].")
            terminated = self.sanitize_bool(terminated)
            truncated = self.sanitize_bool(truncated)
        if truncated:
            done = DoneTypes.TRUNCATED
        elif terminated:
            done = DoneTypes.TERMINATED
        else:
            done = DoneTypes.NONE
        return cast(TObsType, state), rewards, done

    def _step2(self, state: TObsType, rewards: List[float], done: DoneTypes):
        """env.stepの結果からの後処理"""
        self._state = state
        self._step_rewards = rewards
        self._done = done

        invalid_actions = self.env.get_invalid_actions(self.env.next_player)
        for p in self._processors_invalid_actions:
            invalid_actions = p[0].remap_invalid_actions(invalid_actions, p[1], [2], env_run=self)

        if self.config.enable_assertion:
            self.assert_invalid_actions(invalid_actions)
        elif self.config.enable_sanitize:
            invalid_actions = self.sanitize_invalid_actions(invalid_actions, "invalid_actions in env.step may not be SpaceType.")
        self._invalid_actions_list[self.env.next_player] = invalid_actions
        self._step_num += 1
        self._episode_rewards = [self._episode_rewards[i] + self._step_rewards[i] for i in range(self.env.player_num)]

        # action check
        if (self._done == DoneTypes.NONE) and len(invalid_actions) > 0:
            # 有効なアクションがある事
            if isinstance(self.action_space, DiscreteSpace):
                assert len(invalid_actions) < self.action_space.n

        # done step
        if self._done == DoneTypes.NONE:
            if self._step_num > self.max_episode_steps:
                self._done = DoneTypes.TRUNCATED
                self.env.done_reason = "episode step over"
            elif self.config.episode_timeout > 0 and time.time() - self._t0 > self.config.episode_timeout:
                self._done = DoneTypes.TRUNCATED
                self.env.done_reason = "timeout"

    # ------------------------------------
    # check
    # ------------------------------------
    def sanitize_action(self, action: Any, error_msg: str = "") -> TActType:
        try:
            for inv_act in self.get_invalid_actions():
                if action == inv_act:
                    logger.error(f"{action}({type(action)}), {error_msg}, invalid action {self.get_invalid_actions()}")
                    break
            return self._remapped_act_space.sanitize(action)
        except Exception as e:
            logger.error(f"{action}({type(action)}), {error_msg}, {e}")
        return self._remapped_act_space.get_default()

    def assert_action(self, action: Any):
        assert self._remapped_act_space.check_val(action), f"The type of action is different. {action}({type(action)})"

    def sanitize_state(self, state: Any, error_msg: str = "") -> TObsType:
        try:
            return self._remapped_obs_space.sanitize(state)
        except Exception as e:
            logger.error(f"{state}({type(state)}), {error_msg}, {e}")
        return self._remapped_obs_space.get_default()

    def assert_state(self, state: Any):
        assert self._remapped_obs_space.check_val(state), f"The type of state is different. {state}({type(state)})"

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
        return [0.0 for _ in range(self.env.player_num)]

    def assert_rewards(self, rewards: Any):
        assert isinstance(rewards, list), f"Rewards must be arrayed. {rewards}({type(rewards)})"
        assert len(rewards) == self.env.player_num, f"Array sizes are different. {len(rewards)} != {self.env.player_num}, {rewards}({type(rewards)})"
        for r in rewards:
            assert isinstance(r, float), f"The type of reward is different. {r}({type(r)}), {rewards}"

    def sanitize_bool(self, val: Any) -> bool:
        try:
            val = bool(val)
        except Exception as e:
            logger.error(f"{val}({type(val)}), {e}")
            val = False
        return val

    def assert_bool(self, val: Any):
        assert isinstance(val, bool), f"The type of bool is different. {val}({type(val)})"

    def sanitize_invalid_actions(self, invalid_actions: Any, error_msg: str = "") -> List[TActType]:
        try:
            for j in range(len(invalid_actions)):
                invalid_actions[j] = self.action_space.sanitize(invalid_actions[j])
            return invalid_actions
        except Exception as e:
            logger.error(f"{invalid_actions}, {error_msg}, {e}")
        return []

    def assert_invalid_actions(self, invalid_actions: Any):
        assert isinstance(invalid_actions, list), f"invalid_actions must be arrayed. {invalid_actions}({type(invalid_actions)})"
        for a in invalid_actions:
            assert self.action_space.check_val(a), f"The type of invalid_action is different. {a}({type(a)})"

    # ------------------------------------
    # property
    # ------------------------------------
    @property
    def id(self) -> str:
        return self.config.id

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def action_space(self) -> SpaceBase:
        return self._remapped_act_space

    @property
    def observation_space(self) -> SpaceBase:
        return self._remapped_obs_space

    @property
    def max_episode_steps(self) -> int:
        if self.config.max_episode_steps > 0:
            return self.config.max_episode_steps
        return self.env.max_episode_steps

    @property
    def player_num(self) -> int:
        return self.env.player_num

    @property
    def reward_range(self) -> Tuple[float, float]:
        return self.env.reward_range

    @property
    def reward_baseline(self):
        return self.env.reward_baseline

    # state properties
    @property
    def state(self) -> TObsType:
        return self._state

    @property
    def prev_player(self) -> int:
        return self._prev_player

    @property
    def next_player(self) -> int:
        return self.env.next_player

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
        return self.env.done_reason

    @property
    def episode_rewards(self) -> List[float]:
        return self._episode_rewards

    @property
    def rewards(self) -> List[float]:
        return self._step_rewards

    @property
    def info(self) -> Info:
        return self.env.info

    @property
    def action(self) -> TActType:
        """直前のactionを返す"""
        return self._action

    @property
    def reward(self) -> float:
        """直前のrewardを返す"""
        return self._step_rewards[self._prev_player]

    @property
    def invalid_actions(self) -> List[TActType]:
        """次のプレイヤーのinvalid actionsを返す"""
        return self._invalid_actions_list[self.env.next_player]

    # invalid actions
    def get_invalid_actions(self, player_index: int = -1) -> List[TActType]:
        if player_index == -1:
            player_index = self.env.next_player
        return self._invalid_actions_list[player_index]

    def get_valid_actions(self, player_index: int = -1) -> List[TActType]:
        if isinstance(self.action_space, DiscreteSpace):
            invalid_actions = self.get_invalid_actions(player_index)
            val_acts = [a for a in range(self.action_space.n) if a not in invalid_actions]
            return cast(List[TActType], val_acts)
        return []

    def add_invalid_actions(self, invalid_actions: List[TActType], player_index: int) -> None:
        if self.config.enable_assertion:
            self.assert_invalid_actions(invalid_actions)
        elif self.config.enable_sanitize:
            invalid_actions = self.sanitize_invalid_actions(invalid_actions, "invalid_actions in 'env.add_invalid_actions' may not be SpaceType.")

        if isinstance(self.action_space, DiscreteSpace):
            self._invalid_actions_list[player_index] += invalid_actions
            self._invalid_actions_list[player_index] = list(set(self._invalid_actions_list[player_index]))
        else:
            assert False, "not support"

    # other functions
    def action_to_str(self, action: Union[str, TActType]) -> str:
        return self.env.action_to_str(action)

    def get_key_bind(self) -> Optional[KeyBindType]:
        return self.env.get_key_bind()

    def make_worker(self, name: str, worker_kwargs: dict = {}, enable_raise: bool = True):
        from srl.base.rl.registration import make_env_worker

        return make_env_worker(self, name, worker_kwargs, enable_raise)

    @property
    def unwrapped(self) -> Any:
        return self.env.unwrapped

    def abort_episode(self):
        self._done = DoneTypes.ABORT
        self.env.done_reason = "call abort_episode"

    # ------------------------------------
    # render
    # ------------------------------------
    def set_render_options(
        self,
        interval: float = -1,  # ms
        scale: float = 1.0,
        font_name: str = "",
        font_size: int = 18,
    ):
        self._render.set_render_options(interval, scale, font_name, font_size)

    def get_render_interval(self, interval: float = -1):  # ms
        if interval < 0:
            interval = self._render.interval
        if interval < 1:
            interval = 1
        if interval > 2000:
            interval = 2000
        return interval

    def render(self, **kwargs):
        logger.debug("render")
        return self._render.render(**kwargs)

    def render_terminal_text(self, **kwargs) -> str:
        return self._render.get_cached_terminal_text(**kwargs)

    def render_terminal_text_to_image(self, **kwargs):
        return self._render.get_cached_terminal_text_to_image(**kwargs)

    def render_rgb_array(self, **kwargs) -> Optional[np.ndarray]:
        return self._render.get_cached_rgb_array(**kwargs)

    # ------------------------------------
    # simulation
    # ------------------------------------
    def step_from_rl(self, action: "RLActionType", rl_config: "RLConfig", **step_kwargs) -> "RLObservationType":
        env_action = cast(TActType, rl_config.action_decode(action))
        self.step(env_action, **step_kwargs)
        rl_state = rl_config.state_encode_one_step(cast(EnvObservationType, self.state), self)
        return rl_state

    # ------------------------------------
    # direct
    # ------------------------------------
    def direct_step(self, *args, **kwargs) -> Tuple[bool, bool]:
        logger.debug("direct_step")
        self._is_direct_step = True
        self._prev_player = self.env.next_player
        is_start_episode, state, is_end_episode = self.env.direct_step(*args, **kwargs)
        if self.config.enable_assertion:
            self.assert_bool(is_start_episode)
            self.assert_state(state)
        if self.config.enable_sanitize:
            is_start_episode = self.sanitize_bool(is_start_episode)
            state = self.sanitize_state(state, "'state' in 'env.direct_step' may not be SpaceType.")
        if is_start_episode:
            self._reset_vals()

        if is_end_episode:
            done = DoneTypes.TERMINATED
            self.env.done_reason = "direct_step end episode"
        else:
            done = DoneTypes.NONE
        self._step2(state, [0] * self.env.player_num, done)
        return is_start_episode, is_end_episode

    def decode_action(self, action: TActType) -> Any:
        if self.config.enable_assertion:
            self.assert_action(action)
        elif self.config.enable_sanitize:
            action = self.sanitize_action(action, "The format of 'action' entered in 'env.decode_action' was wrong.")
        return self.env.decode_action(action)

    # ------------------------------------
    # util functions
    # ------------------------------------
    def sample_action(self, player_index: int = -1) -> TActType:
        return self.action_space.sample(self.get_invalid_actions(player_index))

    def sample_observation(self, player_index: int = -1) -> TActType:
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
