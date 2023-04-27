import logging
import pickle
import time
import traceback
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np

import srl
from srl.base.define import EnvAction, EnvObservation, EnvObservationType, Info, KeyBindType, PlayRenderMode
from srl.base.env.config import EnvConfig
from srl.base.env.spaces.discrete import DiscreteSpace
from srl.base.env.spaces.space import SpaceBase
from srl.base.render import IRender, Render

logger = logging.getLogger(__name__)


class EnvBase(ABC, IRender):
    # --------------------------------
    # implement properties
    # --------------------------------

    # --- action
    @property
    @abstractmethod
    def action_space(self) -> SpaceBase:
        raise NotImplementedError()

    # --- observation
    @property
    @abstractmethod
    def observation_space(self) -> SpaceBase:
        raise NotImplementedError()

    @property
    @abstractmethod
    def observation_type(self) -> EnvObservationType:
        raise NotImplementedError()

    # --- properties
    @property
    @abstractmethod
    def max_episode_steps(self) -> int:
        raise NotImplementedError()

    @property
    @abstractmethod
    def player_num(self) -> int:
        raise NotImplementedError()

    # --- reward(option)
    @property
    def reward_info(self) -> dict:
        return {
            "range": None,
        }

    # --------------------------------
    # implement functions
    # --------------------------------
    @abstractmethod
    def reset(self) -> Tuple[EnvObservation, Info]:
        """reset

        Returns: init_state, info
        """
        raise NotImplementedError()

    @abstractmethod
    def step(self, action: EnvAction) -> Tuple[EnvObservation, List[float], bool, Info]:
        """step

        Args:
            action (EnvAction): player_index action

        Returns:(
            next_state,
            [
                player1 reward,
                player2 reward,
                ...
            ],
            done,
            info,
        )
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def next_player_index(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def backup(self) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def restore(self, data: Any) -> None:
        raise NotImplementedError()

    # --------------------------------
    # options
    # --------------------------------
    def close(self) -> None:
        pass

    def get_invalid_actions(self, player_index: int = -1) -> List[int]:
        return []

    def action_to_str(self, action: Union[str, EnvAction]) -> str:
        return str(action)

    def get_key_bind(self) -> KeyBindType:
        return None

    def make_worker(
        self,
        name: str,
        **kwargs,
    ) -> Optional["srl.base.rl.base.WorkerBase"]:
        return None

    def get_original_env(self) -> object:
        return self

    def set_seed(self, seed: Optional[int] = None) -> None:
        pass

    @property
    def render_interval(self) -> float:
        return 1000 / 60

    # --------------------------------
    # direct
    # --------------------------------
    def direct_step(self, *args, **kwargs) -> Tuple[bool, EnvObservation, int, Info]:
        """direct step
        外部で環境を動かしてpolicyだけ実行したい場合に実装します。
        これは学習で使う場合を想定していません。

        Returns:(
            is_start_episode,
            state,
            player_index,
            info,
        )
        """
        raise NotImplementedError()

    def decode_action(self, action: EnvAction) -> Any:
        raise NotImplementedError()

    @property
    def can_simulate_from_direct_step(self) -> bool:
        """
        direct_stepで実行した場合に、そのあとにstepでシミュレーション可能かどうかを返します。
        direct_step後にstepを機能させるには、direct_step内でstepが実行できるまでenv環境を復元する必要があります。
        主にMCTS等、シミュレーションが必要なアルゴリズムに影響があります。
        """
        raise NotImplementedError()

    # --------------------------------
    # utils
    # --------------------------------
    def copy(self):
        env = self.__class__()
        env.restore(self.backup())
        return env

    def get_valid_actions(self, player_index: int = -1) -> List[int]:
        if isinstance(self.action_space, DiscreteSpace):
            invalid_actions = self.get_invalid_actions(player_index)
            return [a for a in range(self.action_space.n) if a not in invalid_actions]
        else:
            return []


class EnvRun:
    def __init__(self, env: EnvBase, config: EnvConfig) -> None:
        self.env = env
        self.config = config
        self.init()

        self._render = Render(env, config.font_name, config.font_size)
        self._render_interval = self.env.render_interval
        self._t0 = 0

    def init(self):
        self._step_num = 0
        self._state = None
        self._episode_rewards = np.array(0)
        self._step_rewards = np.array(0)
        self._done = True
        self._done_reason = ""
        self._prev_player_index = 0
        self._invalid_actions_list = [[] for _ in range(self.env.player_num)]
        self._info = {}
        self._is_direct_step = False

    # --- with
    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self) -> None:
        logger.debug("env.close")
        try:
            self.env.close()
        except Exception:
            logger.error(traceback.format_exc())

    # ------------------------------------
    # change internal state
    # ------------------------------------
    def reset(
        self,
        render_mode: Union[str, PlayRenderMode] = "",
        render_interval: float = -1,
        seed: Optional[int] = None,
    ) -> None:
        logger.debug(f"env.reset({render_mode}, {render_interval}, {seed})")

        # --- seed
        self.env.set_seed(seed)

        # --- render
        self._render.cache_reset()
        if render_interval > 0:
            self._render_interval = render_interval
        self._render.reset(render_mode, self.render_interval)

        # --- env reset
        self._state, self._info = self.env.reset()
        if self.config.check_val:
            self._state = self.check_state(self._state, "state in env.reset may not be SpaceType.")
        self._reset()

    def _reset(self):
        self._step_num = 0
        self._done = False
        self._done_reason = ""
        self._prev_player_index = 0
        self._episode_rewards = np.zeros(self.player_num)
        self._step_rewards = np.zeros(self.player_num)
        self._invalid_actions_list = [self.env.get_invalid_actions(i) for i in range(self.env.player_num)]
        self._t0 = time.time()

    def step(
        self,
        action: EnvAction,
        skip_function: Optional[Callable[[], None]] = None,
    ) -> None:
        assert not self.done, "It is in the done state. Please execute reset()."
        if self._is_direct_step:
            assert self.env.can_simulate_from_direct_step, "env does not support 'step' after 'direct_step'."

        if self.config.check_action:
            action = self.check_action(action, "The format of 'action' entered in 'env.direct' was wrong.")
        self._prev_player_index = self.env.next_player_index
        state, rewards, done, info = self.env.step(action)
        if self.config.check_val:
            state = self.check_state(state, "'state' in 'env.step' may not be SpaceType.")
            rewards = self.check_rewards(rewards, "'rewards' in 'env.step' may not be List[float].")
            done = self.check_done(done, "'done' in 'env.reset may' not be bool.")

        self._render.cache_reset()
        step_rewards = np.array(rewards, dtype=np.float32)

        # skip frame の間は同じアクションを繰り返す
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

    def _step(self, state, rewards, done, info):
        self._state = state
        self._step_rewards = rewards
        self._done = done
        self._info = info

        invalid_actions = self.env.get_invalid_actions(self.next_player_index)
        self._invalid_actions_list[self.next_player_index] = invalid_actions
        self._step_num += 1
        self._episode_rewards += self.step_rewards

        # action check
        if not self.done and len(invalid_actions) > 0:
            assert len(invalid_actions) < self.action_space.get_action_discrete_info()

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
    def check_action(self, action, error_msg: str = "") -> EnvAction:
        try:
            if action in self.get_invalid_actions():
                logger.error(f"{action}({type(action)}), {error_msg}, invalid action {self.get_invalid_actions()}")
            return self.env.action_space.convert(action)
        except Exception as e:
            logger.error(f"{action}({type(action)}), {error_msg}, {e}")
        return self.env.action_space.get_default()

    def check_state(self, state, error_msg: str = "") -> EnvObservation:
        try:
            return self.env.observation_space.convert(state)
        except Exception as e:
            logger.error(f"{state}({type(state)}), {error_msg}, {e}")
        return self.env.observation_space.get_default()

    def check_rewards(self, rewards, error_msg: str = "") -> List[float]:
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

    def check_done(self, done, error_msg: str = "") -> bool:
        try:
            return bool(done)
        except Exception as e:
            logger.error(f"{done}({type(done)}), {error_msg}, {e}")
        return False

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
    def observation_type(self) -> EnvObservationType:
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
    def state(self) -> EnvObservation:
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
    def info(self) -> Info:
        return self._info

    @property
    def reward(self) -> float:
        """直前のrewardを返す"""
        return self.step_rewards[self.prev_player_index]

    # invalid actions
    def get_invalid_actions(self, player_index: int = -1) -> List[int]:
        if isinstance(self.action_space, DiscreteSpace):
            if player_index == -1:
                player_index = self.next_player_index
            return self._invalid_actions_list[player_index]
        else:
            return []

    def get_valid_actions(self, player_index: int = -1) -> List[int]:
        if isinstance(self.action_space, DiscreteSpace):
            invalid_actions = self.get_invalid_actions(player_index)
            return [a for a in range(self.action_space.n) if a not in invalid_actions]
        else:
            assert False, "not support"

    def add_invalid_actions(self, invalid_actions: List[int], player_index: int) -> None:
        self._invalid_actions_list[player_index] += invalid_actions
        self._invalid_actions_list[player_index] = list(set(self._invalid_actions_list[player_index]))

    # other functions
    def action_to_str(self, action: Union[str, EnvAction]) -> str:
        return self.env.action_to_str(action)

    def get_key_bind(self) -> KeyBindType:
        return self.env.get_key_bind()

    def make_worker(
        self,
        name: str,
        training: bool = False,
        distributed: bool = False,
        font_name: str = "",
        font_size: int = 12,
        enable_raise: bool = True,
        env_worker_kwargs: dict = {},
    ) -> "srl.base.rl.worker.WorkerRun":
        env_worker_kwargs = env_worker_kwargs.copy()
        env_worker_kwargs["training"] = training
        env_worker_kwargs["distributed"] = distributed
        worker = self.env.make_worker(name, **env_worker_kwargs)
        if worker is None:
            if enable_raise:
                raise ValueError(f"'{name}' worker is not found.")
            return None

        from srl.base.rl.worker import WorkerRun

        return WorkerRun(worker, font_name, font_size)

    def get_original_env(self) -> object:
        return self.env.get_original_env()

    @property
    def render_interval(self) -> float:
        return self._render_interval

    # ------------------------------------
    # render
    # ------------------------------------
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
            self._reset()
        self._step(state, [0.0] * self.player_num, False, info)

    def decode_action(self, action: EnvAction) -> Any:
        if self.config.check_action:
            action = self.check_action(action, "The format of 'action' entered in 'env.direct' was wrong.")
        return self.env.decode_action(action)

    # ------------------------------------
    # util functions
    # ------------------------------------
    def sample(self, player_index: int = -1) -> EnvAction:
        return self.action_space.sample(self.get_invalid_actions(player_index))

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
