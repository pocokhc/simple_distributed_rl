import logging
import pickle
import time
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import srl
from srl.base.define import EnvAction, EnvObservation, EnvObservationType, Info, KeyBindType, PlayRenderMode
from srl.base.env.config import EnvConfig
from srl.base.env.space import SpaceBase
from srl.base.env.spaces.discrete import DiscreteSpace
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

    # --------------------------------
    # implement functions
    # --------------------------------
    @abstractmethod
    def reset(self) -> Tuple[EnvObservation, int, Info]:
        """reset

        Returns: (
            init_state,
            next_player_index,
            info,
        )
        """
        raise NotImplementedError()

    @abstractmethod
    def step(
        self,
        action: EnvAction,
        player_index: int,
    ) -> Tuple[EnvObservation, List[float], bool, int, Info]:
        """step

        Args:
            action (EnvAction): player_index action
            player_index (int): stepで行動するプレイヤーのindex

        Returns:(
            next_state,
            [
                player1 reward,
                player2 reward,
                ...
            ],
            done,
            next_player_index,
            info,
        )
        """

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

    def get_invalid_actions(self, player_index: int) -> List[int]:
        return []

    def action_to_str(self, action: Union[str, EnvAction]) -> str:
        return str(action)

    def get_key_bind(self) -> KeyBindType:
        return None

    def make_worker(self, name: str) -> Optional["srl.base.rl.base.WorkerBase"]:
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
    def direct_reset(self, *args, **kwargs) -> Tuple[EnvObservation, int, Info]:
        raise NotImplementedError()

    def direct_step(self, *args, **kwargs) -> Tuple[EnvObservation, List[float], bool, int, Info]:
        raise NotImplementedError()

    # --------------------------------
    # utils
    # --------------------------------
    def copy(self):
        env = self.__class__()
        env.restore(self.backup())
        return env

    def get_valid_actions(self, player_index: int) -> List[int]:
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

        self.t0 = 0

    def init(self):
        self._step_num = 0
        self._state = None
        self._episode_rewards = np.array(0)
        self._step_rewards = np.array(0)
        self._done = True
        self._done_reason = ""
        self._next_player_index = 0
        self._invalid_actions_list = [[] for _ in range(self.env.player_num)]
        self._info = None

    # --- with
    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self) -> None:
        logger.debug("env.close")
        self.env.close()

    # ------------------------------------
    # change internal state
    # ------------------------------------
    def reset(self) -> None:
        logger.debug("env.reset")

        self._state, self._next_player_index, self._info = self.env.reset()
        self._step_num = 0
        self._done = False
        self._done_reason = ""
        self._episode_rewards = np.zeros(self.player_num)
        self._step_rewards = np.zeros(self.player_num)
        self._invalid_actions_list = [self.env.get_invalid_actions(i) for i in range(self.env.player_num)]
        self._render.cache_reset()

        self.t0 = time.time()

    def step(self, action: EnvAction, skip_function=None) -> None:
        assert not self.done, "It is in the done state. Please execute reset ()."
        logger.debug("env.step")

        self._state, rewards, self._done, self._next_player_index, self._info = self.env.step(
            action, self.next_player_index
        )
        self._step_rewards = np.asarray(rewards, dtype=np.float32)
        self._render.cache_reset()

        # skip frame の間は同じアクションを繰り返す
        for _ in range(self.config.frameskip):
            assert self.player_num == 1
            self._state, rewards, self._done, self._next_player_index, self._info = self.env.step(
                action, self.next_player_index
            )
            self._step_rewards += np.asarray(rewards, dtype=np.float32)
            self._render.cache_reset()
            if self.done:
                break

            if skip_function is not None:
                skip_function()

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
        elif self.config.episode_timeout > 0 and time.time() - self.t0 > self.config.episode_timeout:
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
            self.next_player_index,
            self._invalid_actions_list,
            self.info,
            self.t0,
        ]
        if include_env:
            d.append(self.env.backup())
        return pickle.dumps(d)

    def restore(self, data: Any) -> None:
        logger.debug("env.restore")
        d = pickle.loads(data)
        self._step_num = d[0]
        self._episode_rewards = d[1]
        self._state = d[2]
        self._step_rewards = d[3]
        self._done = d[4]
        self._done_reason = d[5]
        self._next_player_index = d[6]
        self._invalid_actions_list = d[7]
        self._info = d[8]
        self.t0 = d[9]
        if len(d) == 11:
            self.env.restore(d[10])

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

    # state properties
    @property
    def state(self) -> EnvObservation:
        return self._state

    @property
    def next_player_index(self) -> int:
        return self._next_player_index

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
    def info(self) -> Optional[Info]:
        return self._info

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

    def make_worker(self, name: str) -> Optional["srl.base.rl.base.WorkerRun"]:
        worker = self.env.make_worker(name)
        if worker is None:
            return None

        from srl.base.rl.worker import WorkerRun

        return WorkerRun(worker)

    def get_original_env(self) -> object:
        return self.env.get_original_env()

    def set_seed(self, seed: Optional[int] = None) -> None:
        self.env.set_seed(seed)

    @property
    def render_interval(self) -> float:
        return self._render_interval

    # ------------------------------------
    # render
    # ------------------------------------
    def set_render_mode(self, mode: Union[str, PlayRenderMode], interval: float = -1) -> None:
        if interval > 0:
            self._render_interval = interval
        self._render.reset(mode, self._render_interval)

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
    def direct_reset(self, *args, **kwargs) -> None:
        logger.debug("env.direct_reset")

        self._state, self._next_player_index, self._info = self.env.direct_reset(*args, **kwargs)
        self._step_num = 0
        self._done = False
        self._done_reason = ""
        self._episode_rewards = np.zeros(self.player_num)
        self._step_rewards = np.zeros(self.player_num)
        self._invalid_actions_list = [self.env.get_invalid_actions(i) for i in range(self.env.player_num)]

    def direct_step(self, *args, **kwargs) -> None:
        logger.debug("env.direct_step")

        self._state, rewards, self._done, self._next_player_index, self._info = self.env.direct_step(*args, **kwargs)
        self._step_rewards = np.asarray(rewards)

        self._invalid_actions_list = [self.env.get_invalid_actions(i) for i in range(self.env.player_num)]
        self._step_num += 1
        self._episode_rewards += self.step_rewards

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
