import logging
import pickle
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import srl
from srl.base.define import EnvAction, EnvInvalidAction, EnvObservation, EnvObservationType, Info, RenderType
from srl.base.env.space import SpaceBase

logger = logging.getLogger(__name__)


@dataclass
class EnvConfig:
    name: str
    kwargs: Dict = field(default_factory=dict)

    # gym
    gym_prediction_by_simulation: bool = True

    def make_env(self) -> "srl.base.env.base.EnvRun":
        return srl.envs.make(self)


class EnvBase(ABC):
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
    def reset(self) -> Tuple[EnvObservation, List[int]]:
        """reset

        Returns: (
            init_state,
            next_player_indices,
        )
        """
        raise NotImplementedError()

    @abstractmethod
    def step(
        self,
        actions: List[EnvAction],
        player_indices: List[int],
    ) -> Tuple[EnvObservation, List[float], bool, List[int], Info]:
        """one step

        Args:
            actions (List[Optional[EnvAction]): [
                    player1 action,
                    player2 action,
                    ...
                ]
            player_indices (List[int]): stepで行動するプレイヤーのindex
            # player_indices と actions にあるindexは連動しており、
            # player_indices にないindexのactionは None である事

        Returns:
            (
            next_state,
            [
                player1 reward,
                player2 reward,
                ...
            ],
            done,
            next_player_indices,
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

    # option
    def close(self) -> None:
        pass

    # option
    def render_terminal(self, **kwargs) -> None:
        raise NotImplementedError()

    # option
    def render_gui(self, **kwargs) -> None:
        raise NotImplementedError()

    # option
    def render_rgb_array(self, **kwargs) -> np.ndarray:
        raise NotImplementedError()

    # option
    def get_invalid_actions(self, player_index: int) -> List[EnvInvalidAction]:
        return []

    # option
    def action_to_str(self, action: EnvAction) -> str:
        return str(action)

    # option
    def make_worker(self, name: str) -> Optional["srl.base.rl.base.WorkerBase"]:
        return None

    # option
    def get_original_env(self) -> object:
        return self

    def copy(self):
        env = self.__class__()
        env.restore(self.backup())
        return env

    # --------------------------------
    # other
    # --------------------------------
    def _set_env_run(self, env_run: "EnvRun"):
        self._env_run = env_run

    def set_run_step(
        self,
        state: EnvObservation,
        rewards: List[float],
        done: bool,
        next_player_indices: List[int],
        info: Info,
    ):
        self._env_run._state = state
        self._env_run._step_rewards = np.asarray(rewards)
        self._env_run._done = done
        self._env_run._next_player_indices = next_player_indices
        self._env_run._info = info


# 実装と実行で名前空間を分けるために別クラスに
class EnvRun:
    def __init__(self, env: EnvBase) -> None:
        self.env = env
        self.env._set_env_run(self)
        self.init()

    def init(self):
        self._step_num = 0
        self._state = None
        self._episode_rewards = np.array(0)
        self._step_rewards = np.array(0)
        self._done = True
        self._done_reason = ""
        self._next_player_indices = []
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

    # --------------------------------
    # implement properties
    # --------------------------------
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
        return self.env.max_episode_steps

    @property
    def player_num(self) -> int:
        return self.env.player_num

    # ------------------------------------
    # episode functions
    # ------------------------------------
    @property
    def state(self) -> EnvObservation:
        return self._state

    @property
    def next_player_indices(self) -> List[int]:
        return self._next_player_indices

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

    def reset(self, max_steps: int = -1, timeout: int = -1) -> None:
        logger.debug("env.reset")
        self._state, self._next_player_indices = self.env.reset()
        self._step_num = 0
        self._done = False
        self._done_reason = ""
        self._episode_rewards = np.zeros(self.player_num)
        self._step_rewards = np.zeros(self.player_num)
        self._invalid_actions_list = [self.env.get_invalid_actions(i) for i in range(self.env.player_num)]

        self.t0 = time.time()
        self.max_steps = max_steps
        self.timeout = timeout

    def step(self, actions: List[EnvAction], skip_frames: int = 0, skip_function=None) -> Info:
        logger.debug("env.step")
        assert (
            len(actions) == self.player_num
        ), "The number of actions does not match. (player: {self.player_num}, actions: {actions})"
        for idx in self.next_player_indices:
            assert actions[idx] is not None

        self._state, rewards, self._done, self._next_player_indices, self._info = self.env.step(
            actions, self.next_player_indices
        )
        self._step_rewards = np.asarray(rewards)

        # skip frame の間は同じアクションを繰り返す
        for _ in range(skip_frames):
            assert self.player_num == 1
            self._state, rewards, self._done, self._next_player_indices, self._info = self.env.step(
                actions, self.next_player_indices
            )
            self._step_rewards += np.asarray(rewards)
            if self.done:
                break

            if skip_function is not None:
                skip_function()

        self._invalid_actions_list = [self.env.get_invalid_actions(i) for i in range(self.env.player_num)]
        self._step_num += 1
        self._episode_rewards += self.step_rewards

        # done step
        if self.done:
            self._done_reason = "env"
        elif self.step_num > self.max_episode_steps:
            self._done = True
            self._done_reason = "env max steps"
        elif self.max_steps > 0 and self.step_num > self.max_steps:
            self._done = True
            self._done_reason = "episode max steps"
        elif self.timeout > 0 and time.time() - self.t0 > self.timeout:
            self._done = True
            self._done_reason = "timeout"

        return self.info

    def backup(self) -> Any:
        logger.debug("env.backup")
        d = [
            self.env.backup(),
            self.step_num,
            self.episode_rewards,
            self.state,
            self.step_rewards,
            self.done,
            self.done_reason,
            self.next_player_indices,
            self._invalid_actions_list,
            self.info,
        ]
        return pickle.dumps(d)

    def restore(self, data: Any) -> None:
        logger.debug("env.restore")
        d = pickle.loads(data)
        self.env.restore(d[0])
        self._step_num = d[1]
        self._episode_rewards = d[2]
        self._state = d[3]
        self._step_rewards = d[4]
        self._done = d[5]
        self._done_reason = d[6]
        self._next_player_indices = d[7]
        self._invalid_actions_list = d[8]
        self._info = d[9]

    def render(
        self,
        mode: Union[str, RenderType] = RenderType.Terminal,
        is_except: bool = False,
        **kwargs,
    ) -> Any:

        logger.debug(f"env.render({mode})")
        if isinstance(mode, str):
            for t in RenderType:
                if t.value == mode:
                    mode = t
                    break
            else:
                mode = RenderType.NONE

        try:
            if mode == RenderType.Terminal:
                return self.env.render_terminal(**kwargs)
            elif mode == RenderType.GUI:
                return self.env.render_gui(**kwargs)
            elif mode == RenderType.RGB_Array:
                return self.env.render_rgb_array(**kwargs)
        except NotImplementedError:
            # logger.info(f"render NotImplementedError({mode})")
            if is_except:
                raise

    def get_invalid_actions(self, player_index: int) -> List[EnvInvalidAction]:
        return self._invalid_actions_list[player_index]

    def action_to_str(self, action: EnvAction) -> str:
        return self.env.action_to_str(action)

    def make_worker(self, name: str) -> Optional["srl.base.rl.base.WorkerRun"]:
        worker = self.env.make_worker(name)
        if worker is None:
            return None

        from srl.base.rl.base import WorkerRun

        return WorkerRun(worker)

    def get_original_env(self) -> object:
        return self.env.get_original_env()

    # ------------------------------------
    # util functions
    # ------------------------------------
    def samples(self) -> List[EnvAction]:
        return [self.action_space.sample(self.get_invalid_actions(i)) for i in range(self.player_num)]

    def sample(self, player_index: int) -> EnvAction:
        return self.action_space.sample(self.get_invalid_actions(player_index))

    def copy(self):
        org_env = self.env.__class__()
        env = self.__class__(org_env)
        env.restore(self.backup())
        return env
