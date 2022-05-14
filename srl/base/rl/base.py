import logging
import os
import pickle
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
from srl.base.define import (Action, EnvObservationType, Info, RLActionType,
                             RLObservationType)
from srl.base.env.base import EnvBase, SpaceBase
from srl.base.env.processor import Processor
from srl.base.env.spaces.box import BoxSpace

logger = logging.getLogger(__name__)


class RLConfig(ABC):
    def __init__(self) -> None:
        self.processors: List[Processor] = []
        self.override_env_observation_type: EnvObservationType = EnvObservationType.UNKNOWN
        self.action_division_num: int = 5

    @staticmethod
    @abstractmethod
    def getName() -> str:
        raise NotImplementedError()

    @property
    @abstractmethod
    def action_type(self) -> RLActionType:
        raise NotImplementedError()

    @property
    @abstractmethod
    def observation_type(self) -> RLObservationType:
        raise NotImplementedError()

    @abstractmethod
    def _set_config_by_env(
        self,
        env: EnvBase,
        env_action_space: SpaceBase,
        env_observation_space: SpaceBase,
        env_observation_type: EnvObservationType,
    ) -> None:
        raise NotImplementedError()

    def set_config_by_env(self, env: EnvBase) -> None:
        self._env_action_space = env.action_space
        observation_space = env.observation_space
        observation_type = env.observation_type

        # observation_typeの上書き
        if self.override_env_observation_type != EnvObservationType.UNKNOWN:
            observation_type = self.override_env_observation_type

        # processor
        for processor in self.processors:
            observation_space, observation_type = processor.change_observation_info(
                observation_space,
                observation_type,
                self.observation_type,
                env.get_original_env(),
            )
        self._env_observation_space = observation_space
        self._env_observation_type = observation_type

        # action division
        if isinstance(self._env_action_space, BoxSpace) and self.action_type == RLActionType.DISCRETE:
            self._env_action_space.set_division(self.action_division_num)

        self._set_config_by_env(env, self._env_action_space, observation_space, observation_type)
        self._is_set_config_by_env = True

    @property
    def is_set_config_by_env(self) -> bool:
        return hasattr(self, "_is_set_config_by_env")

    @property
    def env_action_space(self) -> SpaceBase:
        return self._env_action_space

    @property
    def env_observation_space(self) -> SpaceBase:
        return self._env_observation_space

    @property
    def env_observation_type(self) -> EnvObservationType:
        return self._env_observation_type

    def assert_params(self) -> None:
        pass  # do nothing

    def copy(self) -> "RLConfig":
        # TODO
        return pickle.loads(pickle.dumps(self))


class RLParameter(ABC):
    def __init__(self, config: RLConfig):
        self.config = config

    @abstractmethod
    def restore(self, data: Any) -> None:
        raise NotImplementedError()

    @abstractmethod
    def backup(self) -> Any:
        raise NotImplementedError()

    def save(self, path: str) -> None:
        logger.debug(f"save: {path}")
        try:
            with open(path, "wb") as f:
                pickle.dump(self.backup(), f)
        except Exception:
            if os.path.isfile(path):
                os.remove(path)
            raise

    def load(self, path: str) -> None:
        logger.debug(f"load: {path}")
        with open(path, "rb") as f:
            self.restore(pickle.load(f))

    def summary(self):
        pass  # do nothing


class RLRemoteMemory(ABC):
    def __init__(self, config: RLConfig):
        self.config = config

    @abstractmethod
    def length(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def restore(self, data: Any) -> None:
        raise NotImplementedError()

    @abstractmethod
    def backup(self) -> Any:
        raise NotImplementedError()

    def save(self, path: str) -> None:
        logger.debug(f"save: {path}")
        try:
            with open(path, "wb") as f:
                pickle.dump(self.backup(), f)
        except Exception:
            if os.path.isfile(path):
                os.remove(path)

    def load(self, path: str) -> None:
        logger.debug(f"load: {path}")
        with open(path, "rb") as f:
            self.restore(pickle.load(f))


class RLTrainer(ABC):
    def __init__(self, config: RLConfig, parameter: RLParameter, remote_memory: RLRemoteMemory):
        self.config = config
        self.parameter = parameter
        self.remote_memory = remote_memory

    @abstractmethod
    def get_train_count(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def train(self) -> Dict[str, Any]:
        raise NotImplementedError()


class RLWorker(ABC):
    def __init__(
        self,
        config: RLConfig,
        parameter: Optional[RLParameter] = None,
        remote_memory: Optional[RLRemoteMemory] = None,
        worker_id: int = 0,
    ):
        self.config = config
        self.parameter = parameter
        self.remote_memory = remote_memory
        self.worker_id = worker_id
        self._training = False
        self._distributed = False

    def observation_encode(self, state, env):
        state = np.asarray(state)
        for processor in self.config.processors:
            state = processor.process_observation(state, env.get_original_env())

        if self.config.observation_type == RLObservationType.DISCRETE:
            state = self.config.env_observation_space.observation_discrete_encode(state)
        elif self.config.observation_type == RLObservationType.CONTINUOUS:
            state = self.config.env_observation_space.observation_continuous_encode(state)
        return state

    def action_decode(self, action):
        if self.config.action_type == RLActionType.DISCRETE:
            action = self.config.env_action_space.action_discrete_decode(action)
        elif self.config.action_type == RLActionType.CONTINUOUS:
            action = self.config.env_action_space.action_continuous_decode(action)
        return action

    def set_training(self, training: bool, distributed: bool) -> None:
        self._training = training
        self._distributed = distributed

    @property
    def training(self) -> bool:
        return self._training

    @property
    def distributed(self) -> bool:
        return self._distributed

    @abstractmethod
    def _on_reset(
        self,
        state: np.ndarray,
        player_index: int,
        env: EnvBase,
    ) -> None:
        raise NotImplementedError()

    def on_reset(
        self,
        state: np.ndarray,
        player_index: int,
        env: EnvBase,
    ) -> None:
        state = self.observation_encode(state, env)
        self._on_reset(state, player_index, env)

    @abstractmethod
    def _policy(
        self,
        state: np.ndarray,
        player_index: int,
        env: EnvBase,
    ) -> Action:
        raise NotImplementedError()

    def policy(
        self,
        state: np.ndarray,
        player_index: int,
        env: EnvBase,
    ) -> Action:
        state = self.observation_encode(state, env)
        action = self._policy(state, player_index, env)
        return self.action_decode(action)

    @abstractmethod
    def _on_step(
        self,
        next_state: np.ndarray,
        reward: float,
        done: bool,
        player_index: int,
        env: EnvBase,
    ) -> Info:
        raise NotImplementedError()

    def on_step(
        self,
        next_state: np.ndarray,
        reward: float,
        done: bool,
        player_index: int,
        env: EnvBase,
    ) -> Info:
        next_state = self.observation_encode(next_state, env)
        return self._on_step(next_state, reward, done, player_index, env)

    @abstractmethod
    def render(self, env: EnvBase) -> None:
        raise NotImplementedError()
