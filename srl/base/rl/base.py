import logging
import pickle
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from srl.base.define import RLActionType, RLObservationType

logger = logging.getLogger(__name__)


class RLConfig(ABC):
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
    def set_config_by_env(self, env: "srl.base.env.env_for_rl.EnvForRL") -> None:
        raise NotImplementedError()

    @property
    def is_set_config_by_env(self) -> bool:
        return hasattr(self, "_is_set_config_by_env")

    def assert_params(self) -> None:
        pass


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
        with open(path, "wb") as f:
            pickle.dump(self.backup(), f)

    def load(self, path: str) -> None:
        logger.debug(f"load: {path}")
        with open(path, "rb") as f:
            self.restore(pickle.load(f))

    def summary(self):
        pass

    # 行動価値(option)
    def get_action_values(
        self, state: np.ndarray, invalid_actions: List[int], env: "srl.base.env.env_for_rl.EnvForRL"
    ) -> List[float]:
        return []

    # 状態価値(option)
    def get_state_value(
        self, state: np.ndarray, invalid_actions: List[int], env: "srl.base.env.env_for_rl.EnvForRL"
    ) -> float:
        return 0


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
        with open(path, "wb") as f:
            pickle.dump(self.backup(), f)

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
        self.training = False

    def set_training(self, training: bool) -> None:
        self.training = training

    @abstractmethod
    def on_reset(
        self,
        state: np.ndarray,
        invalid_actions: List[int],
        env: "srl.base.env.env_for_rl.EnvForRL",
        start_player_indexes: List[int],
    ) -> None:
        raise NotImplementedError()

    @abstractmethod
    def policy(
        self,
        state: np.ndarray,
        invalid_actions: List[int],
        env: "srl.base.env.env_for_rl.EnvForRL",
        player_indexes: List[int],
    ) -> Tuple[Any, Any]:  # (env_action, agent_action)
        raise NotImplementedError()

    @abstractmethod
    def on_step(
        self,
        state: np.ndarray,
        action: Any,
        next_state: np.ndarray,
        reward: float,
        done: bool,
        invalid_actions: List[int],
        next_invalid_actions: List[int],
        env: "srl.base.env.env_for_rl.EnvForRL",
    ) -> Dict[str, Union[float, int]]:  # info
        raise NotImplementedError()

    @abstractmethod
    def render(
        self,
        state: np.ndarray,
        invalid_actions: List[int],
        env: "srl.base.env.env_for_rl.EnvForRL",
    ) -> None:
        raise NotImplementedError()


if __name__ == "__main__":
    pass
