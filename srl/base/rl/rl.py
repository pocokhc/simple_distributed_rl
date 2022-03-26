import os
import pickle
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from srl.base.rl.config import RLConfig
from srl.base.rl.env_for_rl import EnvForRL


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
        with open(path, "wb") as f:
            pickle.dump(self.backup(), f)

    def load(self, path: str) -> None:
        if not os.path.isfile(path):
            return
        with open(path, "rb") as f:
            self.restore(pickle.load(f))

    def summary(self):
        pass


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
        with open(path, "wb") as f:
            pickle.dump(self.backup(), f)

    def load(self, path: str) -> None:
        if not os.path.isfile(path):
            return
        with open(path, "rb") as f:
            self.restore(pickle.load(f))


class RLTrainer(ABC):
    def __init__(self, config: RLConfig, parameter: RLParameter, memory: RLRemoteMemory):
        self.config = config
        self.parameter = parameter
        self.memory = memory

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
        memory: Optional[RLRemoteMemory] = None,
        worker_id: int = 0,
    ):
        self.config = config
        self.parameter = parameter
        self.memory = memory
        self.worker_id = worker_id
        self.training = False

    def set_training(self, training: bool) -> None:
        self.training = training

    @abstractmethod
    def on_reset(self, state: Any, valid_actions: Optional[List[int]], env: EnvForRL) -> None:
        raise NotImplementedError()

    @abstractmethod
    def policy(
        self, state: Any, valid_actions: Optional[List[int]], env: EnvForRL
    ) -> Tuple[Any, Any]:  # (env_action, agent_action)
        raise NotImplementedError()

    @abstractmethod
    def on_step(
        self,
        state: Any,
        action: Any,
        next_state: Any,
        reward: float,
        done: bool,
        valid_actions: Optional[List[int]],
        next_valid_actions: Optional[List[int]],
        env: EnvForRL,
    ) -> Dict[str, Union[float, int]]:  # info
        raise NotImplementedError()

    @abstractmethod
    def render(
        self,
        state: Any,
        valid_actions: Optional[List[int]],
        action_to_str: Callable[[Any], str],
    ) -> None:
        raise NotImplementedError()


if __name__ == "__main__":
    pass
