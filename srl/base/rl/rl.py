from __future__ import annotations

import os
import pickle
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

from srl.base.rl.config import RLConfig
from srl.base.rl.memory import Memory


class RLParameter(ABC):
    def __init__(self, config: RLConfig):
        self.config = config

    @abstractmethod
    def restore(self, data: Optional[Any]) -> None:
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


class RLTrainer(ABC):
    def __init__(self, config: RLConfig, parameter: RLParameter):
        self.config = config
        self.parameter = parameter

        self.train_count = 0

    @abstractmethod
    def train_on_batchs(self, batchs: list, weights: list[float]) -> tuple[list[float], dict[str, float | int]]:
        raise NotImplementedError()

    def train(self, memory: Memory) -> dict[str, Any]:
        memory_len = memory.length()
        warmup_size = self.config.memory_warmup_size
        batch_size = self.config.batch_size

        info: dict[str, Any] = {
            "memory": memory_len,
            "train": self.train_count,
        }

        if memory_len < warmup_size:
            return info

        t0 = time.time()
        (indexes, batchs, weights) = memory.sample(batch_size, self.train_count)
        priorities, train_info = self.train_on_batchs(batchs, weights)
        self.train_count += 1

        # memory update
        memory.update(indexes, batchs, priorities)

        # info
        info["train_time"] = time.time() - t0
        info["info"] = train_info
        return info


class RLWorker(ABC):
    def __init__(
        self,
        config: RLConfig,
        parameter: Optional[RLParameter] = None,
        worker_id: int = 0,
    ):
        self.config = config
        self.parameter = parameter
        self.worker_id = worker_id
        self.training = False

    def set_training(self, training: bool) -> None:
        self.training = training

    @abstractmethod
    def on_reset(self, state: Any, valid_actions: Optional[list[int]]) -> None:
        raise NotImplementedError()

    @abstractmethod
    def policy(self, state: Any, valid_actions: Optional[list[int]]) -> tuple[Any, Any]:  # (env_action, agent_action)
        raise NotImplementedError()

    @abstractmethod
    def on_step(
        self,
        state: Any,
        action: Any,
        next_state: Any,
        reward: float,
        done: bool,
        valid_actions: Optional[list[int]],
        next_valid_actions: Optional[list[int]],
    ) -> dict[str, float | int] | tuple[Any, float, dict[str, float | int]]:
        # if self.training:
        #    return batch, priority, info
        # else:
        #    return info
        raise NotImplementedError()

    @abstractmethod
    def render(self, state: Any, valid_actions: Optional[list[int]]) -> None:
        raise NotImplementedError()


if __name__ == "__main__":
    pass
