from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np

from srl.base.rl.base import RLMemory
from srl.rl.memories.priority_memories.imemory import IPriorityMemory


@dataclass
class _PriorityExperienceReplayConfig:
    capacity: int = 100_000
    warmup_size: int = 1_000
    _name: str = field(init=False, default="ReplayMemory")
    _kwargs: dict = field(init=False, default_factory=dict)

    def set_replay_memory(self):
        self._name = "ReplayMemory"

    def set_proportional_memory(
        self,
        alpha: float = 0.6,
        beta_initial: float = 0.4,
        beta_steps: int = 1_000_000,
        has_duplicate: bool = True,
        epsilon: float = 0.0001,
    ):
        self._name = "ProportionalMemory"
        self._kwargs = dict(
            alpha=alpha,
            beta_initial=beta_initial,
            beta_steps=beta_steps,
            has_duplicate=has_duplicate,
            epsilon=epsilon,
        )

    def set_rankbase_memory(
        self,
        alpha: float = 0.6,
        beta_initial: float = 0.4,
        beta_steps: int = 1_000_000,
    ):
        self._name = "RankBaseMemory"
        self._kwargs = dict(
            alpha=alpha,
            beta_initial=beta_initial,
            beta_steps=beta_steps,
        )

    def set_rankbase_memory_linear(
        self,
        alpha: float = 0.6,
        beta_initial: float = 0.4,
        beta_steps: int = 1_000_000,
    ):
        self._name = "RankBaseMemoryLinear"
        self._kwargs = dict(
            alpha=alpha,
            beta_initial=beta_initial,
            beta_steps=beta_steps,
        )

    def set_custom_memory(self, entry_point: str, kwargs: dict):
        self._name = "custom"
        self._kwargs = dict(
            entry_point=entry_point,
            kwargs=kwargs,
        )

    # ---------------------------

    def create_memory(self) -> IPriorityMemory:
        if self._name == "ReplayMemory":
            from .priority_memories.replay_memory import ReplayMemory

            memory = ReplayMemory(self.capacity)
        elif self._name == "ProportionalMemory":
            from .priority_memories.proportional_memory import ProportionalMemory

            memory = ProportionalMemory(self.capacity, **self._kwargs)

        elif self._name == "RankBaseMemory":
            from .priority_memories.rankbase_memory import RankBaseMemory

            memory = RankBaseMemory(self.capacity, **self._kwargs)

        elif self._name == "RankBaseMemoryLinear":
            from .priority_memories.rankbase_memory_linear import RankBaseMemoryLinear

            memory = RankBaseMemoryLinear(self.capacity, **self._kwargs)
        elif self._name == "custom":
            from srl.utils.common import load_module

            return load_module(self._kwargs["entry_point"])(**self._kwargs["kwargs"])
        else:
            raise ValueError(self._name)

        return memory

    def requires_priority(self) -> bool:
        if self._name == "ReplayMemory":
            return False
        elif self._name == "ProportionalMemory":
            return True
        elif self._name == "RankBaseMemory":
            return True
        elif self._name == "RankBaseMemoryLinear":
            return True
        return False


@dataclass
class PriorityExperienceReplayConfig:
    """PriorityExperienceReplay

    これを継承しているアルゴリズムはbatch_size変数とmemory変数を持ちます。
    memory変数から関数を呼ぶことで各memoryを設定できます。
    また、memory変数から任意のパラメータを設定できます。

    Examples:
       >>> from srl.algorithms import dqn
       >>> rl_config = dqn.Config()
       >>>
       >>> # 各パラメータの設定例
       >>> rl_config.batch_size = 64
       >>> rl_config.memory.capacity = 10000
       >>> rl_config.memory.warmup_size = 10
       >>>
       >>> # ProportionalMemory の設定例
       >>> rl_config.memory.set_proportional_memory()
    """

    batch_size: int = 32
    memory: _PriorityExperienceReplayConfig = field(
        init=False, default_factory=lambda: _PriorityExperienceReplayConfig()
    )

    def assert_params_memory(self):
        assert self.batch_size > 0
        assert self.memory.warmup_size <= self.memory.capacity
        assert self.batch_size <= self.memory.warmup_size


class PriorityExperienceReplay(RLMemory):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: PriorityExperienceReplayConfig = self.config
        self.batch_size = self.config.batch_size
        self.memory = self.config.memory.create_memory()

    def length(self) -> int:
        return self.memory.length()

    def is_warmup_needed(self) -> bool:
        return self.memory.length() < self.config.memory.warmup_size

    def add(self, batch: Any, priority: Optional[float] = None):
        self.memory.add(batch, priority)

    def sample(self, batch_size: int, step: int) -> Tuple[List[int], List[Any], np.ndarray]:
        return self.memory.sample(batch_size, step)

    def update(self, memory_update_args: Tuple[List[int], List[Any], np.ndarray]) -> None:
        self.memory.update(*memory_update_args)

    def call_backup(self, **kwargs):
        return self.memory.backup()

    def call_restore(self, data: Any, **kwargs) -> None:
        self.memory.restore(data)
