import pickle
import zlib
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, cast

import numpy as np

from srl.base.define import RLMemoryTypes
from srl.base.exception import UndefinedError
from srl.base.rl.config import RLConfig
from srl.base.rl.memory import RLMemory
from srl.rl.memories.priority_memories.imemory import IPriorityMemory


@dataclass
class _PriorityExperienceReplayConfig:
    capacity: int = 10_000
    warmup_size: int = 1_000
    _name: str = field(init=False, default="ReplayMemory")
    _kwargs: dict = field(init=False, default_factory=dict)

    def set_replay_memory(self):
        self._name = "ReplayMemory"
        return self

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
        return self

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
        return self

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
        return self

    def set_custom_memory(self, entry_point: str, kwargs: dict):
        self._name = "custom"
        self._kwargs = dict(
            entry_point=entry_point,
            kwargs=kwargs,
        )
        return self


@dataclass
class RLConfigComponentPriorityExperienceReplay:
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


class PriorityExperienceReplay(RLMemory[RLConfigComponentPriorityExperienceReplay]):
    def __init__(self, *args):
        super().__init__(*args)
        self.batch_size = self.config.batch_size
        self.memory = self.create_memory(
            self.config.memory._name,
            self.config.memory.capacity,
            self.config.memory._kwargs,
            cast(RLConfig, self.config).dtype,
        )

    def create_memory(self, name: str, capacity: int, kwargs, dtype=np.float32) -> IPriorityMemory:
        if name == "ReplayMemory":
            from .priority_memories.replay_memory import ReplayMemory

            memory = ReplayMemory(dtype=dtype, **kwargs)
        elif name == "ProportionalMemory":
            from .priority_memories.proportional_memory import ProportionalMemory

            memory = ProportionalMemory(capacity, dtype=dtype, **kwargs)

        elif name == "RankBaseMemory":
            from .priority_memories.rankbase_memory import RankBaseMemory

            memory = RankBaseMemory(capacity, dtype=dtype, **kwargs)

        elif name == "RankBaseMemoryLinear":
            from .priority_memories.rankbase_memory_linear import RankBaseMemoryLinear

            memory = RankBaseMemoryLinear(capacity, dtype=dtype, **kwargs)
        elif name == "custom":
            from srl.utils.common import load_module

            return load_module(kwargs["entry_point"])(capacity, dtype=dtype, **kwargs["kwargs"])
        else:
            raise UndefinedError(name)

        return memory

    def requires_priority(self) -> bool:
        name = self.config.memory._name
        if name == "ReplayMemory":
            return False
        elif name == "ProportionalMemory":
            return True
        elif name == "RankBaseMemory":
            return True
        elif name == "RankBaseMemoryLinear":
            return True
        elif name == "custom":
            return True
        return False

    # ---------------------------

    @property
    def memory_type(self) -> RLMemoryTypes:
        return RLMemoryTypes.PRIORITY

    def length(self) -> int:
        return self.memory.length()

    def is_warmup_needed(self) -> bool:
        return self.memory.length() < self.config.memory.warmup_size

    def add(self, batch: Any, priority: Optional[float] = None):
        if cast(RLConfig, self.config).memory_compress:
            batch = zlib.compress(pickle.dumps(batch), level=cast(RLConfig, self.config).memory_compress_level)
        self.memory.add(batch, priority)

    def sample(self, batch_size: int, step: int) -> Tuple[List[int], List[Any], np.ndarray]:
        index_list, batchs, weights = self.memory.sample(batch_size, step)
        if cast(RLConfig, self.config).memory_compress:
            batchs = [pickle.loads(zlib.decompress(b)) for b in batchs]
        return index_list, batchs, weights

    def update(self, indices: List[int], batchs: List[Any], priorities: np.ndarray) -> None:
        batchs = [pickle.loads(zlib.decompress(b)) for b in batchs]
        self.memory.update(indices, batchs, priorities)

    def call_backup(self, **kwargs):
        return self.memory.backup()

    def call_restore(self, data: Any, **kwargs) -> None:
        self.memory.restore(data)
