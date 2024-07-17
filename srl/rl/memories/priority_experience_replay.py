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
       >>> rl_config.memory_capacity = 10000
       >>> rl_config.memory_warmup_size = 10
       >>>
       >>> # ProportionalMemory の設定例
       >>> rl_config.set_proportional_memory()
    """

    #: Batch size
    batch_size: int = 32
    #: capacity
    memory_capacity: int = 100_000
    #: warmup_size
    memory_warmup_size: int = 1_000

    #: memoryデータを圧縮してやり取りするかどうか
    memory_compress: bool = True
    #: memory(zlib)の圧縮レベル
    memory_compress_level: int = -1

    memory_name: str = field(init=False, default="ReplayMemory")
    memory_kwargs: dict = field(init=False, default_factory=dict)

    def assert_params_memory(self):
        assert self.batch_size > 0
        assert self.memory_warmup_size <= self.memory_capacity
        assert self.batch_size <= self.memory_warmup_size

    def set_replay_memory(self):
        self.memory_name = "ReplayMemory"
        self.memory_kwargs = {}
        return self

    def set_proportional_memory(
        self,
        alpha: float = 0.6,
        beta_initial: float = 0.4,
        beta_steps: int = 1_000_000,
        has_duplicate: bool = True,
        epsilon: float = 0.0001,
    ):
        self.memory_name = "ProportionalMemory"
        self.memory_kwargs = dict(
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
        self.memory_name = "RankBaseMemory"
        self.memory_kwargs = dict(
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
        self.memory_name = "RankBaseMemoryLinear"
        self.memory_kwargs = dict(
            alpha=alpha,
            beta_initial=beta_initial,
            beta_steps=beta_steps,
        )
        return self

    def set_custom_memory(self, entry_point: str, kwargs: dict):
        self.memory_name = "custom"
        self.memory_kwargs = dict(
            entry_point=entry_point,
            kwargs=kwargs,
        )
        return self

    # ---------------------

    def create_memory(self, capacity: int, dtype=np.float32) -> IPriorityMemory:
        if self.memory_name == "ReplayMemory":
            from .priority_memories.replay_memory import ReplayMemory

            memory = ReplayMemory(capacity, dtype=dtype, **self.memory_kwargs)
        elif self.memory_name == "ProportionalMemory":
            from .priority_memories.proportional_memory import ProportionalMemory

            memory = ProportionalMemory(capacity, dtype=dtype, **self.memory_kwargs)

        elif self.memory_name == "RankBaseMemory":
            from .priority_memories.rankbase_memory import RankBaseMemory

            memory = RankBaseMemory(capacity, dtype=dtype, **self.memory_kwargs)

        elif self.memory_name == "RankBaseMemoryLinear":
            from .priority_memories.rankbase_memory_linear import RankBaseMemoryLinear

            memory = RankBaseMemoryLinear(capacity, dtype=dtype, **self.memory_kwargs)
        elif self.memory_name == "custom":
            from srl.utils.common import load_module

            return load_module(self.memory_kwargs["entry_point"])(
                capacity, dtype=dtype, **self.memory_kwargs["kwargs"]
            )
        else:
            raise UndefinedError(self.memory_name)

        return memory

    def requires_priority(self) -> bool:
        name = self.memory_name
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


class PriorityExperienceReplay(RLMemory[RLConfigComponentPriorityExperienceReplay]):
    def __init__(self, *args):
        super().__init__(*args)
        self.batch_size = self.config.batch_size
        self.memory = self.config.create_memory(
            self.config.memory_capacity,
            cast(RLConfig, self.config).dtype,
        )

    @property
    def memory_type(self) -> RLMemoryTypes:
        return RLMemoryTypes.PRIORITY

    def length(self) -> int:
        return self.memory.length()

    def is_warmup_needed(self) -> bool:
        return self.memory.length() < self.config.memory_warmup_size

    def add(self, batch: Any, priority: Optional[float] = None, serialized: bool = False) -> None:
        if serialized:
            if self.config.memory_compress:
                pass  # nothing
            else:
                batch = pickle.loads(batch)
        else:
            if self.config.memory_compress:
                batch = zlib.compress(pickle.dumps(batch), self.config.memory_compress_level)
            else:
                pass  # nothing
        self.memory.add(batch, priority)

    def serialize_add_args(self, batch: Any, priority: Optional[float]) -> Any:
        batch = pickle.dumps(batch)
        if self.config.memory_compress:
            batch = zlib.compress(batch, level=self.config.memory_compress_level)
        return (batch, priority)

    def deserialize_add_args(self, raw: Any) -> Any:
        return raw[0], raw[1], True

    def sample(self, step: int, batch_size: int = 0) -> Tuple[List[Any], np.ndarray, List[Any]]:
        batchs, weights, update_args = self.memory.sample(
            batch_size if batch_size > 0 else self.config.batch_size, step
        )
        if self.config.memory_compress:
            batchs = [pickle.loads(zlib.decompress(b)) for b in batchs]
        return batchs, weights, update_args

    def update(self, update_args: List[Any], priorities: np.ndarray) -> None:
        self.memory.update(update_args, priorities)

    def call_backup(self, **kwargs):
        return self.memory.backup()

    def call_restore(self, data: Any, **kwargs) -> None:
        self.memory.restore(data)
