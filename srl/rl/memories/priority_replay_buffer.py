import pickle
import zlib
from dataclasses import dataclass, field
from typing import Any, Generic, List, Optional

import numpy as np

from srl.base.define import RLMemoryTypes
from srl.base.exception import UndefinedError
from srl.base.rl.config import TRLConfig
from srl.base.rl.memory import RLMemory
from srl.rl.memories.priority_memories.imemory import IPriorityMemory


@dataclass
class PriorityReplayBufferConfig:
    #: capacity
    capacity: int = 100_000
    #: warmup_size
    warmup_size: int = 1_000

    #: memoryデータを圧縮してやり取りするかどうか
    compress: bool = True
    #: memory(zlib)の圧縮レベル
    compress_level: int = -1

    name: str = field(init=False, default="ReplayBuffer")
    kwargs: dict = field(init=False, default_factory=dict)

    def set_replay_buffer(self):
        self.name = "ReplayBuffer"
        self.kwargs = {}
        return self

    def set_proportional(
        self,
        alpha: float = 0.6,
        beta_initial: float = 0.4,
        beta_steps: int = 1_000_000,
        has_duplicate: bool = True,
        epsilon: float = 0.0001,
    ):
        self.name = "Proportional"
        self.kwargs = dict(
            alpha=alpha,
            beta_initial=beta_initial,
            beta_steps=beta_steps,
            has_duplicate=has_duplicate,
            epsilon=epsilon,
        )
        return self

    def set_rankbased(
        self,
        alpha: float = 0.6,
        beta_initial: float = 0.4,
        beta_steps: int = 1_000_000,
    ):
        self.name = "RankBased"
        self.kwargs = dict(
            alpha=alpha,
            beta_initial=beta_initial,
            beta_steps=beta_steps,
        )
        return self

    def set_rankbased_linear(
        self,
        alpha: float = 0.6,
        beta_initial: float = 0.4,
        beta_steps: int = 1_000_000,
    ):
        self.name = "RankBasedLinear"
        self.kwargs = dict(
            alpha=alpha,
            beta_initial=beta_initial,
            beta_steps=beta_steps,
        )
        return self

    def set_custom(self, entry_point: str, kwargs: dict):
        self.name = "custom"
        self.kwargs = dict(
            entry_point=entry_point,
            kwargs=kwargs,
        )
        return self

    # ---------------------

    def create_memory(self, capacity: int, dtype=np.float32) -> IPriorityMemory:
        if self.name == "ReplayBuffer":
            from .priority_memories.replay_buffer import ReplayBuffer

            memory = ReplayBuffer(capacity, dtype=dtype, **self.kwargs)
        elif self.name == "Proportional":
            from .priority_memories.proportional_memory import ProportionalMemory

            memory = ProportionalMemory(capacity, dtype=dtype, **self.kwargs)

        elif self.name == "RankBased":
            from .priority_memories.rankbased_memory import RankBasedMemory

            memory = RankBasedMemory(capacity, dtype=dtype, **self.kwargs)

        elif self.name == "RankBasedLinear":
            from .priority_memories.rankbased_memory_linear import RankBasedMemoryLinear

            memory = RankBasedMemoryLinear(capacity, dtype=dtype, **self.kwargs)
        elif self.name == "custom":
            from srl.utils.common import load_module

            return load_module(self.kwargs["entry_point"])(capacity, dtype=dtype, **self.kwargs["kwargs"])
        else:
            raise UndefinedError(self.name)

        return memory

    def requires_priority(self) -> bool:
        if self.name == "ReplayBuffer":
            return False
        elif self.name == "Proportional":
            return True
        elif self.name == "RankBased":
            return True
        elif self.name == "RankBasedLinear":
            return True
        elif self.name == "custom":
            return True
        return False


class PriorityReplayBuffer:
    def __init__(self, config: PriorityReplayBufferConfig, batch_size: int, dtype=np.float32):
        self.cfg = config
        self.batch_size = batch_size
        self.memory = self.cfg.create_memory(self.cfg.capacity, dtype)
        self.step = 0

        self._validate_params(batch_size)

    def _validate_params(self, batch_size: int):
        if not (self.cfg.warmup_size <= self.cfg.capacity):
            raise ValueError(f"assert {self.cfg.warmup_size} <= {self.cfg.capacity}")
        if not (batch_size > 0):
            raise ValueError(f"assert {batch_size} > 0")
        if not (batch_size <= self.cfg.warmup_size):
            raise ValueError(f"assert {batch_size} <= {self.cfg.warmup_size}")

    @property
    def memory_type(self) -> RLMemoryTypes:
        return RLMemoryTypes.PRIORITY

    def length(self) -> int:
        return self.memory.length()

    def add(self, batch: Any, priority: Optional[float] = None, serialized: bool = False) -> None:
        # compressなら圧縮状態で、違うならdeserializeしたものをbufferに入れる
        if serialized:
            if not self.cfg.compress:
                batch = pickle.loads(batch)
        else:
            if self.cfg.compress:
                batch = zlib.compress(pickle.dumps(batch), level=self.cfg.compress_level)

        self.memory.add(batch, priority)

    def serialize(self, batch: Any, priority: Optional[float] = None) -> Any:
        batch = pickle.dumps(batch)
        if self.cfg.compress:
            batch = zlib.compress(batch, level=self.cfg.compress_level)
        return (batch, priority)

    def is_warmup_needed(self) -> bool:
        return self.memory.length() < self.cfg.warmup_size

    def sample(self, step: int = -1, batch_size: int = -1):
        if self.memory.length() < self.cfg.warmup_size:
            return None
        batch_size = batch_size if batch_size > -1 else self.batch_size
        step = step if step > -1 else self.step
        batches, weights, update_args = self.memory.sample(batch_size, step)

        if self.cfg.compress:
            batches = [pickle.loads(zlib.decompress(b)) for b in batches]
        return batches, weights, update_args

    def update(self, update_args: List[Any], priorities: np.ndarray, step: int) -> None:
        self.memory.update(update_args, priorities)
        self.step = step

    def call_backup(self, **kwargs):
        return self.memory.backup()

    def call_restore(self, data: Any, **kwargs) -> None:
        self.memory.restore(data)


class RLPriorityReplayBuffer(Generic[TRLConfig], PriorityReplayBuffer, RLMemory[TRLConfig]):
    def __init__(self, *args):
        RLMemory.__init__(self, *args)
        assert hasattr(self.config, "memory")
        assert hasattr(self.config, "batch_size")
        assert isinstance(self.config.memory, PriorityReplayBufferConfig)  # type: ignore
        PriorityReplayBuffer.__init__(self, self.config.memory, self.config.batch_size)  # type: ignore

        self.register_worker_func(self.add, self.serialize)
        self.register_trainer_recv_func(self.sample)
        self.register_trainer_send_func(self.update)
