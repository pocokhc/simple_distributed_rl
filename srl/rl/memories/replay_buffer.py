import pickle
import random
import zlib
from dataclasses import dataclass
from typing import Any, Generic

from srl.base.rl.config import TRLConfig
from srl.base.rl.memory import RLMemory


@dataclass
class ReplayBufferConfig:
    #: capacity
    capacity: int = 100_000
    #: warmup_size
    warmup_size: int = 1_000

    #: memoryデータを圧縮してやり取りするかどうか
    compress: bool = True
    #: memory(zlib)の圧縮レベル
    compress_level: int = -1

    def create_memory(self, batch_size: int = 32):
        return ReplayBuffer(
            batch_size,
            capacity=self.capacity,
            warmup_size=self.warmup_size,
            compress=self.compress,
            compress_level=self.compress_level,
        )


class ReplayBuffer:
    def __init__(
        self,
        batch_size: int = 32,
        capacity: int = 100_000,
        warmup_size: int = 1000,
        compress: bool = True,
        compress_level: int = -1,
    ):
        self.batch_size = batch_size
        self.capacity = capacity
        self.warmup_size = warmup_size
        self.compress = compress
        self.compress_level = compress_level

        self.buffer = []
        self.idx = 0
        self._validate_params(batch_size)

    def _validate_params(self, batch_size: int):
        if not (self.warmup_size <= self.capacity):
            raise ValueError(f"assert {self.warmup_size} <= {self.capacity}")
        if not (batch_size > 0):
            raise ValueError(f"assert {batch_size} > 0")
        if not (batch_size <= self.warmup_size):
            raise ValueError(f"assert {batch_size} <= {self.warmup_size}")

    def clear(self):
        self.buffer = []
        self.idx = 0

    def length(self) -> int:
        return len(self.buffer)

    def add(self, batch: Any, serialized: bool = False) -> None:
        # compressなら圧縮状態で、違うならdeserializeしたものをbufferに入れる
        if serialized:
            if not self.compress:
                batch = pickle.loads(batch)
        else:
            if self.compress:
                batch = zlib.compress(pickle.dumps(batch), level=self.compress_level)

        if len(self.buffer) < self.capacity:
            self.buffer.append(batch)
        else:
            self.buffer[self.idx] = batch
        self.idx += 1
        if self.idx >= self.capacity:
            self.idx = 0

    def serialize(self, batch: Any) -> Any:
        batch = pickle.dumps(batch)
        if self.compress:
            batch = zlib.compress(batch, level=self.compress_level)
        return batch

    def is_warmup_needed(self) -> bool:
        return len(self.buffer) < self.warmup_size

    def sample(self, batch_size: int = -1):
        if len(self.buffer) < self.warmup_size:
            return None
        if batch_size < 1:
            batch_size = self.batch_size
        batches = random.sample(self.buffer, batch_size)

        if self.compress:
            batches = [pickle.loads(zlib.decompress(b)) for b in batches]
        return batches

    def call_backup(self, **kwargs):
        return [
            self.buffer[:],
            self.idx,
            self.compress,
        ]

    def call_restore(self, data: Any, **kwargs) -> None:
        self.buffer = data[0][:]
        self.idx = data[1]
        compressed = data[2]

        if len(self.buffer) > self.capacity:
            self.idx -= len(self.buffer) - self.capacity
            if self.idx < 0:
                self.idx = 0
            self.buffer = self.buffer[-self.capacity :]
        if self.idx >= self.capacity:
            self.idx = 0

        if compressed and not self.compress:
            self.buffer = [pickle.loads(zlib.decompress(b)) for b in self.buffer]
        if not compressed and self.compress:
            self.buffer = [zlib.compress(pickle.dumps(b)) for b in self.buffer]


class RLReplayBuffer(Generic[TRLConfig], ReplayBuffer, RLMemory[TRLConfig]):
    def __init__(self, *args):
        RLMemory.__init__(self, *args)
        assert hasattr(self.config, "batch_size")
        assert hasattr(self.config, "memory")
        assert isinstance(self.config.memory, ReplayBufferConfig)  # type: ignore
        ReplayBuffer.__init__(
            self,
            self.config.batch_size,  # type: ignore
            self.config.memory.capacity,  # type: ignore
            self.config.memory.warmup_size,  # type: ignore
            self.config.memory.compress,  # type: ignore
            self.config.memory.compress_level,  # type: ignore
        )

    def setup(self, register_add: bool = True, register_sample: bool = True) -> None:
        if register_add:
            self.register_worker_func_custom(self.add, self.serialize)
        if register_sample:
            self.register_trainer_recv_func(self.sample)
