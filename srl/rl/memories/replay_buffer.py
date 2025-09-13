import logging
import pickle
import random
import zlib
from dataclasses import dataclass
from typing import Any, Generic

from srl.base.rl.config import TRLConfig
from srl.base.rl.memory import RLMemory

logger = logging.getLogger(__name__)


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


class ReplayBuffer:
    def __init__(self, config: ReplayBufferConfig, batch_size: int):
        self.cfg = config
        self.batch_size = batch_size
        self.buffer = []
        self.idx = 0
        self._validate_params(batch_size)

    def _validate_params(self, batch_size: int):
        if not (self.cfg.warmup_size <= self.cfg.capacity):
            raise ValueError(f"assert {self.cfg.warmup_size} <= {self.cfg.capacity}")
        if not (batch_size > 0):
            raise ValueError(f"assert {batch_size} > 0")
        if not (batch_size <= self.cfg.warmup_size):
            raise ValueError(f"assert {batch_size} <= {self.cfg.warmup_size}")

    def length(self) -> int:
        return len(self.buffer)

    def add(self, batch: Any, serialized: bool = False) -> None:
        # compressなら圧縮状態で、違うならdeserializeしたものをbufferに入れる
        if serialized:
            if not self.cfg.compress:
                batch = pickle.loads(batch)
        else:
            if self.cfg.compress:
                batch = zlib.compress(pickle.dumps(batch), level=self.cfg.compress_level)

        if len(self.buffer) < self.cfg.capacity:
            self.buffer.append(batch)
        else:
            self.buffer[self.idx] = batch
        self.idx += 1
        if self.idx >= self.cfg.capacity:
            self.idx = 0

    def serialize(self, batch: Any) -> Any:
        batch = pickle.dumps(batch)
        if self.cfg.compress:
            batch = zlib.compress(batch, level=self.cfg.compress_level)
        return batch

    def is_warmup_needed(self) -> bool:
        return len(self.buffer) < self.cfg.warmup_size

    def sample(self, batch_size: int = -1):
        if len(self.buffer) < self.cfg.warmup_size:
            return None
        if batch_size < 1:
            batch_size = self.batch_size
        batches = random.sample(self.buffer, batch_size)

        if self.cfg.compress:
            batches = [pickle.loads(zlib.decompress(b)) for b in batches]
        return batches

    def call_backup(self, **kwargs):
        return [
            self.buffer[:],
            self.idx,
            self.cfg.compress,
        ]

    def call_restore(self, data: Any, **kwargs) -> None:
        self.buffer = data[0][:]
        self.idx = data[1]
        compressed = data[2]

        if len(self.buffer) > self.cfg.capacity:
            self.idx -= len(self.buffer) - self.cfg.capacity
            if self.idx < 0:
                self.idx = 0
            self.buffer = self.buffer[-self.cfg.capacity :]
        if self.idx >= self.cfg.capacity:
            self.idx = 0

        if compressed and not self.cfg.compress:
            self.buffer = [pickle.loads(zlib.decompress(b)) for b in self.buffer]
        if not compressed and self.cfg.compress:
            self.buffer = [zlib.compress(pickle.dumps(b)) for b in self.buffer]


class RLReplayBuffer(Generic[TRLConfig], ReplayBuffer, RLMemory[TRLConfig]):
    def __init__(self, *args):
        RLMemory.__init__(self, *args)
        assert hasattr(self.config, "batch_size")
        assert hasattr(self.config, "memory")
        assert isinstance(self.config.memory, ReplayBufferConfig)  # type: ignore
        ReplayBuffer.__init__(self, self.config.memory, self.config.batch_size)  # type: ignore

    def setup(self, register_add: bool = True, register_sample: bool = True) -> None:
        if register_add:
            self.register_worker_func_custom(self.add, self.serialize)
        if register_sample:
            self.register_trainer_recv_func(self.sample)
