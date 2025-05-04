import logging
import pickle
import random
import zlib
from dataclasses import dataclass
from typing import Any, Generic, List, Literal, cast

from srl.base.define import RLMemoryTypes
from srl.base.rl.config import TRLConfig
from srl.base.rl.memory import RLMemory

logger = logging.getLogger(__name__)


@dataclass
class EpisodeReplayBufferConfig:
    #: capacity
    capacity: int = 10_000
    #: warmup_size
    warmup_size: int = 1_000

    #: memoryデータを圧縮してやり取りするかどうか
    compress: bool = True
    #: memory(zlib)の圧縮レベル
    compress_level: int = -1


class EpisodeReplayBuffer:
    def __init__(
        self,
        config: EpisodeReplayBufferConfig,
        batch_size: int,
        batch_length: int,
        sample_type: Literal["random", "episode"] = "random",
    ):
        self.cfg = config
        self.batch_size = batch_size
        self.batch_length = batch_length
        self.sample_type = sample_type

        self.buffer = []
        self.idx = 0
        self.total_size = 0
        self._sequential_batches = [[] for _ in range(self.batch_size)]

        self._validate_params(batch_size, batch_length)

    def _validate_params(self, batch_size: int, batch_length: int):
        if not (self.cfg.warmup_size <= self.cfg.capacity):
            raise ValueError(f"assert {self.cfg.warmup_size} <= {self.cfg.capacity}")
        if not (batch_size > 0):
            raise ValueError(f"assert {batch_size} > 0")
        if not (batch_size <= self.cfg.warmup_size):
            raise ValueError(f"assert {batch_size} <= {self.cfg.warmup_size}")
        if not (batch_length > 0):
            raise ValueError(f"assert {batch_length} > 0")

    @property
    def memory_type(self) -> RLMemoryTypes:
        return RLMemoryTypes.BUFFER

    def length(self) -> int:
        return self.total_size

    def add(self, steps: List[Any], size: int = 0, serialized: bool = False) -> None:
        # compressなら圧縮状態で、違うならdeserializeしたものをbufferに入れる
        if serialized:
            if not self.cfg.compress:
                steps = pickle.loads(cast(bytes, steps))
        else:
            size = len(steps)
            if self.cfg.compress:
                steps = cast(List[Any], zlib.compress(pickle.dumps(steps), level=self.cfg.compress_level))

        if self.sample_type == "random":
            if size < self.batch_length:
                logger.warning(f"Episode length must be equal to or greater than batch_length. episode_len={size}")
            self.total_size += size - self.batch_length
        else:
            self.total_size += size

        if len(self.buffer) < self.cfg.capacity:
            self.buffer.append((steps, size))
        else:
            self.total_size -= self.buffer[self.idx][1]
            self.buffer[self.idx] = (steps, size)
        self.idx += 1
        if self.idx >= self.cfg.capacity:
            self.idx = 0

    def serialize(self, steps) -> Any:
        size = len(steps)
        steps = pickle.dumps(steps)
        if self.cfg.compress:
            steps = zlib.compress(steps, level=self.cfg.compress_level)
        return steps, size

    def sample(
        self,
        batch_size: int = -1,
        batch_length: int = -1,
        skip_head: int = 0,
        skip_tail: int = 0,
    ):
        if self.total_size < self.cfg.warmup_size:
            return None
        batch_size = self.batch_size if batch_size == -1 else batch_size
        batch_length = self.batch_length if batch_length == -1 else batch_length

        # 各batchの各stepからランダムに取得
        batches = []
        while len(batches) < batch_size:
            i = random.randint(0, len(self.buffer) - 1)
            steps, size = self.buffer[i]
            if size <= batch_length + skip_head + skip_tail:
                logger.warning(f"Episode length must be equal to or greater than batch_length. episode_len={size}")
                continue
            j = random.randint(0, size - batch_length - skip_head - skip_tail) + skip_head
            if self.cfg.compress:
                steps = pickle.loads(zlib.decompress(steps))
            batches.append(steps[j : j + batch_length])
        return batches

    def sample_sequential_episodes(self):
        """時系列に沿ったbatchを生成"""
        if self.total_size < self.cfg.warmup_size:
            return None

        batches = []
        for i in range(self.batch_size):
            while len(self._sequential_batches[i]) < self.batch_length:
                r = random.randint(0, len(self.buffer) - 1)
                steps = self.buffer[r][0]
                if self.cfg.compress:
                    steps = pickle.loads(zlib.decompress(steps))
                self._sequential_batches[i].extend(steps)

            batches.append(self._sequential_batches[i][: self.batch_length])
            self._sequential_batches[i] = self._sequential_batches[i][self.batch_length :]

        return batches

    def call_backup(self, **kwargs):
        return [
            self.idx,
            self.total_size,
            self.buffer[:],
        ]

    def call_restore(self, data: Any, **kwargs) -> None:
        self.idx = data[0]
        self.total_size = data[1]
        self.buffer = data[2][:]
        self._sequential_batches = [[] for _ in range(self.batch_size)]


class RLEpisodeReplayBuffer(Generic[TRLConfig], EpisodeReplayBuffer, RLMemory[TRLConfig]):
    def __init__(self, *args):
        RLMemory.__init__(self, *args)
        assert hasattr(self.config, "batch_size")
        assert hasattr(self.config, "batch_length")
        assert hasattr(self.config, "memory")
        assert isinstance(self.config.memory, EpisodeReplayBufferConfig)  # type: ignore
        EpisodeReplayBuffer.__init__(self, self.config.memory, self.config.batch_size, self.config.batch_length)  # type: ignore

        self.register_worker_func(self.add, self.serialize)
        self.register_trainer_recv_func(self.sample)
