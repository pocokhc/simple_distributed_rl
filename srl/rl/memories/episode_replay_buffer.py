import logging
import pickle
import random
import zlib
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, cast

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
        prefix_size: int = 0,
        suffix_size: int = 0,
        skip_head: int = 0,
        skip_tail: int = 0,
        sequential_stride: int = 1,
    ):
        self.cfg = config
        self.batch_size = batch_size
        self.prefix_size = prefix_size
        self.suffix_size = suffix_size
        self.skip_head = skip_head
        self.skip_tail = skip_tail
        self.sequential_stride = sequential_stride

        self.buffer = []
        self.total_size = 0
        self._sequential_batches = [[] for _ in range(self.batch_size)]

        # --- validate
        if not (self.cfg.warmup_size <= self.cfg.capacity):
            raise ValueError(f"assert {self.cfg.warmup_size} <= {self.cfg.capacity}")
        if not (batch_size > 0):
            raise ValueError(f"assert {batch_size} > 0")
        if not (batch_size <= self.cfg.warmup_size):
            raise ValueError(f"assert {batch_size} <= {self.cfg.warmup_size}")

    @property
    def batch_length(self) -> int:
        return self.prefix_size + 1 + self.suffix_size

    def length(self) -> int:
        return self.total_size

    def is_warmup(self) -> bool:
        return self.total_size < self.cfg.warmup_size

    def add(self, steps: List[Any], size: int = 0, serialized: bool = False) -> None:
        # compressなら圧縮状態で、違うならdeserializeしたものをbufferに入れる
        if serialized:
            if not self.cfg.compress:
                steps = pickle.loads(cast(bytes, steps))
        else:
            size = len(steps)
            if self.cfg.compress:
                steps = cast(List[Any], zlib.compress(pickle.dumps(steps), level=self.cfg.compress_level))

        sample_size = size - (self.batch_length + self.skip_head + self.skip_tail) + 1
        if sample_size < 0:
            sample_size = 0
        self.total_size += sample_size
        self.buffer.append((steps, sample_size))

        # capacityを超えないように減らす
        while self.total_size > self.cfg.capacity:
            _, sample_size = self.buffer.pop(0)
            self.total_size -= sample_size

    def serialize(self, steps) -> Any:
        size = len(steps)
        steps = pickle.dumps(steps)
        if self.cfg.compress:
            steps = zlib.compress(steps, level=self.cfg.compress_level)
        return steps, size

    def sample(
        self,
        batch_size: int = -1,
        prefix_size: int = -1,
        suffix_size: int = -1,
        skip_head: int = -1,
        skip_tail: int = -1,
    ):
        if self.total_size < self.cfg.warmup_size:
            return None
        batch_size = self.batch_size if batch_size == -1 else batch_size
        prefix_size = self.prefix_size if prefix_size == -1 else prefix_size
        suffix_size = self.suffix_size if suffix_size == -1 else suffix_size
        skip_head = self.skip_head if skip_head == -1 else skip_head
        skip_tail = self.skip_tail if skip_tail == -1 else skip_tail

        # 各batchの各stepからランダムに取得
        batch_length = prefix_size + 1 + suffix_size
        batches = []
        while len(batches) < batch_size:
            i = random.randint(0, len(self.buffer) - 1)
            steps, _ = self.buffer[i]
            if self.cfg.compress:
                steps = pickle.loads(zlib.decompress(steps))
            sample_size = len(steps) - batch_length - skip_tail
            if len(steps) < sample_size + batch_length:
                logger.warning(f"Episode length must be equal to or greater than batch_length. {len(steps)} >= {sample_size + batch_length}")
                continue
            j = random.randint(skip_head, sample_size)
            batches.append(steps[j : j + batch_length])
        return batches

    def sample_steps(self, batch_size: int = -1):
        if self.total_size < self.cfg.warmup_size:
            return None
        batch_size = self.batch_size if batch_size == -1 else batch_size
        steps, _ = self.buffer[random.randint(0, len(self.buffer) - 1)]
        if self.cfg.compress:
            steps = pickle.loads(zlib.decompress(steps))
        return steps

    def sample_sequential(
        self,
        dummy_step: Optional[list] = None,
        should_drop_batch_func: Optional[Callable[[int, List[list]], bool]] = None,
    ):
        """時系列に沿ったbatchを生成"""
        if self.total_size < self.cfg.warmup_size:
            return None

        batches: List[list] = []
        for i in range(self.batch_size):
            for j in range(99):  # for safety
                # --- 足りなくなったらbufferから追加
                while len(self._sequential_batches[i]) < self.batch_length:
                    r = random.randint(0, len(self.buffer) - 1)
                    steps, _ = self.buffer[r]
                    if self.cfg.compress:
                        steps = pickle.loads(zlib.decompress(steps))
                    if len(steps) <= self.skip_head + self.skip_tail:
                        logger.warning(f"Episode length must be equal to or greater than batch_length. {len(steps)} > {self.skip_head + self.skip_tail}")
                        continue

                    if self.skip_tail <= 0:
                        steps = steps[self.skip_head :]
                    else:
                        steps = steps[self.skip_head : -self.skip_tail]
                    if dummy_step is not None:
                        self._sequential_batches[i].extend([dummy_step] * i)
                    self._sequential_batches[i].extend(steps)

                # --- batchを追加
                batch = self._sequential_batches[i][: self.batch_length]
                self._sequential_batches[i] = self._sequential_batches[i][self.sequential_stride :]
                if should_drop_batch_func is not None:
                    if should_drop_batch_func(i, batch):
                        continue  # 追加できるまで繰り返す
                batches.append(batch)
                break
            else:
                logger.error("Failed to add batch.")
                batches.append([None])
                continue

        return batches

    def call_backup(self, **kwargs):
        return [
            self.total_size,
            self.buffer[:],
        ]

    def call_restore(self, data: Any, **kwargs) -> None:
        self.total_size = data[0]
        self.buffer = data[1][:]
        self._sequential_batches = [[] for _ in range(self.batch_size)]
