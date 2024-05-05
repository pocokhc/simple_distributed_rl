import logging
import pickle
import random
import zlib
from dataclasses import dataclass, field
from typing import Any, List, cast

from srl.base.define import RLMemoryTypes
from srl.base.rl.config import RLConfig
from srl.base.rl.memory import RLMemory

logger = logging.getLogger(__name__)


@dataclass
class _ExperienceReplayBufferConfig:
    capacity: int = 100_000
    warmup_size: int = 1_000


@dataclass
class RLConfigComponentExperienceReplayBuffer:
    """ExperienceReplayBuffer

    これを継承しているアルゴリズムはbatch_size変数とmemory変数を持ちます。
    memory変数からパラメータを設定できます。

    Parameter:
       capacity(int): メモリに保存できる最大サイズ. default is 100_000
       warmup_size(int): warmup size. default is 1_000
       batch_size(int): Batch size. default is 32

    Examples:
       >>> from srl.algorithms import alphazero
       >>> rl_config = alphazero.Config()
       >>> rl_config.batch_size = 64
       >>> rl_config.memory.capacity = 1000
       >>> rl_config.memory.warmup_size = 1000
    """

    #: Batch size
    batch_size: int = 32
    memory: _ExperienceReplayBufferConfig = field(init=False, default_factory=lambda: _ExperienceReplayBufferConfig())

    def assert_params_memory(self):
        assert self.batch_size > 0
        assert self.memory.warmup_size <= self.memory.capacity
        assert self.batch_size <= self.memory.warmup_size


class RandomMemory:
    def __init__(self, capacity: int, compress: bool, compress_level: int = 1):
        self.capacity = capacity
        self.compress = compress
        self.compress_level = compress_level

        self.memory = []
        self.idx = 0

    def length(self) -> int:
        return len(self.memory)

    def add(self, batch: Any) -> None:
        if self.compress:
            batch = zlib.compress(pickle.dumps(batch), level=self.compress_level)

        if len(self.memory) < self.capacity:
            self.memory.append(batch)
        else:
            self.memory[self.idx] = batch
        self.idx += 1
        if self.idx >= self.capacity:
            self.idx = 0

    def sample(self, batch_size: int) -> List[Any]:
        batchs = random.sample(self.memory, batch_size)
        if self.compress:
            return [pickle.loads(zlib.decompress(b)) for b in batchs]
        else:
            return batchs

    def call_backup(self, **kwargs):
        return [
            self.memory,
            self.idx,
        ]

    def call_restore(self, data: Any, **kwargs) -> None:
        self.memory = data[0]
        self.idx = data[1]
        if len(self.memory) > self.capacity:
            self.idx -= len(self.memory) - self.capacity
            if self.idx < 0:
                self.idx = 0
            self.memory = self.memory[-self.capacity :]
        if self.idx >= self.capacity:
            self.idx = 0


class ExperienceReplayBuffer(RandomMemory, RLMemory[RLConfigComponentExperienceReplayBuffer]):
    def __init__(self, *args):
        RLMemory.__init__(self, *args)
        RandomMemory.__init__(
            self,
            self.config.memory.capacity,
            cast(RLConfig, self.config).memory_compress,
            cast(RLConfig, self.config).memory_compress_level,
        )

    @property
    def memory_type(self) -> RLMemoryTypes:
        return RLMemoryTypes.BUFFER

    def is_warmup_needed(self) -> bool:
        return len(self.memory) < self.config.memory.warmup_size
