import logging
import pickle
import random
import zlib
from dataclasses import dataclass
from typing import Any, List

from srl.base.define import RLMemoryTypes
from srl.base.rl.memory import RLMemory

logger = logging.getLogger(__name__)


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
       >>> rl_config.memory_capacity = 1000
       >>> rl_config.memory_warmup_size = 1000
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

    def assert_params_memory(self):
        assert self.batch_size > 0
        assert self.memory_warmup_size <= self.memory_capacity
        assert self.batch_size <= self.memory_warmup_size


class RandomMemory:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = []
        self.idx = 0

    def length(self) -> int:
        return len(self.memory)

    def add(self, batch: Any) -> None:
        if len(self.memory) < self.capacity:
            self.memory.append(batch)
        else:
            self.memory[self.idx] = batch
        self.idx += 1
        if self.idx >= self.capacity:
            self.idx = 0

    def sample(self, batch_size: int) -> List[Any]:
        return random.sample(self.memory, batch_size)

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
        RandomMemory.__init__(self, self.config.memory_capacity)

    @property
    def memory_type(self) -> RLMemoryTypes:
        return RLMemoryTypes.BUFFER

    def is_warmup_needed(self) -> bool:
        return len(self.memory) < self.config.memory_warmup_size

    def add(self, batch: Any, serialized: bool = False) -> None:
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
        RandomMemory.add(self, batch)

    def serialize_add_args(self, batch: Any) -> Any:
        batch = pickle.dumps(batch)
        if self.config.memory_compress:
            batch = zlib.compress(batch, level=self.config.memory_compress_level)
        return batch

    def deserialize_add_args(self, raw: Any) -> Any:
        return raw, True

    def sample(self, batch_size: int = 0) -> List[Any]:
        batchs = RandomMemory.sample(self, batch_size if batch_size > 0 else self.config.batch_size)
        if self.config.memory_compress:
            batchs = [pickle.loads(zlib.decompress(b)) for b in batchs]
        return batchs
