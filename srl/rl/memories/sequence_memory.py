from typing import Any, List

from srl.base.define import RLMemoryTypes
from srl.base.rl.base import RLMemory


class SequenceMemory(RLMemory):
    """SequenceMemory

    FIFO形式でやりとりするシンプルなメモリです。
    パラメータは特にありません。
    """

    def __init__(self, *args):
        super().__init__(*args)
        self.buffer = []

    @property
    def memory_type(self) -> RLMemoryTypes:
        return RLMemoryTypes.SEQUENCE

    def length(self) -> int:
        return len(self.buffer)

    def is_warmup_needed(self) -> bool:
        return len(self.buffer) == 0

    def add(self, batch: Any) -> None:
        self.buffer.append(batch)

    def sample(self, step: int = 0, batch_size: int = -1) -> List[Any]:
        buffer = self.buffer
        self.buffer = []
        return buffer

    def call_backup(self, **kwargs):
        return self.buffer

    def call_restore(self, data: Any, **kwargs) -> None:
        self.buffer = data
