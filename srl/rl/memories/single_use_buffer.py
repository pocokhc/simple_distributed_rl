import pickle
from typing import Any, Generic

from srl.base.define import RLMemoryTypes
from srl.base.rl.config import TRLConfig
from srl.base.rl.memory import RLMemory


class SingleUseBuffer:
    def __init__(self):
        self.buffer = []

    @property
    def memory_type(self) -> RLMemoryTypes:
        return RLMemoryTypes.SEQUENCE

    def length(self) -> int:
        return len(self.buffer)

    def add(self, batch: Any, serialized: bool = False) -> None:
        if serialized:
            batch = pickle.loads(batch)
        self.buffer.append(batch)

    def serialize(self, batch: Any) -> Any:
        return pickle.dumps(batch)

    def sample(self):
        if len(self.buffer) == 0:
            return None
        buffer = self.buffer
        self.buffer = []
        return buffer

    def call_backup(self, **kwargs):
        return self.buffer[:]

    def call_restore(self, data: Any, **kwargs) -> None:
        self.buffer = data[:]


class RLSingleUseBuffer(Generic[TRLConfig], SingleUseBuffer, RLMemory[TRLConfig]):
    def __init__(self, *args):
        RLMemory.__init__(self, *args)
        SingleUseBuffer.__init__(self)

    def setup(self, register_add: bool = True, register_sample: bool = True) -> None:
        if register_add:
            self.register_worker_func(self.add, self.serialize)
        if register_sample:
            self.register_trainer_recv_func(self.sample)
