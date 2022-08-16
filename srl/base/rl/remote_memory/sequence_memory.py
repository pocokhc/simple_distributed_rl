from typing import Any

from srl.base.rl.base import RLRemoteMemory


class SequenceRemoteMemory(RLRemoteMemory):
    def __init__(self, *args):
        super().__init__(*args)
        self.buffer = []

    def length(self) -> int:
        return len(self.buffer)

    def call_restore(self, data: Any, **kwargs) -> None:
        self.buffer = data

    def call_backup(self, **kwargs):
        return self.buffer

    # --------------------

    def add(self, batch: Any) -> None:
        self.buffer.append(batch)

    def sample(self):
        buffer = self.buffer
        self.buffer = []
        return buffer
