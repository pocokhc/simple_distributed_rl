from typing import Any

from srl.base.rl.base import RLRemoteMemory


class SequenceRemoteMemory(RLRemoteMemory):
    def __init__(self, *args):
        super().__init__(*args)
        self.buffer = []

    def length(self) -> int:
        return len(self.buffer)

    def restore(self, data: Any) -> None:
        self.buffer = data

    def backup(self):
        return self.buffer

    # --------------------

    def add(self, batch: Any) -> None:
        self.buffer.append(batch)

    def sample(self):
        buffer = self.buffer
        self.buffer = []
        return buffer
