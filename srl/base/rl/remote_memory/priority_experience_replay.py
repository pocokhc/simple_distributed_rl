from typing import Any, List, Tuple

from srl.base.rl.base import RLRemoteMemory
from srl.rl.memory.proportional_memory import ProportionalMemory
from srl.rl.memory.rankbase_memory import RankBaseMemory
from srl.rl.memory.replay_memory import ReplayMemory


class PriorityExperienceReplay(RLRemoteMemory):
    def __init__(self, *args):
        super().__init__(*args)

    def init(self, name: str, capacity: int, alpha: float, beta_initial: float, beta_steps: int):

        memories = [
            ReplayMemory,
            RankBaseMemory,
            ProportionalMemory,
        ]
        names = [m.getName() for m in memories]
        if name not in names:
            raise ValueError("Unknown memory({}). Memories is [{}].".format(name, ",".join(names)))

        for m in memories:
            if m.getName() == name:
                self.memory = m(capacity, alpha, beta_initial, beta_steps)
                break

    def length(self) -> int:
        return len(self.memory)

    def restore(self, data: Any) -> None:
        self.memory.restore(data)

    def backup(self):
        return self.memory.backup()

    # ---------------------------

    def add(self, batch, priority):
        self.memory.add(batch, priority)

    def sample(self, step: int, batch_size: int) -> Tuple[list, list, list]:
        return self.memory.sample(batch_size, step)

    def update(self, indices: List[int], batchs: List[Any], priorities: List[float]) -> None:
        self.memory.update(indices, batchs, priorities)


if __name__ == "__main__":
    pass
