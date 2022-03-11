from srl.base.rl.memory import Memory
from srl.rl.memory.proportional_memory import ProportionalMemory
from srl.rl.memory.rankbase_memory import RankBaseMemory
from srl.rl.memory.replay_memory import ReplayMemory


def create(name, kwargs) -> Memory:

    memories = [
        ReplayMemory,
        RankBaseMemory,
        ProportionalMemory,
    ]

    for m in memories:
        if m.getName() == name:
            return m(**kwargs)

    names = [m.getName() for m in memories]
    raise ValueError("Unknown memory({}). Memories is [{}].".format(name, ",".join(names)))
