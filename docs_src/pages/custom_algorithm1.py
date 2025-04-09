from srl.base.rl.config import DummyRLConfig
from srl.rl.memories.single_use_buffer import RLSingleUseBuffer


class MyRemoteMemory(RLSingleUseBuffer):
    pass


memory = MyRemoteMemory(DummyRLConfig())
memory.add([1, 2])
memory.add([2, 3])
memory.add([3, 4])
dat = memory.sample()
print(dat)  # [[1, 2], [2, 3], [3, 4]]
