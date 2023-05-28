from srl.base.rl.remote_memory import SequenceRemoteMemory


class MyRemoteMemory(SequenceRemoteMemory):
    pass


# 実行例
memory = MyRemoteMemory(None)
memory.add([1, 2])
memory.add([2, 3])
memory.add([3, 4])
dat = memory.sample()
print(dat)  # [[1, 2], [2, 3], [3, 4]]
