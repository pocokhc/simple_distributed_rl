import os
import pickle

import pytest

from srl.base.rl.config import DummyRLConfig
from srl.base.rl.memory import RLMemory


def _assert_memory_mp(memory: RLMemory, bach_size: int, tmpdir):
    # --- send funcs
    worker_funcs = memory.get_worker_funcs()
    for _ in range(100):  # warmup以上データをためる
        for k, (func, serialize_func) in worker_funcs.items():
            assert isinstance(k, str)
            batch = (1, 2, 3, 4)
            if serialize_func is None:
                raw = pickle.dumps(batch)
                dat = pickle.loads(raw)
                func(dat)
            else:
                raw = serialize_func(batch)
                raw = raw if isinstance(raw, tuple) else (raw,)
                func(*raw, serialized=True)
    assert memory.length() == len(worker_funcs) * 100

    # --- memory -> trainer
    def assert_sample():
        for func in memory.get_trainer_recv_funcs():
            batch = func()
            batch = pickle.loads(pickle.dumps(batch))
            if bach_size > 0:
                for i in range(bach_size):
                    if memory.__class__.__name__ == "RLPriorityReplayBuffer":
                        assert batch[0][i] == (1, 2, 3, 4)
                    else:
                        assert batch[i] == (1, 2, 3, 4)

    assert_sample()

    # --- trainer -> memory
    for k, func in memory.get_trainer_send_funcs().items():
        assert isinstance(k, str)
        assert callable(func)

    # --- backup/restore
    memory.restore(memory.backup(compress=False))
    assert_sample()
    memory.restore(memory.backup(compress=True))
    assert_sample()

    # --- save/load
    memory.save(os.path.join(tmpdir, "tmp"), compress=False)
    memory.load(os.path.join(tmpdir, "tmp"))
    assert_sample()
    memory.save(os.path.join(tmpdir, "tmp"), compress=True)
    memory.load(os.path.join(tmpdir, "tmp"))
    assert_sample()


def test_single_use_buffer(tmpdir):
    from srl.rl.memories.single_use_buffer import RLSingleUseBuffer

    memory = RLSingleUseBuffer(DummyRLConfig())
    _assert_memory_mp(memory, -1, tmpdir)


@pytest.mark.parametrize("compress", [False, True])
def test_replay_buffer(tmpdir, compress):
    from srl.rl.memories.replay_buffer import ReplayBufferConfig, RLReplayBuffer

    rl_cfg = DummyRLConfig()
    batch_size = 5
    rl_cfg.batch_size = batch_size
    rl_cfg.memory = ReplayBufferConfig(warmup_size=10, compress=compress)

    memory = RLReplayBuffer(rl_cfg)
    _assert_memory_mp(memory, batch_size, tmpdir)


@pytest.mark.parametrize("compress", [False, True])
def test_priority_replay_buffer(tmpdir, compress):
    from srl.rl.memories.priority_replay_buffer import PriorityReplayBufferConfig, RLPriorityReplayBuffer

    rl_cfg = DummyRLConfig()
    batch_size = 5
    rl_cfg.batch_size = batch_size
    rl_cfg.memory = PriorityReplayBufferConfig(warmup_size=10, compress=compress).set_replay_buffer()

    memory = RLPriorityReplayBuffer(rl_cfg)
    _assert_memory_mp(memory, batch_size, tmpdir)


@pytest.mark.parametrize("compress", [False, True])
def test_priority_proportional_memory(tmpdir, compress):
    from srl.rl.memories.priority_replay_buffer import PriorityReplayBufferConfig, RLPriorityReplayBuffer

    rl_cfg = DummyRLConfig()
    batch_size = 5
    rl_cfg.batch_size = batch_size
    rl_cfg.memory = PriorityReplayBufferConfig(warmup_size=10, compress=compress).set_proportional()

    memory = RLPriorityReplayBuffer(rl_cfg)
    _assert_memory_mp(memory, batch_size, tmpdir)


@pytest.mark.parametrize("compress", [False, True])
def test_priority_rankbased_memory(tmpdir, compress):
    from srl.rl.memories.priority_replay_buffer import PriorityReplayBufferConfig, RLPriorityReplayBuffer

    rl_cfg = DummyRLConfig()
    batch_size = 5
    rl_cfg.batch_size = batch_size
    rl_cfg.memory = PriorityReplayBufferConfig(warmup_size=10, compress=compress).set_rankbased()

    memory = RLPriorityReplayBuffer(rl_cfg)
    _assert_memory_mp(memory, batch_size, tmpdir)


@pytest.mark.parametrize("compress", [False, True])
def test_priority_rankbased_memory_linear(tmpdir, compress):
    from srl.rl.memories.priority_replay_buffer import PriorityReplayBufferConfig, RLPriorityReplayBuffer

    rl_cfg = DummyRLConfig()
    batch_size = 5
    rl_cfg.batch_size = batch_size
    rl_cfg.memory = PriorityReplayBufferConfig(warmup_size=10, compress=compress).set_rankbased_linear()

    memory = RLPriorityReplayBuffer(rl_cfg)
    _assert_memory_mp(memory, batch_size, tmpdir)
