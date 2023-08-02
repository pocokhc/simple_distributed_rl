import os
import pickle

from srl.algorithms import ql, ql_agent57
from srl.runner.callbacks.checkpoint import Checkpoint
from srl.runner.runner import Runner
from srl.utils import common

common.logger_print()


def test_pickle():
    pickle.loads(pickle.dumps(Checkpoint()))


def test_train():
    rl_config = ql_agent57.Config(memory_warmup_size=10, batch_size=2)
    runner = Runner("OX", rl_config)

    runner.set_save_dir("tmp_test")
    runner.set_checkpoint(interval=1)
    runner.train(timeout=3)

    path = os.path.join(runner.context.save_dir, "checkpoints")
    assert len(os.listdir(path)) > 0


def test_train_only():
    rl_config = ql_agent57.Config(memory_warmup_size=10, batch_size=2)
    runner = Runner("Grid", rl_config)

    runner.train(timeout=1, disable_trainer=True)
    assert runner.remote_memory.length() > rl_config.memory_warmup_size

    runner.set_save_dir("tmp_test")
    runner.set_checkpoint(interval=1)
    runner.train_only(timeout=3)

    path = os.path.join(runner.context.save_dir, "checkpoints")
    assert len(os.listdir(path)) > 0


def test_mp():
    runner = Runner("Grid", ql.Config())

    runner.set_save_dir("tmp_test")
    runner.set_checkpoint(interval=1)
    runner.train_mp(timeout=3)

    path = os.path.join(runner.context.save_dir, "checkpoints")
    assert len(os.listdir(path)) > 0
