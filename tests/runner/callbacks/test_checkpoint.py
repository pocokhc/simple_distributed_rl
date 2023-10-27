import os
import pickle

import srl
from srl.algorithms import ql, ql_agent57
from srl.runner.callbacks.checkpoint import Checkpoint


def test_pickle():
    pickle.loads(pickle.dumps(Checkpoint()))


def test_train():
    rl_config = ql_agent57.Config(batch_size=2)
    rl_config.memory.warmup_size = 10
    runner = srl.Runner("OX", rl_config)

    runner.set_wkdir("tmp_test")
    runner.set_checkpoint(interval=1)
    runner.train(timeout=3)

    path = os.path.join(runner.context.wkdir, "checkpoints")
    assert len(os.listdir(path)) > 0


def test_train_only():
    rl_config = ql_agent57.Config(batch_size=2)
    rl_config.memory.warmup_size = 10
    runner = srl.Runner("Grid", rl_config)

    runner.train(timeout=1, disable_trainer=True)
    assert runner.memory is not None
    assert runner.memory.length() > rl_config.memory.warmup_size

    runner.set_wkdir("tmp_test")
    runner.set_checkpoint(interval=1)
    runner.train_only(timeout=3)

    path = os.path.join(runner.context.wkdir, "checkpoints")
    assert len(os.listdir(path)) > 0


def test_mp():
    runner = srl.Runner("Grid", ql.Config())

    runner.set_wkdir("tmp_test")
    runner.set_checkpoint(interval=1)
    runner.train_mp(timeout=3)

    path = os.path.join(runner.context.wkdir, "checkpoints")
    assert len(os.listdir(path)) > 0
