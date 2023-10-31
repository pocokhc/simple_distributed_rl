import os
import pickle
import shutil

import srl
from srl.algorithms import ql, ql_agent57
from srl.runner.callbacks.checkpoint import Checkpoint


def test_pickle():
    pickle.loads(pickle.dumps(Checkpoint()))


def test_train(tmp_path):
    rl_config = ql_agent57.Config(batch_size=2)
    rl_config.memory.warmup_size = 10
    runner = srl.Runner("OX", rl_config)

    runner.setup_wkdir(tmp_path)
    runner.set_checkpoint(interval=1)
    runner.train(timeout=3)

    path = os.path.join(runner.config.wkdir1, "checkpoint")
    assert len(os.listdir(path)) > 0


def test_train_only(tmp_path):
    rl_config = ql_agent57.Config(batch_size=2)
    rl_config.memory.warmup_size = 10
    runner = srl.Runner("Grid", rl_config)

    runner.rollout(max_memory=100)
    assert runner.memory is not None
    assert runner.memory.length() > rl_config.memory.warmup_size

    runner.setup_wkdir(tmp_path)
    runner.set_checkpoint(interval=1)
    runner.train_only(timeout=3)

    path = os.path.join(runner.config.wkdir1, "checkpoint")
    assert len(os.listdir(path)) > 0


def test_mp(tmp_path):
    runner = srl.Runner("Grid", ql.Config())

    runner.setup_wkdir(tmp_path)
    runner.set_checkpoint(interval=1)
    runner.train_mp(timeout=3)

    path = os.path.join(runner.config.wkdir1, "checkpoint")
    assert len(os.listdir(path)) > 0
