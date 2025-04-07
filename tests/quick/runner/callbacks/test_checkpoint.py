import os
import pickle

import numpy as np
import pytest

import srl
from srl.algorithms import ql, ql_agent57
from srl.runner.callbacks.checkpoint import Checkpoint


def test_pickle():
    pickle.loads(pickle.dumps(Checkpoint()))


def test_train(tmp_path):
    rl_config = ql_agent57.Config(batch_size=2)
    rl_config.memory.warmup_size = 10
    runner = srl.Runner("OX", rl_config)

    runner.set_checkpoint(tmp_path, is_load=False, interval=1)
    runner.train(timeout=3)

    assert len(os.listdir(tmp_path)) > 0


def test_train_load(tmp_path):
    rl_config = ql.Config()
    runner = srl.Runner("Grid", rl_config)

    runner.set_checkpoint(tmp_path, is_load=False, interval=1)
    runner.train(max_train_count=10_000)
    assert np.mean(runner.evaluate(1000)) > 0.6

    runner.set_checkpoint(tmp_path, is_load=True, interval=1)
    runner.train(max_train_count=1)
    assert np.mean(runner.evaluate(1000)) > 0.6


def test_train_only(tmp_path):
    rl_config = ql_agent57.Config(batch_size=2)
    rl_config.memory.warmup_size = 10
    runner = srl.Runner("Grid", rl_config)

    runner.rollout(max_memory=100)
    assert runner.memory is not None
    assert runner.memory.length() > rl_config.memory.warmup_size

    runner.set_checkpoint(tmp_path, is_load=False, interval=1)
    runner.train_only(timeout=3)

    assert len(os.listdir(tmp_path)) > 0


@pytest.mark.parametrize("enable_mp_memory", [False, True])
def test_mp(tmp_path, enable_mp_memory):
    runner = srl.Runner("Grid", ql.Config())

    runner.set_checkpoint(tmp_path, is_load=False, interval=1)
    runner.train_mp(timeout=3, enable_mp_memory=enable_mp_memory)

    assert len(os.listdir(tmp_path)) > 0
