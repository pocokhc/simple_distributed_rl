import numpy as np
import pytest

import srl
from srl.algorithms import ql, ql_agent57
from srl.runner.callback import Callback, TrainerCallback


class _AssertTrainCallbacks(Callback, TrainerCallback):
    def on_episodes_end(self, runner: srl.Runner) -> None:
        assert runner.state.sync_actor > 1

    def on_trainer_end(self, runner: srl.Runner) -> None:
        assert runner.state.sync_trainer > 1


@pytest.mark.parametrize("enable_prepare_sample_batch", [False, True])
def test_train(enable_prepare_sample_batch):
    runner = srl.Runner("Grid", ql.Config())
    runner.train_mp_debug(
        actor_num=2,
        max_train_count=10_000,
        enable_eval=True,
        enable_prepare_sample_batch=enable_prepare_sample_batch,
        callbacks=[_AssertTrainCallbacks()],
        trainer_parameter_send_interval=0,
        actor_parameter_sync_interval=0,
    )

    # eval
    rewards = runner.evaluate(max_episodes=100)
    rewards = np.mean(rewards)
    print(rewards)
    assert rewards > 0.5


def test_train2():
    rl_config = ql_agent57.Config()
    rl_config.memory.capacity = 1000
    rl_config.memory.set_replay_memory()
    runner = srl.Runner("Grid", rl_config)
    runner.train(max_episodes=5000)

    assert runner.memory is not None
    memory_len = runner.memory.length()
    runner.memory.memory.capacity = 1100  # type: ignore , 直接変更

    runner.train_mp_debug(max_episodes=10)
    memory2 = runner.memory

    # eval
    rewards = runner.evaluate(max_episodes=100)
    rewards = np.mean(rewards)
    print(rewards)
    assert rewards > 0.5

    assert memory2 is not None
    print(memory2.length(), memory_len)
    assert memory2.length() > memory_len
