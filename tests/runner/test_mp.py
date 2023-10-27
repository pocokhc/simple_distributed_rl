import numpy as np

import srl
from srl.algorithms import ql_agent57
from srl.runner.callback import Callback, TrainerCallback


class _AssertTrainCallbacks(Callback, TrainerCallback):
    def on_episodes_end(self, runner: srl.Runner) -> None:
        assert runner.state.sync_actor > 1

    def on_trainer_end(self, runner: srl.Runner) -> None:
        assert runner.state.sync_trainer > 1


def test_train():
    runner = srl.Runner("Grid", ql_agent57.Config(batch_size=2))
    runner.train_mp(actor_num=2, max_train_count=10_000, callbacks=[_AssertTrainCallbacks()])

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
    runner.train(max_episodes=1000)

    assert runner.memory is not None
    memory_len = runner.memory.length()
    runner.memory.memory.capacity = 1100  # type: ignore , 直接変更

    runner.train_mp(timeout=1)
    memory2 = runner.memory

    # eval
    rewards = runner.evaluate(max_episodes=100)
    rewards = np.mean(rewards)
    print(rewards)
    assert rewards > 0.5

    assert memory2 is not None
    print(memory2.length(), memory_len)
    assert memory2.length() > memory_len
