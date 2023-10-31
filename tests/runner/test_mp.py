import numpy as np
import pytest

import srl
from srl.algorithms import ql_agent57
from srl.runner.callback import Callback, TrainerCallback


class _AssertTrainCallbacks(Callback, TrainerCallback):
    def on_episodes_end(self, runner: srl.Runner) -> None:
        assert runner.state.sync_actor > 1

    def on_trainer_end(self, runner: srl.Runner) -> None:
        assert runner.state.sync_trainer > 1


@pytest.mark.parametrize("enable_prepare_sample_batch", [False, True])
def test_train(enable_prepare_sample_batch):
    rl_config = ql_agent57.Config(batch_size=2)
    rl_config.memory.warmup_size = 10
    runner = srl.Runner("Grid", rl_config)
    runner.train_mp(
        actor_num=2,
        max_train_count=10_000,
        enable_eval=True,
        enable_prepare_sample_batch=enable_prepare_sample_batch,
        callbacks=[_AssertTrainCallbacks()],
        trainer_parameter_send_interval=1,
        actor_parameter_sync_interval=1,
    )

    # eval
    rewards = runner.evaluate(max_episodes=100)
    rewards = np.mean(rewards)
    print(rewards)
    assert rewards > 0.5
