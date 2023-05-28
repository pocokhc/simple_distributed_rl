from srl import runner
from srl.algorithms import ql, ql_agent57
from srl.envs import grid  # noqa F401
from srl.runner.callback import Callback
from srl.runner.callbacks.evaluate import Evaluate
from srl.runner.core import play
from srl.utils import common

common.logger_print()


class _Check(Callback):
    def __init__(self):
        self.last_rewards = []

    # --- Actor
    def on_episode_end(self, info) -> None:
        if "eval_rewards" in info:
            self.last_rewards = info["eval_rewards"]

    # --- Trainer
    def on_trainer_train(self, info) -> None:
        if "eval_rewards" in info:
            self.last_rewards = info["eval_rewards"]


def test_run():
    config = runner.Config("Grid", ql.Config())
    callback = Evaluate()
    check = _Check()
    play(
        config,
        timeout=5,
        train_only=False,
        enable_profiling=False,
        training=True,
        eval=None,
        history=None,
        checkpoint=None,
        callbacks=[callback, check],
    )
    assert check.last_rewards[0] > 0.2


def test_train():
    rl_config = ql_agent57.Config(memory_warmup_size=10, batch_size=2)
    config = runner.Config("Grid", rl_config)
    callback = Evaluate()
    check = _Check()

    parameter, memory, _ = runner.train(config, timeout=1, history=None)
    assert memory.length() > rl_config.memory_warmup_size

    play(
        config,
        max_train_count=5,
        parameter=parameter,
        remote_memory=memory,
        train_only=True,
        enable_profiling=False,
        training=True,
        eval=None,
        history=None,
        checkpoint=None,
        callbacks=[callback, check],
    )
    assert check.last_rewards[0] > -10
