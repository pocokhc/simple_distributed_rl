from srl import runner
from srl.algorithms import ql, ql_agent57
from srl.envs import grid  # noqa F401
from srl.runner.callbacks.print_progress import PrintProgress
from srl.runner.core import play
from srl.utils import common

common.logger_print()


def test_train():
    rl_config = ql_agent57.Config(memory_warmup_size=10, batch_size=2)
    config = runner.Config("Grid", rl_config)
    callback = PrintProgress(
        print_start_time=1,
        print_env_info=True,
    )

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
        callbacks=[callback],
    )


def test_mp():
    rl_config = ql.Config()
    config = runner.Config("Grid", rl_config)
    runner.train_mp(config, timeout=5)
