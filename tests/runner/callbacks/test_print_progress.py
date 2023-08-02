import pickle

from srl.algorithms import ql, ql_agent57
from srl.runner.callbacks.print_progress import PrintProgress
from srl.runner.runner import Runner
from srl.utils import common

common.logger_print()


def test_pickle():
    pickle.loads(pickle.dumps(PrintProgress()))


def test_train():
    rl_config = ql_agent57.Config(memory_warmup_size=10, batch_size=2)
    runner = Runner("OX", rl_config)

    runner.train(
        timeout=7,
        enable_progress=True,
        progress_start_time=1,
        progress_env_info=True,
        enable_eval=True,
    )


def test_train_only():
    rl_config = ql_agent57.Config(memory_warmup_size=10, batch_size=2)
    runner = Runner("Grid", rl_config)

    runner.train(
        timeout=1,
        disable_trainer=True,
        enable_progress=True,
        progress_start_time=1,
        progress_env_info=True,
        enable_eval=True,
    )
    assert runner.remote_memory.length() > rl_config.memory_warmup_size

    runner.train_only(
        timeout=7,
        enable_progress=True,
    )


def test_mp():
    runner = Runner("Grid", ql.Config())

    callback = PrintProgress(
        start_time=1,
        progress_env_info=True,
        enable_eval=True,
    )

    runner.train_mp(timeout=5, enable_progress=False, callbacks=[callback])
