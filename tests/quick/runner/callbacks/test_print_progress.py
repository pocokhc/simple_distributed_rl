import pickle

import srl
from srl.algorithms import ql, ql_agent57
from srl.runner.callbacks.print_progress import PrintProgress


def test_pickle():
    pickle.loads(pickle.dumps(PrintProgress()))


def test_train():
    rl_config = ql_agent57.Config(batch_size=2)
    rl_config.memory.warmup_size = 10
    runner = srl.Runner("OX", rl_config)

    runner.set_progress_options(start_time=1, interval_limit=1, env_info=True, enable_eval=True)
    runner.train(timeout=3, enable_progress=True)


def test_train_only():
    rl_config = ql_agent57.Config(batch_size=2)
    rl_config.memory.warmup_size = 10
    runner = srl.Runner("Grid", rl_config)

    runner.set_progress_options(start_time=1, interval_limit=1, env_info=True)
    runner.rollout(timeout=3, enable_progress=True)
    assert runner.memory is not None
    assert runner.memory.length() > rl_config.memory.warmup_size

    runner.set_progress_options(start_time=1, interval_limit=1)
    runner.train_only(timeout=3, enable_progress=True)


def test_mp():
    runner = srl.Runner("Grid", ql.Config())

    callback = PrintProgress(
        start_time=1,
        progress_env_info=True,
        enable_eval=True,
    )

    runner.train_mp(timeout=3, enable_progress=False, callbacks=[callback])
