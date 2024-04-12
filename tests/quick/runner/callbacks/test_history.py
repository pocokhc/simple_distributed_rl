import pickle
from pprint import pprint

import pytest

import srl
from srl.algorithms import ql, ql_agent57
from srl.runner.callbacks.history_on_file import HistoryOnFile
from srl.runner.callbacks.history_on_memory import HistoryOnMemory
from srl.runner.callbacks.history_viewer import HistoryViewer


def test_pickle():
    pickle.loads(pickle.dumps(HistoryOnMemory()))
    pickle.loads(pickle.dumps(HistoryOnFile()))
    pickle.loads(pickle.dumps(HistoryViewer()))


@pytest.mark.parametrize("interval_mode", ["step", "time"])
def test_on_memory_train(interval_mode):
    runner = srl.Runner("OX", ql.Config())

    runner.set_history_on_memory(interval_mode=interval_mode, enable_eval=True)
    if interval_mode == "step":
        runner.train(max_steps=10)
    else:
        runner.train(timeout=1.2)

    history = runner.get_history()
    pprint(history.logs[-1])
    if interval_mode == "step":
        assert len(history.logs) == 10 + 1
    assert history.logs[0]["name"] == "actor0"
    assert history.logs[0]["time"] >= 0
    assert history.logs[0]["step"] >= 0
    assert history.logs[0]["episode"] >= 0
    assert history.logs[0]["episode_step"] >= 0
    assert history.logs[0]["episode_time"] >= 0
    assert history.logs[0]["eval_reward0"] > -2
    assert history.logs[0]["eval_reward1"] > -2
    assert history.logs[0]["memory"] >= 0
    for i, h in enumerate(history.logs):
        if h["episode"] > 0:
            assert h["reward0"] > -2
            assert h["reward1"] > -2


def test_on_memory_train_plot():
    pytest.importorskip("pandas")
    pytest.importorskip("matplotlib")

    runner = srl.Runner("OX", ql.Config())

    runner.set_history_on_memory(enable_eval=True)
    runner.train(max_episodes=10)
    history = runner.get_history()

    df = history.get_df()
    print(df.tail().T)

    # --- plot test
    # history.plot()
    history.plot(_for_test=True)


@pytest.mark.parametrize("interval_mode", ["step", "time"])
def test_on_memory_train_only(interval_mode):
    runner = srl.Runner("OX", ql_agent57.Config())
    runner.rollout(max_memory=10)

    runner.set_history_on_memory(interval_mode=interval_mode, enable_eval=True)
    if interval_mode == "step":
        runner.train_only(max_train_count=10)
    else:
        runner.train_only(timeout=1.2)

    history = runner.get_history()
    pprint(history.logs[-1])
    assert len(history.logs) > 1
    assert history.logs[0]["name"] == "trainer"
    assert history.logs[0]["time"] >= 0
    assert history.logs[-1]["train"] > 0
    assert history.logs[0]["train_time"] >= 0
    assert history.logs[0]["trainer_ext_td_error"] > -99
    assert history.logs[0]["trainer_int_td_error"] > -99
    assert history.logs[0]["trainer_size"] > 0


def test_on_memory_train_only_plot():
    pytest.importorskip("pandas")
    pytest.importorskip("matplotlib")

    rl_config = ql_agent57.Config(batch_size=1)
    rl_config.memory.warmup_size = 5
    runner = srl.Runner("OX", rl_config)
    runner.rollout(max_memory=10)

    runner.set_history_on_memory(interval_mode="step", enable_eval=True)
    runner.train_only(max_train_count=5)
    history = runner.get_history()

    df = history.get_df()
    print(df.tail().T)

    # --- plot test
    # history.plot(ylabel_left=["train"])
    history.plot(ylabel_left=["train"], _for_test=True)


@pytest.mark.parametrize("interval_mode", ["step", "time"])
def test_on_file_train(tmp_path, interval_mode):
    runner = srl.Runner("OX", ql.Config())

    runner.set_history_on_file(tmp_path, interval_mode=interval_mode, enable_eval=True, write_system=True)
    if interval_mode == "step":
        runner.train(max_episodes=2)
    else:
        runner.train(timeout=1.2)

    history = runner.get_history()
    pprint(history.logs)
    if interval_mode == "step":
        assert len(history.logs) >= 1
    else:
        assert len(history.logs) == 2 * 2

    # add history
    if interval_mode == "step":
        runner.train(max_episodes=2)
    else:
        runner.train(timeout=1.2)
    history = runner.get_history()
    if interval_mode == "step":
        assert len(history.logs) >= 1
    else:
        assert len(history.logs) == 2 * 4

    # actor
    actor_log = [h for h in history.logs if h["name"] == "actor0"]
    pprint(actor_log)
    if interval_mode == "step":
        assert len(actor_log) >= 1
    else:
        assert len(actor_log) == 4
    for h in actor_log:
        assert h["time"] >= 0
        assert h["train"] >= 0
        assert h["memory"] >= 0
        assert h["step"] >= 0
        assert h["episode"] >= 0
        assert h["episode_time"] >= 0
        assert h["episode_step"] >= 0
        if h["episode"] > 0:
            assert h["episode_time"] >= 0
            assert h["reward0"] > -2
            assert h["reward1"] > -2
            assert h["eval_reward0"] > -2
            assert h["eval_reward1"] > -2

    # system
    system_log = [h for h in history.logs if h["name"] == "system"]
    if interval_mode == "step":
        assert len(system_log) >= 1
    else:
        assert len(system_log) == 4
    for h in system_log:
        assert h["time"] > 0


def test_on_file_train_plot(tmp_path):
    pytest.importorskip("pandas")
    pytest.importorskip("matplotlib")

    runner = srl.Runner("OX", ql.Config())

    runner.set_history_on_file(tmp_path, interval_mode="step", enable_eval=True)
    runner.train(max_train_count=5)
    history = runner.get_history()

    df = history.get_df()
    print(df.tail().T)

    # --- plot test
    # history.plot()
    history.plot(_for_test=True)


@pytest.mark.parametrize("interval_mode", ["step", "time"])
def test_on_file_train_only(tmp_path, interval_mode):
    runner = srl.Runner("OX", ql_agent57.Config(batch_size=1))
    runner.rollout(max_memory=10)

    runner.set_history_on_file(tmp_path, interval_mode=interval_mode, enable_eval=True)
    if interval_mode == "step":
        runner.train_only(max_train_count=10)
    else:
        runner.train_only(timeout=1.2)

    history = runner.get_history()
    pprint(history.logs[-1])
    if interval_mode == "time":
        assert len(history.logs) == 2
    assert history.logs[0]["name"] == "trainer"
    assert history.logs[0]["time"] >= 0
    assert history.logs[-1]["train"] > 0
    assert history.logs[0]["train_time"] >= 0
    assert history.logs[0]["trainer_ext_td_error"] > -99
    assert history.logs[0]["trainer_int_td_error"] > -99
    assert history.logs[0]["trainer_size"] > 0


def test_on_file_train_only_plot(tmp_path):
    pytest.importorskip("pandas")
    pytest.importorskip("matplotlib")

    runner = srl.Runner("OX", ql_agent57.Config(batch_size=1))
    runner.rollout(max_memory=10)

    runner.set_history_on_file(tmp_path, interval_mode="step", enable_eval=True)
    runner.train_only(max_train_count=5)
    history = runner.get_history()

    df = history.get_df()
    print(df.tail().T)

    # --- plot test
    # history.plot(ylabel_left=["train"])
    history.plot(ylabel_left=["train"], _for_test=True)


@pytest.mark.parametrize("interval_mode", ["step", "time"])
def test_on_file_mp(tmp_path, interval_mode):
    runner = srl.Runner("OX", ql.Config())

    runner.set_history_on_file(tmp_path, interval_mode=interval_mode, enable_eval=True)
    if interval_mode == "step":
        runner.train_mp(max_train_count=10)
    else:
        runner.train_mp(timeout=1.2)

    history = runner.get_history()
    pprint(history.logs[-1])
    assert len(history.logs) > 0
    for i, h in enumerate(history.logs):
        if h["name"] == "actor0":
            assert h["time"] >= 0
            if "episode_time" in h:
                assert h["episode_time"] >= 0
                assert h["episode"] >= 0
        elif h["name"] == "trainer":
            assert h["train"] >= 0
        else:
            assert False


def test_on_file_mp_plot(tmp_path):
    pytest.importorskip("pandas")
    pytest.importorskip("matplotlib")

    runner = srl.Runner("OX", ql.Config())

    runner.set_history_on_file(tmp_path, interval_mode="step", enable_eval=True)
    runner.train_mp(max_train_count=5)
    history = runner.get_history()

    df = history.get_df()
    print(df.tail().T)

    # --- plot test
    # history.plot()
    history.plot(_for_test=True)
