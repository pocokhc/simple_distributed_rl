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


def test_on_memory_train():
    runner = srl.Runner("OX", ql.Config())

    runner.set_history(
        enable_history=True,
        enable_eval=True,
        write_memory=True,
        write_file=False,
    )
    runner.train(max_episodes=100)

    history = runner.get_history()
    pprint(history.logs[-1])
    assert len(history.logs) == 100
    assert history.logs[0]["name"] == "actor0"
    assert history.logs[0]["time"] > 0
    assert history.logs[0]["episode_time"] > 0
    assert history.logs[0]["reward0"] > -2
    assert history.logs[0]["reward1"] > -2
    assert history.logs[0]["eval_reward0"] > -2
    assert history.logs[0]["eval_reward1"] > -2
    assert history.logs[0]["memory"] >= 0
    for i, h in enumerate(history.logs):
        assert h["episode"] == i


def test_on_memory_train_plot():
    pytest.importorskip("pandas")
    pytest.importorskip("matplotlib")

    runner = srl.Runner("OX", ql.Config())

    runner.set_history(
        enable_history=True,
        enable_eval=True,
        write_memory=True,
        write_file=False,
    )
    runner.train(max_episodes=100)
    history = runner.get_history()

    df = history.get_df()
    print(df.tail().T)

    # --- plot test
    # history.plot()
    history.plot(_no_plot=True)


def test_on_memory_train_only():
    runner = srl.Runner("OX", ql_agent57.Config())
    runner.train(max_episodes=100, disable_trainer=True)

    runner.set_history(
        enable_history=True,
        enable_eval=True,
        write_memory=True,
        write_file=False,
    )
    runner.train_only(timeout=3)

    history = runner.get_history()
    pprint(history.logs[-1])
    assert len(history.logs) > 1
    assert history.logs[0]["name"] == "trainer"
    assert history.logs[0]["time"] > 0
    assert history.logs[-1]["train"] > 0
    assert history.logs[0]["train_time"] > 0
    assert history.logs[0]["trainer_ext_td_error"] > -99
    assert history.logs[0]["trainer_int_td_error"] > -99
    assert history.logs[0]["trainer_size"] > 0


def test_on_memory_train_only_plot():
    pytest.importorskip("pandas")
    pytest.importorskip("matplotlib")

    runner = srl.Runner("OX", ql_agent57.Config())
    runner.train(max_episodes=100, disable_trainer=True)

    runner.set_history(
        enable_history=True,
        enable_eval=True,
        write_memory=True,
        write_file=False,
    )
    runner.train_only(timeout=5)
    history = runner.get_history()

    df = history.get_df()
    print(df.tail().T)

    # --- plot test
    # history.plot(ylabel_left=["train"])
    history.plot(ylabel_left=["train"], _no_plot=True)


def test_on_file_train():
    runner = srl.Runner("OX", ql.Config())

    runner.set_wkdir("tmp_test")
    runner.set_history(
        enable_history=True,
        enable_eval=True,
        write_memory=False,
        write_file=True,
    )
    runner.train(timeout=5)

    history = runner.get_history()
    pprint(history.logs[-1])
    assert len(history.logs) > 0
    for i, h in enumerate(history.logs):
        assert h["name"] == "actor0"
        assert h["time"] > 0
        assert h["episode_time"] >= 0
        assert h["reward0"] > -2
        assert h["reward1"] > -2
        assert h["eval_reward0"] > -2
        assert h["eval_reward1"] > -2
        assert h["memory"] >= 0
        assert h["episode"] > 0


def test_on_file_train_plot():
    pytest.importorskip("pandas")
    pytest.importorskip("matplotlib")

    runner = srl.Runner("OX", ql.Config())

    runner.set_wkdir("tmp_test")
    runner.set_history(
        enable_history=True,
        enable_eval=True,
        write_memory=False,
        write_file=True,
    )
    runner.train(timeout=5)
    history = runner.get_history()

    df = history.get_df()
    print(df.tail().T)

    # --- plot test
    # history.plot()
    history.plot(_no_plot=True)


def test_on_file_train_only():
    runner = srl.Runner("OX", ql_agent57.Config())
    runner.train(max_episodes=100, disable_trainer=True)

    runner.set_wkdir("tmp_test")
    runner.set_history(
        enable_history=True,
        enable_eval=True,
        write_memory=False,
        write_file=True,
    )
    runner.train_only(timeout=3)

    history = runner.get_history()
    pprint(history.logs[-1])
    assert len(history.logs) > 1
    assert history.logs[0]["name"] == "trainer"
    assert history.logs[0]["time"] > 0
    assert history.logs[-1]["train"] > 0
    assert history.logs[0]["train_time"] > 0
    assert history.logs[0]["trainer_ext_td_error"] > -99
    assert history.logs[0]["trainer_int_td_error"] > -99
    assert history.logs[0]["trainer_size"] > 0


def test_on_file_train_only_plot():
    pytest.importorskip("pandas")
    pytest.importorskip("matplotlib")

    runner = srl.Runner("OX", ql_agent57.Config())
    runner.train(max_episodes=100, disable_trainer=True)

    runner.set_wkdir("tmp_test")
    runner.set_history(
        enable_history=True,
        enable_eval=True,
        write_memory=False,
        write_file=True,
    )
    runner.train_only(timeout=5)
    history = runner.get_history()

    df = history.get_df()
    print(df.tail().T)

    # --- plot test
    # history.plot(ylabel_left=["train"])
    history.plot(ylabel_left=["train"], _no_plot=True)


def test_on_file_mp():
    runner = srl.Runner("OX", ql.Config())

    runner.set_wkdir("tmp_test")
    runner.set_history(
        enable_history=True,
        enable_eval=True,
        write_memory=False,
        write_file=True,
    )
    runner.train_mp(timeout=5)

    history = runner.get_history()
    pprint(history.logs[-1])
    assert len(history.logs) > 0
    for i, h in enumerate(history.logs):
        if h["name"] == "actor0":
            assert h["time"] > 0
            assert h["episode_time"] >= 0
            assert h["episode"] > 0
        elif h["name"] == "trainer":
            assert h["train"] >= 0
        else:
            assert False


def test_on_file_mp_plot():
    pytest.importorskip("pandas")
    pytest.importorskip("matplotlib")

    runner = srl.Runner("OX", ql.Config())

    runner.set_wkdir("tmp_test")
    runner.set_history(
        enable_history=True,
        enable_eval=True,
        write_memory=False,
        write_file=True,
    )
    runner.train_mp(timeout=5)
    history = runner.get_history()

    df = history.get_df()
    print(df.tail().T)

    # --- plot test
    # history.plot()
    history.plot(_no_plot=True)
