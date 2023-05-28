import os
import shutil

import pytest

from srl import runner
from srl.algorithms import dqn, ql
from srl.envs import grid, ox  # noqa F401
from srl.runner.core import EvalOption, HistoryOption
from srl.utils import common

common.logger_print()


def create_columns_seq(
    player_num,
    env_infos,
    trainer_infos,
    worker0_infos,
    is_eval,
):
    columns = [
        "index",
        "time",
        "train",
        "remote_memory",
        "actor0_episode",
        "actor0_episode_step",
        "actor0_episode_time",
        # --- system
        # "memory",
        # "cpu",
        # "gpu",
        # "gpu_memory",
    ]
    for n in range(player_num):
        columns.append(f"actor0_episode_reward{n}")

    if is_eval:
        for n in range(player_num):
            columns.append(f"actor0_eval_reward{n}")

    for k in env_infos:
        columns.append(f"env_{k}")
    for k in trainer_infos:
        columns.append(f"trainer_{k}")
    for k in worker0_infos:
        columns.append(f"actor0_worker0_{k}")

    return columns


def create_columns_dist(
    actor_id,
    player_num,
    env_infos,
    trainer_infos,
    worker0_infos,
    is_eval,
):
    columns = [
        "index",
        "time_trainer",
        "time_system",
        "train",
        "train_time",
        "train_sync",
        "remote_memory",
        # --- system
        # "memory",
        # "cpu",
        # "gpu",
        # "gpu_memory",
    ]
    for id in range(actor_id):
        columns.extend(
            [
                f"time_actor{id}",
                f"actor{id}_episode",
                f"actor{id}_episode_step",
                f"actor{id}_episode_time",
            ]
        )
        for n in range(player_num):
            columns.append(f"actor{id}_episode_reward{n}")
        for k in worker0_infos:
            columns.append(f"actor{id}_worker0_{k}")

    if is_eval:
        for n in range(player_num):
            columns.append(f"actor0_eval_reward{n}")

    for k in env_infos:
        columns.append(f"env_{k}")
    for k in trainer_infos:
        columns.append(f"trainer_{k}")

    return columns


# ----------------------------------------------------


def test_memory_train():
    pytest.importorskip("pandas")
    pytest.importorskip("matplotlib")

    rl_config = ql.Config()
    config = runner.Config("OX", rl_config)

    _, _, history = runner.train(
        config,
        timeout=2,
        enable_profiling=True,
        eval=EvalOption(),
        history=HistoryOption(write_memory=True, write_file=False),
        checkpoint=None,
    )
    df = history.get_df()
    print(df.tail().T)

    for n in create_columns_seq(
        config.env_config.player_num,
        [],
        ["size", "td_error"],
        ["epsilon"],
        True,
    ):
        assert n in df

    assert history.player_num == 2
    assert history.actor_num == 1
    assert not history.distributed

    # --- plot test
    # history.plot()
    history.plot(_no_plot=True)


@pytest.mark.skip(reason="TODO")
def test_memory_train_only():
    pytest.importorskip("pandas")
    pytest.importorskip("matplotlib")

    rl_config = dqn.Config()
    config = runner.Config("OX", rl_config)

    _, memory, _ = runner.train(config, max_train_count=10, enable_profiling=False)
    _, _, history = runner.train_only(
        config,
        memory,
        timeout=2,
        eval=EvalOption(),
        history=HistoryOption(write_memory=True, write_file=False),
    )

    df = history.get_df()
    print(df.tail().T)

    for n in create_columns_seq(
        config.env_config.player_num,
        [],
        ["size", "td_error"],
        ["epsilon"],
        True,
    ):
        assert n in df

    assert history.player_num == 2
    assert history.actor_num == 1
    assert not history.distributed

    # --- plot test
    # history.plot()
    history.plot(_no_plot=True)


def test_file_train():
    pytest.importorskip("pandas")

    dir_name = "tmp_test"
    if os.path.isdir(dir_name):
        shutil.rmtree(dir_name)

    rl_config = ql.Config()
    config = runner.Config("OX", rl_config, base_dir=dir_name)

    _, _, history = runner.train(
        config,
        timeout=2,
        enable_profiling=True,
        eval=EvalOption(),
        history=HistoryOption(write_memory=False, write_file=True),
        checkpoint=None,
    )
    df = history.get_df()
    print(df.tail().T)

    for n in create_columns_seq(
        config.env_config.player_num,
        [],
        ["size", "td_error"],
        ["epsilon"],
        True,
    ):
        assert n in df

    assert history.player_num == 2
    assert history.actor_num == 1
    assert not history.distributed

    # --- plot test
    # history.plot()
    history.plot(_no_plot=True)


def test_file_train_mp():
    pytest.importorskip("pandas")

    dir_name = "tmp_test"
    if os.path.isdir(dir_name):
        shutil.rmtree(dir_name)

    rl_config = dqn.Config()
    config = runner.Config("Grid", rl_config, base_dir=dir_name, actor_num=2)
    _, _, history = runner.train_mp(
        config,
        timeout=5,
        eval=EvalOption(),
        history=HistoryOption(write_memory=False, write_file=True),
    )
    df = history.get_df()
    print(df.tail().T)

    for n in create_columns_dist(
        config.actor_num,
        config.env_config.player_num,
        [],
        ["loss", "sync"],
        ["epsilon"],
        True,
    ):
        assert n in df

    assert history.player_num == 1
    assert history.actor_num == 2
    assert history.distributed

    # --- plot test
    # history.plot()
    history.plot(_no_plot=True)
