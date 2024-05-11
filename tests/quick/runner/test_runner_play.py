import io
import os
import time

import numpy as np
import pytest

import srl
from srl.algorithms import ql, ql_agent57
from srl.utils import common
from srl.utils.common import is_available_pygame_video_device


@pytest.mark.parametrize("use_train_thread", [False, True])
def test_train(use_train_thread):
    common.logger_print()

    runner = srl.Runner("OX", ql.Config())

    state = runner.train(max_episodes=10, use_train_thread=use_train_thread)
    assert state.episode_count == 10

    t0 = time.time()
    runner.train(timeout=1, use_train_thread=use_train_thread)
    assert time.time() - t0 >= 0.9

    state = runner.train(max_steps=10, use_train_thread=use_train_thread)
    assert state.total_step == 10

    state = runner.train(max_train_count=10, use_train_thread=use_train_thread)
    assert state.trainer is not None
    if use_train_thread:
        assert state.trainer.get_train_count() > 10
    else:
        assert state.trainer.get_train_count() == 10


@pytest.mark.parametrize("use_train_thread", [False, True])
def test_train_multi_runner(use_train_thread):
    runner1 = srl.Runner("Grid", ql.Config())
    runner2 = srl.Runner("OX", ql.Config())
    runner3 = srl.Runner("OX", ql.Config())

    runner1.train_mp(max_train_count=10, use_train_thread=use_train_thread)
    runner1.train(max_train_count=10, use_train_thread=use_train_thread)

    runner3.train(max_train_count=10, use_train_thread=use_train_thread)
    runner3.train_mp(max_train_count=10, use_train_thread=use_train_thread)
    runner2.train(max_train_count=10, use_train_thread=use_train_thread)
    runner1.train(max_train_count=10, use_train_thread=use_train_thread)


def test_rollout():
    rl_config = ql_agent57.Config()
    runner = srl.Runner("Grid", rl_config)

    runner.rollout(max_memory=100)
    assert runner.memory is not None
    assert runner.memory.length() == 100


@pytest.mark.parametrize("use_train_thread", [False, True])
def test_train_only(use_train_thread):
    rl_config = ql_agent57.Config()
    runner = srl.Runner("Grid", rl_config)

    runner.rollout(max_memory=1010)
    assert runner.memory is not None
    assert runner.memory.length() > 1000

    rl_config.memory.warmup_size = 1000
    runner.train_only(max_train_count=50_000, use_train_thread=use_train_thread, thread_queue_capacity=1000)

    rewards = runner.evaluate(max_episodes=100)
    reward = np.mean(rewards)
    assert reward > 0.5, f"reward: {reward}"


def test_render_terminal():
    env_config = srl.EnvConfig("Grid")
    rl_config = ql.Config()
    runner = srl.Runner(env_config, rl_config)

    # train
    runner.train(max_steps=20000)

    # render terminal
    reward = runner.render_terminal()
    print(reward)
    assert reward[0] > 0.5


def test_render_window():
    """
    docker-low is NG
      Hello from the pygame community. https://www.pygame.org/contribute.html
      Fatal Python error: Segmentation fault
    """
    pytest.importorskip("pygame")
    if not is_available_pygame_video_device():
        pytest.skip("pygame.error: No available video device")

    env_config = srl.EnvConfig("Grid")
    rl_config = ql.Config()
    runner = srl.Runner(env_config, rl_config)

    # train
    runner.train(max_steps=20000)

    # render terminal
    reward = runner.render_window(render_interval=1)
    print(reward)
    assert reward[0] > 0.5


def test_animation(tmp_path):
    pytest.importorskip("cv2")
    pytest.importorskip("PIL")
    pytest.importorskip("pygame")

    runner = srl.Runner("Grid", ql.Config())
    runner.animation_save_gif(os.path.join(tmp_path, "a.gif"), max_steps=10)
    runner.animation_save_avi(os.path.join(tmp_path, "a.avi"), max_steps=10)


def test_replay_window():
    """
    docker-low is NG
      Hello from the pygame community. https://www.pygame.org/contribute.html
      Fatal Python error: Segmentation fault
    """
    pytest.importorskip("cv2")
    pytest.importorskip("PIL")
    pytest.importorskip("pygame")
    if not is_available_pygame_video_device():
        pytest.skip("pygame.error: No available video device")

    runner = srl.Runner("Grid", ql.Config())
    runner.replay_window(_is_test=True)


def test_play_terminal(monkeypatch):
    # 標準入力をモック
    monkeypatch.setattr("sys.stdin", io.StringIO("0\n1\n2\n3\n"))

    runner = srl.Runner("Grid")
    runner.play_terminal(max_steps=3)


def test_play_window():
    """
    docker-low is NG
      Hello from the pygame community. https://www.pygame.org/contribute.html
      Fatal Python error: Segmentation fault
    """
    pytest.importorskip("cv2")
    pytest.importorskip("PIL")
    pytest.importorskip("pygame")
    if not is_available_pygame_video_device():
        pytest.skip("pygame.error: No available video device")

    runner = srl.Runner("Grid")
    runner.play_window(_is_test=True)


def test_gym(tmp_path):
    pytest.importorskip("cv2")
    pytest.importorskip("PIL")
    pytest.importorskip("pygame")
    pytest.importorskip("gym")

    runner = srl.Runner("MountainCar-v0")
    runner.animation_save_gif(os.path.join(tmp_path, "b.gif"), max_steps=10)


def test_gymnasium(tmp_path):
    pytest.importorskip("cv2")
    pytest.importorskip("PIL")
    pytest.importorskip("pygame")
    pytest.importorskip("gymnasium")

    runner = srl.Runner("MountainCar-v0", None)
    runner.animation_save_gif(os.path.join(tmp_path, "b.gif"), max_steps=10)


def test_shuffle_player():
    runner = srl.Runner("OX")
    runner.set_seed(1)
    runner.set_players(["cpu", "random"])

    # shuffle した状態でも報酬は元の順序を継続する
    rewards = runner.evaluate(max_episodes=100, shuffle_player=True)
    rewards = np.mean(rewards, axis=0)
    assert rewards[0] > 0.7  # CPUがまず勝つ
