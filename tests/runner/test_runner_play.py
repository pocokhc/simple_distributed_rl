import io
import time

import numpy as np
import pytest

import srl
from srl.algorithms import ql, ql_agent57
from srl.runner.runner import Runner
from srl.utils.common import is_available_pygame_video_device


def test_train():
    runner = Runner("OX", ql.Config())

    runner.train(max_episodes=10)
    assert runner.state.episode_count == 10

    t0 = time.time()
    runner.train(timeout=1)
    assert time.time() - t0 >= 1

    runner.train(max_steps=10)
    assert runner.state.total_step == 10

    runner.train(max_train_count=10)
    assert runner.state.trainer is not None
    assert runner.state.trainer.get_train_count() == 10


def test_train_multi_runner():
    runner1 = Runner("Grid", ql.Config())
    runner2 = Runner("OX", ql.Config())
    runner3 = Runner("OX", ql.Config())

    runner1.train_mp(max_train_count=10)
    runner1.train(max_train_count=10)

    runner3.train(max_train_count=10)
    runner3.train_mp(max_train_count=10)
    runner2.train(max_train_count=10)
    runner1.train(max_train_count=10)


def test_train_only():
    rl_config = ql_agent57.Config()
    runner = Runner("Grid", rl_config)

    runner.train(max_steps=10_000, disable_trainer=True)
    assert runner.remote_memory.length() > 1000

    rl_config.memory_warmup_size = 1000
    runner.train_only(max_train_count=50_000)

    rewards = runner.evaluate(max_episodes=100)
    reward = np.mean(rewards)
    assert reward > 0.5, f"reward: {reward}"


def test_render_terminal():
    env_config = srl.EnvConfig("Grid")
    rl_config = ql.Config()
    runner = Runner(env_config, rl_config)

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
    runner = Runner(env_config, rl_config)

    # train
    runner.train(max_steps=20000)

    # render terminal
    reward = runner.render_window(render_interval=1)
    print(reward)
    assert reward[0] > 0.5


def test_animation():
    pytest.importorskip("cv2")
    pytest.importorskip("PIL")
    pytest.importorskip("pygame")

    env_config = srl.EnvConfig("Grid")
    rl_config = ql.Config()
    runner = Runner(env_config, rl_config)

    runner.train(max_steps=20000)
    runner.animation_save_gif("tmp_test/a.gif", max_steps=10)


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

    env_config = srl.EnvConfig("Grid")
    rl_config = ql.Config()
    runner = Runner(env_config, rl_config)

    runner.replay_window(_is_test=True)


def test_play_terminal(monkeypatch):
    # 標準入力をモック
    monkeypatch.setattr("sys.stdin", io.StringIO("0\n1\n2\n3\n"))

    runner = Runner("Grid")
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

    runner = Runner("Grid")
    runner.play_window(_is_test=True)


def test_gym():
    pytest.importorskip("cv2")
    pytest.importorskip("PIL")
    pytest.importorskip("pygame")
    pytest.importorskip("gym")

    runner = Runner("MountainCar-v0")
    runner.animation_save_gif("tmp_test/b.gif", max_steps=10)


def test_gymnasium():
    pytest.importorskip("cv2")
    pytest.importorskip("PIL")
    pytest.importorskip("pygame")
    pytest.importorskip("gymnasium")

    runner = Runner("MountainCar-v0", None)
    runner.animation_save_gif("tmp_test/c.gif", max_steps=10)


def test_shuffle_player():
    runner = Runner("OX")
    runner.set_seed(1)
    runner.set_players(["cpu", "random"])

    # shuffle した状態でも報酬は元の順序を継続する
    rewards = runner.evaluate(max_episodes=100, shuffle_player=True)
    rewards = np.mean(rewards, axis=0)
    assert rewards[0] > 0.7  # CPUがまず勝つ
