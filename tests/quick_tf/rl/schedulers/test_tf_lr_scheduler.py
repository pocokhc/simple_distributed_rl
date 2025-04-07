import pytest

from srl.rl.schedulers.lr_scheduler import LRSchedulerConfig


def test_constant_scheduler():
    pytest.importorskip("tensorflow")
    config = LRSchedulerConfig().set_constant()
    assert config.apply_tf_scheduler(0.01) == 0.01


def test_step_scheduler():
    pytest.importorskip("tensorflow")
    from tensorflow import keras

    config = LRSchedulerConfig().set_step(50000, 0.5)
    scheduler = config.apply_tf_scheduler(0.1)
    assert isinstance(scheduler, keras.optimizers.schedules.ExponentialDecay)
    assert scheduler.initial_learning_rate == 0.1
    assert scheduler.decay_steps == 50000
    assert scheduler.decay_rate == 0.5
    assert scheduler.staircase is True


def test_exp_scheduler():
    pytest.importorskip("tensorflow")
    from tensorflow import keras

    config = LRSchedulerConfig().set_exp(75000, 0.9)
    scheduler = config.apply_tf_scheduler(0.1)
    assert isinstance(scheduler, keras.optimizers.schedules.ExponentialDecay)
    assert scheduler.initial_learning_rate == 0.1
    assert scheduler.decay_steps == 75000
    assert scheduler.decay_rate == 0.9
    assert scheduler.staircase is False


def test_cosine_scheduler():
    pytest.importorskip("tensorflow")
    from tensorflow import keras

    config = LRSchedulerConfig().set_cosine(100000, 1e-5)
    scheduler = config.apply_tf_scheduler(0.1)
    assert isinstance(scheduler, keras.optimizers.schedules.CosineDecay)
    assert scheduler.initial_learning_rate == 0.1
    assert scheduler.decay_steps == 100000
    assert abs(scheduler.alpha - (1e-5 / 0.1)) < 1e-9


def test_piecewise_scheduler():
    pytest.importorskip("tensorflow")
    from tensorflow import keras

    boundaries = [100000, 150000]
    values = [1.0, 0.5, 0.1]
    config = LRSchedulerConfig().set_piecewise(boundaries, values)
    scheduler = config.apply_tf_scheduler(0.1)
    assert isinstance(scheduler, keras.optimizers.schedules.PiecewiseConstantDecay)
    assert scheduler.boundaries == boundaries
    assert scheduler.values == values
