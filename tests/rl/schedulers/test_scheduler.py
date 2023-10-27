import math

import numpy as np

from srl.rl.schedulers.scheduler import SchedulerConfig


def test_constant():
    config = SchedulerConfig()
    config.set_constant(0.2)
    sch = config.create_schedulers()

    for i in range(10):
        rate = sch.get_and_update_rate(i)
        assert rate == 0.2


def test_linear_up():
    config = SchedulerConfig()
    config.set_linear(10, 0.0, 1.0)
    sch = config.create_schedulers()

    for i in range(10):
        rate = sch.get_and_update_rate(i)
        print(i, rate)
        assert math.isclose(rate, i / 10)

    for i in range(10, 20):
        rate = sch.get_and_update_rate(i)
        print(i, rate)
        assert math.isclose(rate, 1)


def test_linear_down():
    config = SchedulerConfig()
    config.set_linear(10, 1.0, 0.0)
    sch = config.create_schedulers()

    for i in range(10):
        rate = sch.get_and_update_rate(i)
        print(i, rate)
        assert math.isclose(rate, 1 - i / 10)

    for i in range(10, 20):
        rate = sch.get_and_update_rate(i)
        print(i, rate)
        assert math.isclose(rate, 0)


def test_cosine():
    config = SchedulerConfig()
    config.set_cosine(10, 1.0)
    sch = config.create_schedulers()

    for i in range(10):
        rate = sch.get_and_update_rate(i)
        true_rate = np.cos(((np.pi / 2) / 10) * i)
        print(i, rate, true_rate)
        assert math.isclose(rate, true_rate)

    for i in range(10, 20):
        rate = sch.get_and_update_rate(i)
        print(i, rate)
        assert math.isclose(rate, 0)


def test_cosine_with_hard_restarts():
    config = SchedulerConfig()
    config.set_cosine_with_hard_restarts(10, 1.0, 2)
    sch = config.create_schedulers()

    for i in range(5):
        rate = sch.get_and_update_rate(i)
        true_rate = np.cos(((np.pi / 2) / 5) * i)
        print(i, rate, true_rate)
        assert math.isclose(rate, true_rate)

    for i in range(5, 10):
        rate = sch.get_and_update_rate(i)
        true_rate = np.cos(((np.pi / 2) / 5) * (i - 5))
        print(i, rate, true_rate)
        assert math.isclose(rate, true_rate)

    for i in range(10, 20):
        rate = sch.get_and_update_rate(i)
        print(i, rate)
        assert math.isclose(rate, 0)


def test_polynomial():
    config = SchedulerConfig()
    config.set_polynomial(10, 1.0, power=2)
    sch = config.create_schedulers()

    for i in range(10):
        rate = sch.get_and_update_rate(i)
        true_rate = (1 - (i / 10)) ** 2

        print(i, rate, true_rate)
        assert math.isclose(rate, true_rate)

    for i in range(10, 20):
        rate = sch.get_and_update_rate(i)
        print(i, rate)
        assert math.isclose(rate, 0)


def test_multi_plot():
    config = SchedulerConfig()
    config.add_linear(100, 0.5, 1.0)
    config.add_linear(200, 1.0, 0.1)
    config.add_cosine(200, 0.7)
    config.add_cosine_with_hard_restarts(500, 0.7, 5)
    config.add_polynomial(200, 1.0)

    # config.plot()  # debug
