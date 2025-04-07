import math

import numpy as np

from srl.rl.schedulers.scheduler import SchedulerConfig


def test_constant():
    sch = SchedulerConfig()
    sch.set_constant(0.2)
    val = sch.create()

    for i in range(10):
        rate = val.update(i).to_float()
        assert rate == 0.2


def test_linear_up():
    sch = SchedulerConfig()
    sch.set_linear(0.0, 1.0, 10)
    val = sch.create()

    for i in range(10):
        rate = val.update(i).to_float()
        print(i, rate)
        assert math.isclose(rate, i / 10)

    for i in range(10, 20):
        rate = val.update(i).to_float()
        print(i, rate)
        assert math.isclose(rate, 1)


def test_linear_down():
    sch = SchedulerConfig()
    sch.set_linear(1.0, 0.0, 10)
    val = sch.create()

    for i in range(10):
        rate = val.update(i).to_float()
        print(i, rate)
        assert math.isclose(rate, 1 - i / 10)

    for i in range(10, 20):
        rate = val.update(i).to_float()
        print(i, rate)
        assert math.isclose(rate, 0)


def test_cosine():
    sch = SchedulerConfig()
    sch.set_cosine(1.0, 10)
    val = sch.create()

    for i in range(10):
        rate = val.update(i).to_float()
        true_rate = np.cos(((np.pi / 2) / 10) * i)
        print(i, rate, true_rate)
        assert math.isclose(rate, true_rate)

    for i in range(10, 20):
        rate = val.update(i).to_float()
        print(i, rate)
        assert math.isclose(rate, 0)


def test_cosine_with_hard_restarts():
    sch = SchedulerConfig()
    sch.set_cosine_with_hard_restarts(1.0, 10, 2)
    val = sch.create()

    for i in range(5):
        rate = val.update(i).to_float()
        true_rate = np.cos(((np.pi / 2) / 5) * i)
        print(i, rate, true_rate)
        assert math.isclose(rate, true_rate)

    for i in range(5, 10):
        rate = val.update(i).to_float()
        true_rate = np.cos(((np.pi / 2) / 5) * (i - 5))
        print(i, rate, true_rate)
        assert math.isclose(rate, true_rate)

    for i in range(10, 20):
        rate = val.update(i).to_float()
        print(i, rate)
        assert math.isclose(rate, 0)


def test_polynomial():
    sch = SchedulerConfig()
    sch.set_polynomial(1.0, 10, power=2)
    val = sch.create()

    for i in range(10):
        rate = val.update(i).to_float()
        true_rate = (1 - (i / 10)) ** 2

        print(i, rate, true_rate)
        assert math.isclose(rate, true_rate)

    for i in range(10, 20):
        rate = val.update(i).to_float()
        print(i, rate)
        assert math.isclose(rate, 0)


def test_multi_plot():
    sch = SchedulerConfig()
    sch.add_linear(0.5, 1.0, 100)
    sch.add_linear(1.0, 0.1, 200)
    sch.add_cosine(0.7, 200)
    sch.add_cosine_with_hard_restarts(0.7, 500, 5)
    sch.add_polynomial(1.0, 200)
    sch.add_constant(0.2)
    sch.plot(_no_plot=True)
