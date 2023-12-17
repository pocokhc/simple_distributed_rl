import pytest_mock

import srl
from srl.algorithms import ql
from srl.runner.callback import RunnerCallback


def test_callback(mocker: pytest_mock.MockerFixture):
    runner = srl.Runner("Grid", ql.Config())

    c = mocker.Mock(spec=RunnerCallback)
    state = runner.train(max_steps=10, callbacks=[c])
    assert state.total_step == 10
    assert c.on_runner_start.call_count == 1
    assert c.on_runner_end.call_count == 1
