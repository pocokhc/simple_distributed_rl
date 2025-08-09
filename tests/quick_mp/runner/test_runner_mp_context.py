import numpy as np

import srl
from srl.algorithms import ql


def test_train():
    c = srl.RunContext(srl.EnvConfig("Grid"), ql.Config())
    c.play_mode = "train_mp"
    c.max_train_count = 5000
    runner = srl.Runner(context=c)

    runner.play()
    rewards = np.mean(runner.evaluate(100))
    assert rewards > 0.6
