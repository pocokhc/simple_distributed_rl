import pytest

import srl
from srl.test import TestRL


def test_Pendulum():
    pytest.importorskip("tensorflow")
    from srl.algorithms import agent57_stateful

    tester = TestRL()
    rl_config = agent57_stateful.Config(
        lstm_units=128,
        hidden_layer_sizes=(128,),
        target_model_update_interval=100,
        enable_rescale=True,
        q_ext_lr=0.001,
        q_int_lr=0.001,
        batch_size=32,
        burnin=5,
        sequence_length=10,
        enable_retrace=False,
        actor_num=8,
        input_ext_reward=False,
        input_int_reward=False,
        input_action=False,
        enable_intrinsic_reward=True,
    )
    runner = srl.Runner("Pendulum-v1", rl_config)
    runner.set_seed(1)
    runner.train(max_train_count=200 * 40)
    tester.eval(runner)
