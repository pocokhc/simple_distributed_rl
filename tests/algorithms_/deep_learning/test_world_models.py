import pytest

import srl
from srl import runner
from srl.utils import common

from .common_base_class import CommonBaseClass


class _BaseCase(CommonBaseClass):
    def return_rl_config(self, framework):
        from srl.algorithms import world_models

        return world_models.Config(
            z_size=1,
            sequence_length=10,
            rnn_units=8,
            num_mixture=3,
            batch_size=64,
        )

    def test_Grid(self):
        config, rl_config, tester = self.create_config("Pendulum-v1")

        rl_config.change_observation_render_image = True
        env_config = srl.EnvConfig("Grid")
        config = runner.Config(env_config, rl_config)
        config.rl_config.train_mode = 1
        _, memory, _ = runner.train(
            config,
            max_episodes=100,
            disable_trainer=True,
        )

        # vae
        rl_config.train_mode = 1
        rl_config.lr = 0.001
        rl_config.kl_tolerance = 4.0
        parameter, memory, history = runner.train_only(
            config,
            remote_memory=memory,
            max_train_count=20_000,
        )

        # rnn
        rl_config.train_mode = 2
        rl_config.lr = 0.001
        rl_config.memory_warmup_size = 100
        parameter, memory, history = runner.train_only(
            config,
            parameter=parameter,
            remote_memory=memory,
            max_train_count=40_000,
        )

        # controller
        rl_config.train_mode = 3
        rl_config.num_simulations = 10
        rl_config.num_individual = 4
        rl_config.blx_a = 0.3
        max_episodes = rl_config.num_simulations * rl_config.num_individual * 300
        parameter, memory, history = runner.train(
            config,
            parameter=parameter,
            max_episodes=max_episodes,
        )

        tester.eval(config, parameter, episode=200, baseline=0.3)


class TestTF_CPU(_BaseCase):
    def return_params(self):
        pytest.importorskip("tensorflow")

        return "tensorflow", "CPU"


class TestTF_GPU(_BaseCase):
    def return_params(self):
        pytest.importorskip("tensorflow")
        if not common.is_available_gpu_tf():
            pytest.skip()

        return "tensorflow", "GPU"
