import pytest

import srl
from srl import runner
from srl.utils import common

from .common_base_class import CommonBaseClass


class _BaseCase(CommonBaseClass):
    def return_rl_config(self, framework):
        from srl.algorithms import planet

        return planet.Config(
            deter_size=50,
            stoch_size=10,
            num_units=100,
            cnn_depth=32,
            batch_size=8,
            batch_length=20,
            lr=0.001,
            free_nats=3.0,
            kl_scale=1.0,
            enable_overshooting_loss=False,
            # GA
            pred_action_length=10,
            num_generation=20,
            num_individual=10,
            num_simulations=5,
            print_ga_debug=False,
        )

    def test_EasyGrid(self):
        env_config = srl.EnvConfig("EasyGrid")
        env_config.max_episode_steps = 21

        config, rl_config, tester = self.create_config(env_config)
        rl_config.memory_warmup_size = rl_config.batch_size + 1
        rl_config.change_observation_render_image = True

        # --- train
        _, memory, _ = runner.train(
            config,
            max_episodes=1000,
            disable_trainer=True,
        )
        parameter, _, _ = runner.train_only(
            config,
            remote_memory=memory,
            max_train_count=10_000,
        )

        # --- eval
        tester.eval(config, parameter, episode=10, baseline=0.5)

    def test_Grid(self):
        env_config = srl.EnvConfig("Grid")
        env_config.max_episode_steps = 21

        config, rl_config, tester = self.create_config(env_config)
        rl_config.set_params(
            dict(
                deter_size=50,
                stoch_size=10,
                num_units=100,
                cnn_depth=32,
                batch_size=8,
                batch_length=20,
                lr=0.001,
                free_nats=3.0,
                kl_scale=1.0,
                enable_overshooting_loss=False,
                # GA
                pred_action_length=10,
                num_generation=20,
                num_individual=10,
                num_simulations=5,
                print_ga_debug=False,
            )
        )
        rl_config.memory_warmup_size = rl_config.batch_size + 1
        rl_config.change_observation_render_image = True

        config = runner.Config(env_config, rl_config)

        # train
        _, memory, _ = runner.train(
            config,
            max_episodes=1000,
            disable_trainer=True,
        )
        parameter, _, _ = runner.train_only(
            config,
            remote_memory=memory,
            max_train_count=40_000,
        )

        # eval
        tester.eval(config, parameter, baseline=0.1)

    def test_Grid_overshooting(self):
        env_config = srl.EnvConfig("Grid")
        env_config.max_episode_steps = 21

        config, rl_config, tester = self.create_config(env_config)

        rl_config.set_params(
            dict(
                deter_size=50,
                stoch_size=10,
                num_units=100,
                cnn_depth=32,
                batch_size=8,
                batch_length=20,
                lr=0.001,
                free_nats=3.0,
                kl_scale=1.0,
                enable_overshooting_loss=True,
                # GA
                pred_action_length=10,
                num_generation=20,
                num_individual=10,
                num_simulations=5,
                print_ga_debug=False,
            )
        )
        rl_config.memory_warmup_size = rl_config.batch_size + 1
        rl_config.change_observation_render_image = True

        config = runner.Config(env_config, rl_config)

        # train
        _, memory, _ = runner.train(
            config,
            max_episodes=1000,
            disable_trainer=True,
        )
        parameter, _, _ = runner.train_only(
            config,
            remote_memory=memory,
            max_train_count=40_000,
        )

        # eval
        tester.eval(config, parameter, baseline=0.1)


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
