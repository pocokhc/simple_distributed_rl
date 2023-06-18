from typing import cast

import pytest

import srl
from srl import runner
from srl.utils import common

from .common_base_class import CommonBaseClass


class _BaseCase(CommonBaseClass):
    def return_rl_config(self, framework):
        from srl.algorithms import dreamer

        return dreamer.Config(
            deter_size=30,
            stoch_size=20,
            reward_num_units=30,
            reward_layers=2,
            value_num_units=50,
            value_layers=2,
            action_num_units=50,
            action_layers=2,
            cnn_depth=32,
            batch_size=32,
            batch_length=21,
            free_nats=0.1,
            kl_scale=1.0,
            model_lr=0.001,
            value_lr=0.0005,
            actor_lr=0.0001,
            memory_warmup_size=1000,
            epsilon=1.0,
            value_estimation_method="dreamer",  # "simple" or "dreamer"
            horizon=20,
        )

    def test_EasyGrid(self):
        from srl.algorithms import dreamer

        env_config = srl.EnvConfig("EasyGrid")
        env_config.max_episode_steps = 20
        env_config.check_action = False
        env_config.check_val = False

        config, rl_config, tester = self.create_config(env_config)
        rl_config = cast(dreamer.Config, rl_config)
        rl_config.use_render_image_for_observation = True

        # --- train dynamics
        rl_config.enable_train_model = True
        rl_config.enable_train_actor = False
        rl_config.enable_train_value = False
        parameter, _ = runner.train_simple(config, max_train_count=10_000)

        # --- train value
        rl_config.enable_train_model = False
        rl_config.enable_train_actor = False
        rl_config.enable_train_value = True
        parameter, _ = runner.train_simple(
            config,
            parameter=parameter,
            max_train_count=1_000,
        )

        # --- train actor
        rl_config.enable_train_model = False
        rl_config.enable_train_actor = True
        rl_config.enable_train_value = True
        parameter, _ = runner.train_simple(
            config,
            parameter=parameter,
            max_train_count=3_000,
        )

        # --- eval
        tester.eval(config, parameter, episode=5, baseline=0.2)


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
