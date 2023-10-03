import pytest

import srl

from .common_base_class import CommonBaseClass


class BaseCase(CommonBaseClass):
    def _create_rl_config(self):
        from srl.algorithms import dreamer_v2

        rl_config = dreamer_v2.Config(
            deter_size=100,
            stoch_size=32,
            reward_layer_sizes=(30, 30),
            discount_layer_sizes=(50,),
            critic_layer_sizes=(50, 50),
            actor_layer_sizes=(50, 50),
            discount=0.9,
            batch_size=32,
            batch_length=21,
            free_nats=0.1,
            kl_scale=0.1,
            lr_model=0.002,
            lr_critic=0.0005,
            lr_actor=0.0005,
            memory_warmup_size=1000,
            epsilon=1.0,
            critic_estimation_method="dreamer_v2",
            experience_acquisition_method="episode",
            horizon=15,
            reinforce_rate=0.1,
            entropy_rate=0.1,
            reinforce_baseline="v",
        )
        return rl_config

    def test_EasyGrid(self):
        self.check_skip()

        env_config = srl.EnvConfig("EasyGrid")
        env_config.max_episode_steps = 20

        rl_config = self._create_rl_config()

        runner, tester = self.create_runner(env_config, rl_config)
        rl_config.use_render_image_for_observation = True

        # --- train dynamics
        rl_config.enable_train_model = True
        rl_config.enable_train_actor = False
        rl_config.enable_train_critic = False
        runner.train(max_train_count=3_000)

        # --- train actor
        rl_config.enable_train_model = False
        rl_config.enable_train_actor = True
        rl_config.enable_train_critic = True
        runner.train(max_train_count=2_000)

        # --- eval
        tester.eval(runner, episode=5, baseline=0.9)
