import srl

from .common_base_class import CommonBaseClass


class BaseCase(CommonBaseClass):
    def _create_rl_config(self):
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
        self.check_skip()

        env_config = srl.EnvConfig("EasyGrid")
        env_config.max_episode_steps = 20

        rl_config = self._create_rl_config()

        runner, tester = self.create_runner(env_config, rl_config)
        rl_config.use_render_image_for_observation = True

        # --- train dynamics
        rl_config.enable_train_model = True
        rl_config.enable_train_actor = False
        rl_config.enable_train_value = False
        runner.train(max_train_count=10_000)

        # --- train value
        rl_config.enable_train_model = False
        rl_config.enable_train_actor = False
        rl_config.enable_train_value = True
        runner.train(max_train_count=1_000)

        # --- train actor
        rl_config.enable_train_model = False
        rl_config.enable_train_actor = True
        rl_config.enable_train_value = True
        runner.train(max_train_count=3_000)

        # --- eval
        tester.eval(runner, episode=5, baseline=0.2)
