import srl

from .common_base_class import CommonBaseClass


class BaseCase(CommonBaseClass):
    def _create_rl_config(self, mode):
        from srl.algorithms import dreamer_v3

        rl_config = dreamer_v3.Config()
        if mode == "v1":
            rl_config.set_dreamer_v1()
        elif mode == "v2":
            rl_config.set_dreamer_v2()
        elif mode == "v3":
            rl_config.set_dreamer_v3()

        # model
        rl_config.rssm_deter_size = 64
        rl_config.rssm_stoch_size = 2
        rl_config.rssm_classes = 2
        rl_config.rssm_hidden_units = 64
        rl_config.reward_layer_sizes = (64,)
        rl_config.cont_layer_sizes = (64,)
        rl_config.critic_layer_sizes = (64, 64)
        rl_config.actor_layer_sizes = (64, 64)
        rl_config.encoder_decoder_mlp = (16, 16, 16)
        # lr
        rl_config.batch_size = 32
        rl_config.batch_length = 5
        rl_config.lr_model.set_constant(0.0005)
        rl_config.lr_critic.set_constant(0.0001)
        rl_config.lr_actor.set_constant(0.00005)
        rl_config.horizon = 5
        # memory
        rl_config.memory.warmup_size = 50
        rl_config.memory.capacity = 10_000
        return rl_config

    def test_EasyGrid_v1(self):
        self.check_skip()

        env_config = srl.EnvConfig("EasyGrid")
        rl_config = self._create_rl_config("v1")
        rl_config.actor_loss_type = "dreamer_v2"

        runner, tester = self.create_runner(env_config, rl_config)
        runner.train(max_train_count=10_000)
        tester.eval(runner, episode=5, baseline=0.9)

    def test_EasyGrid_v2(self):
        self.check_skip()

        env_config = srl.EnvConfig("EasyGrid")
        rl_config = self._create_rl_config("v2")

        runner, tester = self.create_runner(env_config, rl_config)
        runner.train(max_train_count=10_000)
        tester.eval(runner, episode=5, baseline=0.9)

    def test_EasyGrid_v3(self):
        self.check_skip()

        env_config = srl.EnvConfig("EasyGrid")
        rl_config = self._create_rl_config("v3")

        runner, tester = self.create_runner(env_config, rl_config)
        runner.train(max_train_count=10_000)
        tester.eval(runner, episode=5, baseline=0.9)
