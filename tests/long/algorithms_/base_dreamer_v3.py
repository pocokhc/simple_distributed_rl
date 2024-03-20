import pytest

import srl
from srl.base.define import RLTypes

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
        # memory
        rl_config.memory.warmup_size = 50
        rl_config.memory.capacity = 10_000
        return rl_config

    @pytest.mark.parametrize("ver", ["v1", "v2", "v3"])
    def test_EasyGrid(self, ver):
        self.check_skip()

        env_config = srl.EnvConfig("EasyGrid")
        rl_config = self._create_rl_config(ver)

        # model
        rl_config.rssm_deter_size = 8
        rl_config.rssm_stoch_size = 16
        rl_config.rssm_classes = 16
        rl_config.rssm_hidden_units = 32
        rl_config.reward_layer_sizes = (32,)
        rl_config.cont_layer_sizes = (32,)
        rl_config.critic_layer_sizes = (128,)
        rl_config.actor_layer_sizes = (128,)
        rl_config.encoder_decoder_mlp = (16, 16)
        # lr
        rl_config.batch_size = 32
        rl_config.batch_length = 5
        rl_config.lr_model.set_constant(0.0005)
        rl_config.lr_critic.set_constant(0.0003)
        rl_config.lr_actor.set_constant(0.0001)
        rl_config.horizon = 3

        rl_config.encoder_decoder_dist = "linear"
        rl_config.free_nats = 0.01
        rl_config.warmup_world_model = 1_000

        runner, tester = self.create_runner(env_config, rl_config)
        if ver == "v1":
            runner.train(max_train_count=3_000)
        elif ver == "v2":
            runner.train(max_train_count=3_000)
        elif ver == "v3":
            rl_config.warmup_world_model = 2_000
            runner.train(max_train_count=3_000)
        tester.eval(runner, episode=5, baseline=0.9)

    @pytest.mark.parametrize("ver", ["v1", "v2", "v3"])
    def test_Grid_cont(self, ver):
        self.check_skip()

        env_config = srl.EnvConfig("Grid")
        rl_config = self._create_rl_config(ver)

        # model
        rl_config.rssm_deter_size = 8
        rl_config.rssm_stoch_size = 16
        rl_config.rssm_classes = 16
        rl_config.rssm_hidden_units = 32
        rl_config.reward_layer_sizes = (32,)
        rl_config.cont_layer_sizes = (32,)
        rl_config.critic_layer_sizes = (64,)
        rl_config.actor_layer_sizes = (64,)
        rl_config.encoder_decoder_mlp = (16, 16)
        # lr
        rl_config.batch_size = 32
        rl_config.batch_length = 5
        rl_config.lr_model.set_constant(0.0005)
        rl_config.lr_critic.set_constant(0.0005)
        rl_config.lr_actor.set_constant(0.0001)
        rl_config.horizon = 5

        rl_config.encoder_decoder_dist = "linear"
        rl_config.free_nats = 0.01
        rl_config.override_action_type = RLTypes.CONTINUOUS
        rl_config.actor_reinforce_rate = 1.0

        runner, tester = self.create_runner(env_config, rl_config)
        if ver == "v1":
            rl_config.warmup_world_model = 3_000
            runner.train(max_train_count=10_000)
        elif ver == "v2":
            rl_config.entropy_rate = 0.1
            rl_config.reinforce_baseline = ""
            rl_config.warmup_world_model = 5_000
            runner.train(max_train_count=10_000)
        elif ver == "v3":
            rl_config.entropy_rate = 0.1
            rl_config.reinforce_baseline = ""
            rl_config.warmup_world_model = 5_000
            runner.train(max_train_count=10_000)
        tester.eval(runner, episode=5, baseline=0.1)

    @pytest.mark.parametrize("params", [["v1"], ["v2"], ["v3"]])
    def test_EasyGrid_img(self, params):
        self.check_skip()

        env_config = srl.EnvConfig("EasyGrid")
        env_config.max_episode_steps = 20
        rl_config = self._create_rl_config(params[0])

        # model
        rl_config.rssm_deter_size = 512
        rl_config.rssm_stoch_size = 16
        rl_config.rssm_classes = 16
        rl_config.rssm_hidden_units = 512
        rl_config.reward_layer_sizes = (512,)
        rl_config.cont_layer_sizes = (512,)
        rl_config.critic_layer_sizes = (128,)
        rl_config.actor_layer_sizes = (128,)
        # lr
        rl_config.batch_size = 16
        rl_config.batch_length = 20
        rl_config.lr_model.set_constant(0.0001)
        rl_config.lr_critic.set_constant(0.0001)
        rl_config.lr_actor.set_constant(0.00005)
        rl_config.horizon = 5

        # other
        rl_config.cnn_depth = 32
        rl_config.cnn_resized_image_size = 1
        rl_config.free_nats = 0.1
        rl_config.encoder_decoder_dist = "linear"
        rl_config.cnn_use_sigmoid = False

        rl_config.warmup_world_model = 10_000
        rl_config.use_render_image_for_observation = True

        runner, tester = self.create_runner(env_config, rl_config)
        runner.train(max_train_count=12_000)
        tester.eval(runner, episode=5, baseline=0.9)
