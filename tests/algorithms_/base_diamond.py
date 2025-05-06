from typing import Tuple

from srl.base.rl.config import RLConfig
from tests.algorithms_.common_long_case import CommonLongCase
from tests.algorithms_.common_quick_case import CommonQuickCase


class QuickCase(CommonQuickCase):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        from srl.algorithms import diamond

        rl_config = diamond.Config(observation_mode="render_image").set_small_params()
        rl_config.batch_size = 2
        rl_config.memory.warmup_size = 2
        rl_config.burnin = 1
        rl_config.horizon = 2
        rl_config.denoiser_cfg = diamond.DenoiserConfig(
            num_steps_conditioning=1,
            condition_channels=2,
            channels_list=[2],
            res_block_num_list=[1],
            use_attention_list=[False],
        )
        self.reward_end_cfg = diamond.RewardEndModelConfig(
            lstm_dim=2,
            condition_channels=2,
            channels_list=[2],
            res_block_num_list=[2],
            use_attention_list=[False],
        )
        self.actor_critic_cfg = diamond.ActorCriticConfig(
            lstm_dim=2,
            channels_list=[2],
            enable_downsampling_list=[True],
        )
        self.sampler_cfg = diamond.DiffusionSamplerConfig(
            num_steps_denoising=1,
        )
        return rl_config, {}


class LongCase(CommonLongCase):
    def _create_rl_config(self):
        self.check_test_skip()

        from srl.algorithms import diamond

        rl_config = diamond.Config(observation_mode="render_image").set_small_params()
        return rl_config

    def test_EasyGrid(self):
        rl_config = self._create_rl_config()
        runner = self.create_test_runner("EasyGrid", rl_config)

        # train diff
        runner.rl_config.train_reward_end = False
        runner.rl_config.train_actor_critic = False
        runner.rollout(max_memory=10000)
        runner.train_only(max_train_count=15000)

        # train rewend
        runner.rl_config.train_diffusion = False
        runner.rl_config.train_reward_end = True
        runner.rl_config.train_actor_critic = False
        runner.rl_config.batch_size = 8
        runner.train_only(max_train_count=500)

        # train act
        runner.rl_config.train_diffusion = False
        runner.rl_config.train_reward_end = True
        runner.rl_config.train_actor_critic = True
        runner.rl_config.batch_size = 8
        runner.train(max_train_count=5000)

        assert runner.evaluate_compare_to_baseline_single_player(episode=10)
