from srl.rl.memories.config import ReplayMemoryConfig

from .common_base_class import CommonBaseClass


class BaseCase(CommonBaseClass):
    def _create_rl_config(self):
        from srl.algorithms import muzero

        return muzero.Config(
            batch_size=16,
            memory_warmup_size=50,
        )

    def test_EasyGrid(self):
        from srl.envs import grid
        from srl.rl.models.alphazero import AlphaZeroImageBlockConfig

        rl_config = self._create_rl_config()
        rl_config.set_parameter(
            dict(
                num_simulations=20,
                discount=0.9,
                batch_size=16,
                memory_warmup_size=200,
                memory=ReplayMemoryConfig(),
                lr_init=0.002,
                lr_decay_steps=10_000,
                v_min=-2,
                v_max=2,
                unroll_steps=1,
                input_image_block=AlphaZeroImageBlockConfig(n_blocks=1, filters=16),
                dynamics_blocks=1,
                enable_rescale=False,
                weight_decay=0,
            )
        )
        rl_config.processors = [grid.LayerProcessor()]
        config, tester = self.create_config("EasyGrid", rl_config)
        tester.train_eval(config, 2000)

    def test_EasyGrid_PER(self):
        from srl.envs import grid
        from srl.rl.models.alphazero import AlphaZeroImageBlockConfig

        rl_config = self._create_rl_config()
        rl_config.set_parameter(
            dict(
                num_simulations=20,
                discount=0.9,
                batch_size=16,
                memory_warmup_size=200,
                lr_init=0.002,
                lr_decay_steps=10_000,
                v_min=-2,
                v_max=2,
                unroll_steps=1,
                input_image_block=AlphaZeroImageBlockConfig(n_blocks=1, filters=16),
                dynamics_blocks=1,
                enable_rescale=False,
                weight_decay=0,
            )
        )
        rl_config.processors = [grid.LayerProcessor()]
        config, tester = self.create_config("EasyGrid", rl_config)
        tester.train_eval(config, 3000)
