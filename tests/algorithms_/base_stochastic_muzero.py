from srl.rl.memories.config import ReplayMemoryConfig

from .common_base_class import CommonBaseClass


class BaseCase(CommonBaseClass):
    def _create_rl_config(self):
        from srl.algorithms import stochastic_muzero
        from srl.rl.models.alphazero import AlphaZeroImageBlockConfig

        return stochastic_muzero.Config(
            num_simulations=10,
            discount=0.9,
            batch_size=16,
            memory_warmup_size=200,
            memory=ReplayMemoryConfig(),
            lr_init=0.01,
            lr_decay_steps=10_000,
            v_min=-2,
            v_max=2,
            unroll_steps=2,
            input_image_block=AlphaZeroImageBlockConfig(n_blocks=1, filters=16),
            dynamics_blocks=1,
            enable_rescale=False,
            codebook_size=4,
        )

    def test_Grid(self):
        from srl.envs import grid

        rl_config = self._create_rl_config()
        rl_config.processors = [grid.LayerProcessor()]

        config, tester = self.create_config("Grid", rl_config)
        tester.train_eval(config, 10000)
