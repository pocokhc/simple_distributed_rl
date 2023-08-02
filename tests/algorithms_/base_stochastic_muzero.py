from .common_base_class import CommonBaseClass


class BaseCase(CommonBaseClass):
    def _create_rl_config(self):
        from srl.algorithms import stochastic_muzero

        rl_config = stochastic_muzero.Config(
            num_simulations=10,
            discount=0.9,
            batch_size=16,
            memory_warmup_size=200,
            lr_init=0.01,
            lr_decay_steps=10_000,
            v_min=-2,
            v_max=2,
            unroll_steps=2,
            dynamics_blocks=1,
            enable_rescale=False,
            codebook_size=4,
        )
        rl_config.memory.set_replay_memory()
        rl_config.input_image_block.set_alphazero_block(1, 16)
        return rl_config

    def test_Grid(self):
        self.check_skip()
        from srl.envs import grid

        rl_config = self._create_rl_config()
        rl_config.processors = [grid.LayerProcessor()]

        runner, tester = self.create_runner("Grid", rl_config)
        runner.train(max_train_count=10000)
        tester.eval(runner)
