from .common_base_class import CommonBaseClass


class BaseCase(CommonBaseClass):
    def _create_rl_config(self):
        from srl.algorithms import muzero

        return muzero.Config(
            batch_size=16,
            memory_warmup_size=50,
        )

    def test_EasyGrid(self):
        self.check_skip()
        from srl.envs import grid

        rl_config = self._create_rl_config()
        rl_config.__init__(
            num_simulations=20,
            discount=0.9,
            batch_size=16,
            memory_warmup_size=200,
            v_min=-2,
            v_max=2,
            unroll_steps=1,
            dynamics_blocks=1,
            enable_rescale=False,
            weight_decay=0,
        )
        rl_config.lr.set_constant(0.001)
        rl_config.input_image_block.set_alphazero_block(1, 16)
        rl_config.memory.set_replay_memory()
        rl_config.processors = [grid.LayerProcessor()]
        runner, tester = self.create_runner("EasyGrid", rl_config)
        runner.train(max_train_count=3000)
        tester.eval(runner)

    def test_EasyGrid_PER(self):
        self.check_skip()
        from srl.envs import grid

        rl_config = self._create_rl_config()
        rl_config.__init__(
            num_simulations=20,
            discount=0.9,
            batch_size=16,
            memory_warmup_size=200,
            v_min=-2,
            v_max=2,
            unroll_steps=1,
            dynamics_blocks=1,
            enable_rescale=False,
            weight_decay=0,
        )
        rl_config.lr.set_linear(10_000, 0.002, 0.0001)
        rl_config.input_image_block.set_alphazero_block(1, 16)
        rl_config.processors = [grid.LayerProcessor()]
        runner, tester = self.create_runner("EasyGrid", rl_config)
        runner.train(max_train_count=3000)
        tester.eval(runner)
