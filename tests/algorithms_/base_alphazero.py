from .common_base_class import CommonBaseClass


class BaseCase(CommonBaseClass):
    def _create_rl_config(self):
        from srl.algorithms import alphazero

        rl_config = alphazero.Config(
            num_simulations=100,
            sampling_steps=1,
            batch_size=32,
            discount=1.0,
        )
        rl_config.lr.clear()
        rl_config.lr.add_constant(100, 0.02)
        rl_config.lr.add_constant(1000, 0.002)
        rl_config.lr.add_constant(1, 0.0002)
        rl_config.input_image_block.set_alphazero_block(1, 32)
        rl_config.value_block.set_mlp((32,))
        return rl_config

    def test_Grid(self):
        self.check_skip()
        from srl.envs import grid

        rl_config = self._create_rl_config()
        rl_config.discount = 0.9
        rl_config.memory_warmup_size = 100
        rl_config.processors = [grid.LayerProcessor()]
        runner, tester = self.create_runner("Grid", rl_config)

        runner.train(max_train_count=1000)
        tester.eval(runner)

    def test_StoneTaking(self):
        self.check_skip()
        rl_config = self._create_rl_config()

        runner, tester = self.create_runner("StoneTaking", rl_config)
        runner.set_seed(2)
        runner.train(max_train_count=300)

        runner.set_players([None, "random"])
        tester.eval(runner, baseline=[0.9, None])
        runner.set_players(["random", None])
        tester.eval(runner, baseline=[None, 0.7])

    def test_OX(self):
        self.check_skip()
        rl_config = self._create_rl_config()
        runner, tester = self.create_runner("OX", rl_config)
        runner.train(max_train_count=200)

        runner.set_players([None, "random"])
        tester.eval(runner, baseline=[0.8, None])
        runner.set_players(["random", None])
        tester.eval(runner, baseline=[None, 0.6])

    def test_OX_mp(self):
        self.check_skip()
        rl_config = self._create_rl_config()
        runner, tester = self.create_runner("OX", rl_config)
        runner.set_seed(2)
        runner.train_mp(max_train_count=300)

        runner.set_players([None, "random"])
        tester.eval(runner, baseline=[0.8, None])
        runner.set_players(["random", None])
        tester.eval(runner, baseline=[None, 0.65])

    def test_Othello4x4(self):
        self.check_skip()
        from srl.envs import othello

        rl_config = self._create_rl_config()
        rl_config.batch_size = 32
        rl_config.memory_warmup_size = 500
        rl_config.lr_schedule = [
            {"train": 0, "lr": 0.001},
            {"train": 1000, "lr": 0.0005},
            {"train": 5000, "lr": 0.0002},
        ]
        rl_config.lr.clear()
        rl_config.lr.add_constant(1000, 0.001)
        rl_config.lr.add_constant(5000, 0.0005)
        rl_config.lr.add_constant(1, 0.0002)
        rl_config.input_image_block.set_alphazero_block(9, 32)
        rl_config.value_block.set_mlp((16, 16))
        rl_config.policy_block.set_mlp((32,))
        rl_config.processors = [othello.LayerProcessor()]

        runner, tester = self.create_runner("Othello4x4", rl_config)
        runner.train(max_train_count=20_000)

        runner.set_players([None, "random"])
        tester.eval(runner, baseline=[0.1, None])
        runner.set_players(["random", None])
        tester.eval(runner, baseline=[None, 0.5])
