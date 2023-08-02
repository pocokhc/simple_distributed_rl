from .common_base_class import CommonBaseClass


class BaseCase(CommonBaseClass):
    def _create_rl_config(self):
        from srl.algorithms import c51

        return c51.Config()

    def test_Grid(self):
        self.check_skip()
        rl_config = self._create_rl_config()
        rl_config.epsilon = 0.5
        rl_config.lr = 0.002
        rl_config.hidden_block.set_mlp((16, 16))
        rl_config.categorical_num_atoms = 11
        rl_config.categorical_v_min = -2
        rl_config.categorical_v_max = 2
        runner, tester = self.create_runner("Grid", rl_config)
        runner.train(max_train_count=6000)
        tester.eval(runner)

    def test_Pendulum(self):
        self.check_skip()
        rl_config = self._create_rl_config()
        rl_config.categorical_v_min = -100
        rl_config.categorical_v_max = 100
        rl_config.batch_size = 64
        rl_config.lr = 0.001
        rl_config.hidden_block.set_mlp((32, 32, 32))
        runner, tester = self.create_runner("Pendulum-v1", rl_config)
        runner.train(max_train_count=200 * 600)
        tester.eval(runner)

    def test_Pendulum_mp(self):
        self.check_skip()
        rl_config = self._create_rl_config()
        rl_config.categorical_v_min = -100
        rl_config.categorical_v_max = 100
        rl_config.batch_size = 64
        rl_config.lr = 0.001
        rl_config.hidden_block.set_mlp((32, 32, 32))
        runner, tester = self.create_runner("Pendulum-v1", rl_config)
        runner.train_mp(max_train_count=200 * 600)
        tester.eval(runner)
