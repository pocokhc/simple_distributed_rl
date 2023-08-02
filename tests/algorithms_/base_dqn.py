from .common_base_class import CommonBaseClass


class BaseCase(CommonBaseClass):
    def _create_rl_config(self):
        from srl.algorithms import dqn

        rl_config = dqn.Config(
            enable_double_dqn=False,
            framework=self.get_framework(),
        )
        rl_config.hidden_block.set_mlp((64, 64))
        return rl_config

    def test_Pendulum(self):
        self.check_skip()
        rl_config = self._create_rl_config()
        runner, tester = self.create_runner("Pendulum-v1", rl_config)
        runner.train(max_steps=200 * 100)
        tester.eval(runner)

    def test_Pendulum_mp(self):
        self.check_skip()
        rl_config = self._create_rl_config()
        rl_config.memory.set_proportional_memory()

        runner, tester = self.create_runner("Pendulum-v1", rl_config)
        runner.train_mp(max_train_count=200 * 100)
        tester.eval(runner)

    def test_Pendulum_DDQN(self):
        self.check_skip()
        rl_config = self._create_rl_config()
        rl_config.enable_double_dqn = True

        runner, tester = self.create_runner("Pendulum-v1", rl_config)
        runner.train(max_steps=200 * 70)
        tester.eval(runner)

    def test_Pendulum_window(self):
        self.check_skip()
        rl_config = self._create_rl_config()
        rl_config.window_length = 4

        runner, tester = self.create_runner("Pendulum-v1", rl_config)
        runner.train(max_steps=200 * 70)
        tester.eval(runner)

        runner.model_summary()

    def test_OX(self):
        self.check_skip()
        rl_config = self._create_rl_config()
        rl_config.hidden_block.set_mlp((64, 32, 16))
        rl_config.epsilon = 0.5

        runner, tester = self.create_runner("OX", rl_config)
        runner.set_seed(2)
        runner.train(max_train_count=30_000)

        runner.set_players([None, "random"])
        tester.eval(runner, baseline=[0.3, None])
        runner.set_players(["random", None])
        tester.eval(runner, baseline=[None, 0.1])

    def case_image_r2d3(self):
        self.check_skip()
        rl_config = self._create_rl_config()
        rl_config.image_block.set_r2d3_image()
        rl_config.hidden_block.set_mlp((128, 16, 16))
        rl_config.use_render_image_for_observation = True

        runner, tester = self.create_runner("Grid", rl_config)
        runner.train(max_train_count=200 * 150)
        tester.eval(runner)
