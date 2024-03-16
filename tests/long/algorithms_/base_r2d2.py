from .common_base_class import CommonBaseClass


class BaseCase(CommonBaseClass):
    def _create_rl_config(self):
        from srl.algorithms import r2d2

        rl_config = r2d2.Config(
            lstm_units=32,
            target_model_update_interval=100,
            enable_rescale=True,
            burnin=5,
            sequence_length=5,
            enable_retrace=False,
        )
        rl_config.dueling_network.set((16, 16), False)
        rl_config.memory.set_replay_memory()
        return rl_config

    def test_Pendulum(self):
        self.check_skip()
        rl_config = self._create_rl_config()
        runner, tester = self.create_runner("Pendulum-v1", rl_config)
        runner.train(max_train_count=200 * 35)
        tester.eval(runner)

    def test_Pendulum_mp(self):
        self.check_skip()
        rl_config = self._create_rl_config()
        runner, tester = self.create_runner("Pendulum-v1", rl_config)
        runner.train_mp(max_train_count=200 * 20)
        tester.eval(runner)

    def test_Pendulum_retrace(self):
        self.check_skip()
        rl_config = self._create_rl_config()
        rl_config.enable_retrace = True
        runner, tester = self.create_runner("Pendulum-v1", rl_config)
        runner.train(max_train_count=200 * 35)
        tester.eval(runner)
