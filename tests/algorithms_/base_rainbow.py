from .common_base_class import CommonBaseClass


class BaseCase(CommonBaseClass):
    def _create_rl_config(self):
        from srl.algorithms import rainbow

        rl_config = rainbow.Config()
        if self.get_framework() == "tensorflow":
            rl_config.framework.set_tensorflow()
        elif self.get_framework() == "torch":
            rl_config.framework.set_torch()
        return rl_config

    def test_Pendulum(self):
        self.check_skip()
        rl_config = self._create_rl_config()
        rl_config.dueling_network.set((64, 64), False)
        rl_config.multisteps = 3
        runner, tester = self.create_runner("Pendulum-v1", rl_config)
        runner.train(max_train_count=200 * 70)
        tester.eval(runner)

    def test_Pendulum_mp(self):
        self.check_skip()
        rl_config = self._create_rl_config()
        rl_config.dueling_network.set((64, 64), False)
        rl_config.multisteps = 3
        runner, tester = self.create_runner("Pendulum-v1", rl_config)
        runner.train_mp(max_train_count=200 * 70)
        tester.eval(runner)

    def test_Pendulum_noisy(self):
        self.check_skip()

        rl_config = self._create_rl_config()
        rl_config.dueling_network.set((64, 64), False)
        rl_config.multisteps = 1
        rl_config.enable_noisy_dense = True
        runner, tester = self.create_runner("Pendulum-v1", rl_config)
        runner.train(max_train_count=200 * 70)
        tester.eval(runner)

    def test_Pendulum_no_multi(self):
        self.check_skip()
        rl_config = self._create_rl_config()
        rl_config.dueling_network.set((64, 64), False)
        rl_config.multisteps = 1
        runner, tester = self.create_runner("Pendulum-v1", rl_config)
        runner.train(max_train_count=200 * 70)
        tester.eval(runner)

    def test_Pendulum_no_multi_mp(self):
        self.check_skip()
        rl_config = self._create_rl_config()
        rl_config.dueling_network.set((64, 64), False)
        rl_config.multisteps = 1
        runner, tester = self.create_runner("Pendulum-v1", rl_config)
        runner.train_mp(max_train_count=200 * 70)
        tester.eval(runner)

    def test_Pendulum_no_multi_noisy(self):
        self.check_skip()

        rl_config = self._create_rl_config()
        rl_config.dueling_network.set((64, 64), False)
        rl_config.multisteps = 1
        rl_config.enable_noisy_dense = True
        runner, tester = self.create_runner("Pendulum-v1", rl_config)
        runner.train(max_train_count=200 * 70)
        tester.eval(runner)

    def test_OX(self):
        self.check_skip()
        # invalid action test
        rl_config = self._create_rl_config()
        rl_config.dueling_network.set((64, 32, 16), False)
        rl_config.epsilon.set_constant(0)
        rl_config.multisteps = 3
        rl_config.memory.set_replay_memory()

        runner, tester = self.create_runner("OX", rl_config)
        runner.set_players([None, "random"])
        runner.train(max_train_count=10000)

        runner.set_players([None, "random"])
        tester.eval(runner, baseline=[0.4, None])
        runner.set_players(["random", None])
        tester.eval(runner, baseline=[None, 0.4])

    def _create_pendulum_config(self):
        rl_config = self._create_rl_config()
        rl_config.epsilon.set_constant(0.1)
        rl_config.discount = 0.9
        rl_config.lr.set_constant(0.001)
        rl_config.batch_size = 32
        rl_config.dueling_network.set((64, 64), False)
        rl_config.enable_double_dqn = False
        rl_config.enable_noisy_dense = False
        rl_config.multisteps = 1
        rl_config.memory.set_replay_memory()
        rl_config.enable_rescale = False
        rl_config.window_length = 1
        return rl_config

    def test_Pendulum_naive(self):
        self.check_skip()
        rl_config = self._create_pendulum_config()
        runner, tester = self.create_runner("Pendulum-v1", rl_config)
        runner.train(max_train_count=200 * 100)
        tester.eval(runner)

    def test_Pendulum_window_length(self):
        self.check_skip()
        rl_config = self._create_pendulum_config()
        rl_config.window_length = 4
        runner, tester = self.create_runner("Pendulum-v1", rl_config)
        runner.train(max_train_count=200 * 70)
        tester.eval(runner)

    def test_Pendulum_ddqn(self):
        self.check_skip()
        rl_config = self._create_pendulum_config()
        rl_config.enable_double_dqn = True
        runner, tester = self.create_runner("Pendulum-v1", rl_config)
        runner.train(max_train_count=200 * 80)
        tester.eval(runner)

    def test_Pendulum_dueling(self):
        self.check_skip()
        rl_config = self._create_pendulum_config()
        rl_config.dueling_network.set((64, 64), True)
        runner, tester = self.create_runner("Pendulum-v1", rl_config)
        runner.train(max_train_count=200 * 70)
        tester.eval(runner)

    def test_Pendulum_multistep(self):
        self.check_skip()
        rl_config = self._create_pendulum_config()
        rl_config.multisteps = 10
        runner, tester = self.create_runner("Pendulum-v1", rl_config)
        runner.train(max_train_count=200 * 80)
        tester.eval(runner)

    def test_Pendulum_proportional(self):
        self.check_skip()
        rl_config = self._create_pendulum_config()
        rl_config.memory.set_proportional_memory(alpha=1.0, beta_initial=1.0)
        runner, tester = self.create_runner("Pendulum-v1", rl_config)
        runner.train(max_train_count=200 * 120)
        tester.eval(runner)

    def test_Pendulum_rankbase(self):
        self.check_skip()
        rl_config = self._create_pendulum_config()
        rl_config.memory.set_rankbase_memory(alpha=1.0, beta_initial=1.0)
        runner, tester = self.create_runner("Pendulum-v1", rl_config)
        runner.train(max_train_count=200 * 120)
        tester.eval(runner)

    def test_Pendulum_rankbaseLinear(self):
        self.check_skip()
        rl_config = self._create_pendulum_config()
        rl_config.memory.set_rankbase_memory_linear(alpha=1.0, beta_initial=1.0)
        runner, tester = self.create_runner("Pendulum-v1", rl_config)
        runner.train(max_train_count=200 * 120)
        tester.eval(runner)

    def test_Pendulum_all(self):
        self.check_skip()

        rl_config = self._create_pendulum_config()
        rl_config.enable_double_dqn = True
        rl_config.dueling_network.set((64, 64), True)
        rl_config.enable_noisy_dense = True
        rl_config.multisteps = 10
        rl_config.memory.set_proportional_memory(alpha=1.0, beta_initial=1.0)
        runner, tester = self.create_runner("Pendulum-v1", rl_config)
        runner.train(max_train_count=200 * 100)
        tester.eval(runner)
